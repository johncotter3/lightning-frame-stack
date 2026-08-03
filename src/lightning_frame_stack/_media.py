"""GIF and video decoding plus automatic base-frame selection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from ._types import FrameStats, MediaInfo, ProgressCallback, _fail

GIF_SUFFIXES = {".gif"}


def inspect_media(path: str | Path) -> MediaInfo:
    """Read metadata from a GIF or OpenCV-supported video file."""

    media_path = Path(path)
    if not media_path.is_file():
        _fail(f"Input file does not exist: {media_path}")

    if media_path.suffix.lower() in GIF_SUFFIXES:
        try:
            with Image.open(media_path) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                image.seek(0)
                width, height = image.size
                duration_ms = float(image.info.get("duration", 100.0) or 100.0)
                fps = 1000.0 / max(duration_ms, 1.0)
        except (OSError, EOFError) as error:
            _fail(f"Pillow could not open the GIF: {media_path} ({error})")
        return MediaInfo(
            path=media_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            is_gif=True,
        )

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        _fail(f"OpenCV could not open the input: {media_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        _fail(f"Could not read valid media metadata from: {media_path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    return MediaInfo(
        path=media_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        is_gif=False,
    )


def frame_bounds(
    info: MediaInfo,
    start_seconds: float,
    end_seconds: float | None,
) -> tuple[int, int]:
    """Convert a time range to a half-open source-frame range."""

    if start_seconds < 0:
        _fail("start must be non-negative")
    if end_seconds is not None and end_seconds <= start_seconds:
        _fail("end must be greater than start")

    start_index = max(0, int(np.floor(start_seconds * info.fps)))
    end_index = info.frame_count
    if end_seconds is not None:
        end_index = min(info.frame_count, int(np.ceil(end_seconds * info.fps)))

    if start_index >= end_index:
        _fail("The requested time range contains no frames")
    return start_index, end_index


def _gif_frame_to_bgr(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, (0, 0, 0))
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def iter_frames(
    info: MediaInfo,
    start_index: int,
    end_index: int,
    every: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(source_index, BGR uint8 frame)`` pairs."""

    if every < 1:
        _fail("every must be at least 1")

    if info.is_gif:
        try:
            with Image.open(info.path) as image:
                for index in range(start_index, end_index, every):
                    image.seek(index)
                    yield index, _gif_frame_to_bgr(image)
        except (OSError, EOFError) as error:
            _fail(f"Could not decode GIF frame: {error}")
        return

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        _fail(f"Could not reopen input: {info.path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    index = start_index
    try:
        while index < end_index:
            ok, frame = capture.read()
            if not ok:
                break
            if (index - start_index) % every == 0:
                yield index, frame
            index += 1
    finally:
        capture.release()


def read_frame(info: MediaInfo, index: int) -> np.ndarray:
    """Decode a single source frame by zero-based index."""

    if index < 0 or index >= info.frame_count:
        _fail(
            f"Base frame {index} is outside the valid range "
            f"0..{info.frame_count - 1}"
        )

    if info.is_gif:
        try:
            with Image.open(info.path) as image:
                image.seek(index)
                return _gif_frame_to_bgr(image)
        except (OSError, EOFError) as error:
            _fail(f"Could not decode GIF base frame {index}: {error}")

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        _fail(f"Could not reopen input: {info.path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        _fail(f"Could not decode base frame {index}")
    return frame


def _resize_to_max(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    if scale >= 1.0:
        return image.copy()
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def analyze_frame(index: int, frame: np.ndarray) -> FrameStats:
    """Measure exposure, sharpness, and likely lightning detail in one frame."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = _resize_to_max(gray, 320)
    small_float = small.astype(np.float32)

    brightness = float(np.median(small_float))
    sharpness = float(cv2.Laplacian(small_float, cv2.CV_32F).var())

    local_background = cv2.GaussianBlur(small_float, (0, 0), 1.2)
    high_pass = small_float - local_background
    cutoff = max(1, int(round(small.shape[0] * 0.72)))
    roi_detail = high_pass[:cutoff, :]
    roi_gray = small_float[:cutoff, :]
    bright_detail_count = int(
        np.count_nonzero((roi_detail > 5.0) & (roi_gray > 90.0))
    )

    return FrameStats(
        index=index,
        brightness=brightness,
        sharpness=sharpness,
        bright_detail_count=bright_detail_count,
    )


def choose_base_frame(
    info: MediaInfo,
    start_index: int,
    end_index: int,
    every: int,
    progress: ProgressCallback | None = None,
) -> int:
    """Choose a relatively dark, sharp frame with little apparent lightning."""

    stats: list[FrameStats] = []
    for count, (index, frame) in enumerate(
        iter_frames(info, start_index, end_index, every), start=1
    ):
        stats.append(analyze_frame(index, frame))
        if progress is not None and count % 50 == 0:
            progress(f"Analyzed {count} frames for automatic base selection...")

    if not stats:
        _fail("No frames were decoded from the requested range")

    brightness = np.asarray([item.brightness for item in stats], dtype=np.float32)
    detail = np.asarray(
        [item.bright_detail_count for item in stats], dtype=np.float32
    )

    brightness_limit = float(np.percentile(brightness, 60.0))
    detail_limit = float(np.percentile(detail, 40.0))
    candidates = [
        item
        for item in stats
        if item.brightness <= brightness_limit
        and item.bright_detail_count <= detail_limit
    ]
    if not candidates:
        candidates = stats

    return max(candidates, key=lambda item: item.sharpness).index
