"""Public orchestration API for lightning frame stacking.

The implementation is split into focused internal modules for media decoding,
alignment, exposure matching, and lightning-detail accumulation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._alignment import ExposureMatcher, Stabilizer
from ._media import (
    choose_base_frame,
    frame_bounds,
    inspect_media,
    iter_frames,
    read_frame,
)
from ._stacker import LightningStacker
from ._types import ProgressCallback, StackConfig, StackResult, _fail


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        _fail(f"OpenCV could not write output image: {path}")


def stack_media(
    input_path: str | Path,
    output_path: str | Path = "lightning_stacked.png",
    *,
    config: StackConfig | None = None,
    mask_output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> StackResult:
    """Stack lightning from a GIF or video into one still image.

    Args:
        input_path: GIF or OpenCV-supported video file.
        output_path: Destination image. PNG is recommended.
        config: Processing options. Defaults to :class:`StackConfig`.
        mask_output_path: Optional grayscale confidence-mask destination.
        progress: Optional callback receiving human-readable progress messages.

    Returns:
        A :class:`StackResult` with processing metadata.

    Raises:
        LightningStackError: If the input, settings, decoding, or output is invalid.
    """

    settings = config or StackConfig()
    settings.validate()

    info = inspect_media(input_path)
    start_index, end_index = frame_bounds(info, settings.start, settings.end)

    if settings.base_frame is None:
        if progress is not None:
            progress("Selecting a clean base frame...")
        base_index = choose_base_frame(
            info,
            start_index,
            end_index,
            settings.every,
            progress,
        )
        if progress is not None:
            progress(f"Selected base frame {base_index}.")
    else:
        base_index = settings.base_frame
        if base_index < start_index or base_index >= end_index:
            _fail(
                f"Base frame {base_index} is outside the requested frame range "
                f"[{start_index}, {end_index})"
            )

    base_frame = read_frame(info, base_index)
    if base_frame.shape[1] != info.width or base_frame.shape[0] != info.height:
        _fail("Decoded frame dimensions changed unexpectedly")

    stabilizer = Stabilizer(
        base_frame=base_frame,
        mode=settings.align,
        scale=settings.align_scale,
        align_from=settings.align_from,
        min_score=settings.min_alignment_score,
    )
    exposure = ExposureMatcher(
        base_frame=base_frame,
        enabled=settings.exposure_match,
    )
    stacker = LightningStacker(base_frame=base_frame, config=settings)

    expected = len(range(start_index, end_index, settings.every))
    processed = 0
    failed_alignments = 0

    for source_index, frame in iter_frames(
        info,
        start_index,
        end_index,
        settings.every,
    ):
        aligned, valid, alignment_score = stabilizer.align(frame)
        if settings.align != "none" and alignment_score == 0.0:
            failed_alignments += 1
        normalized, _gain, _offset = exposure.match(aligned, valid)
        stacker.add(normalized, valid)
        processed += 1

        if progress is not None and (processed % 25 == 0 or processed == expected):
            progress(
                f"Processed {processed}/{expected} frames "
                f"(source frame {source_index})..."
            )

    if processed == 0:
        _fail("No frames were processed")

    destination = Path(output_path)
    _write_image(destination, stacker.result())

    mask_destination = Path(mask_output_path) if mask_output_path is not None else None
    if mask_destination is not None:
        _write_image(mask_destination, stacker.mask_image())

    return StackResult(
        input_path=Path(input_path),
        output_path=destination,
        mask_output_path=mask_destination,
        base_frame_index=base_index,
        processed_frames=processed,
        failed_alignments=failed_alignments,
        media=info,
    )
