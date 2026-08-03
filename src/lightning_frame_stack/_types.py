"""Public types and validation models used by the stacking pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

AlignMode = Literal["none", "translation", "euclidean", "affine"]
StackMode = Literal["lightning", "max"]
ProgressCallback = Callable[[str], None]


class LightningStackError(RuntimeError):
    """Raised when media cannot be read, processed, or written."""


@dataclass(frozen=True)
class MediaInfo:
    """Basic metadata for an input animation or video."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    is_gif: bool


@dataclass(frozen=True)
class FrameStats:
    """Statistics used while selecting an automatic base frame."""

    index: int
    brightness: float
    sharpness: float
    bright_detail_count: int


@dataclass(frozen=True)
class StackConfig:
    """Configuration for :func:`stack_media`.

    Defaults are tuned for typical handheld phone footage of thunderstorms.
    """

    mode: StackMode = "lightning"
    base_frame: int | None = None
    every: int = 1
    start: float = 0.0
    end: float | None = None

    align: AlignMode = "euclidean"
    align_scale: float = 0.35
    align_from: float = 0.48
    min_alignment_score: float = 0.50
    exposure_match: bool = True

    detail_sigma: float = 4.0
    threshold: float = 5.0
    min_difference: float = 2.5
    min_brightness: float = 78.0
    dilation: int = 5
    softness: float = 8.0
    lightning_gain: float = 1.0
    reject_straight_artifacts: bool = True

    def validate(self) -> None:
        """Validate configuration values before processing begins."""

        if self.mode not in {"lightning", "max"}:
            _fail(f"Unsupported mode: {self.mode}")
        if self.align not in {"none", "translation", "euclidean", "affine"}:
            _fail(f"Unsupported alignment mode: {self.align}")
        if self.every < 1:
            _fail("every must be at least 1")
        if self.start < 0:
            _fail("start must be non-negative")
        if self.end is not None and self.end <= self.start:
            _fail("end must be greater than start")
        if not 0.05 <= self.align_scale <= 1.0:
            _fail("align_scale must be between 0.05 and 1.0")
        if not 0.0 <= self.align_from < 1.0:
            _fail("align_from must be in the range [0.0, 1.0)")
        if not 0.0 <= self.min_alignment_score <= 1.0:
            _fail("min_alignment_score must be in the range [0.0, 1.0]")
        if self.detail_sigma <= 0:
            _fail("detail_sigma must be positive")
        if self.threshold < 0 or self.min_difference < 0:
            _fail("threshold and min_difference must be non-negative")
        if not 0.0 <= self.min_brightness <= 255.0:
            _fail("min_brightness must be in the range [0, 255]")
        if self.dilation < 1:
            _fail("dilation must be at least 1")
        if self.softness <= 0:
            _fail("softness must be positive")
        if self.lightning_gain <= 0:
            _fail("lightning_gain must be positive")

    @property
    def effective_dilation(self) -> int:
        """Return an odd dilation diameter accepted by OpenCV."""

        return self.dilation if self.dilation % 2 else self.dilation + 1


@dataclass(frozen=True)
class StackResult:
    """Summary returned after a successful stack operation."""

    input_path: Path
    output_path: Path
    mask_output_path: Path | None
    base_frame_index: int
    processed_frames: int
    failed_alignments: int
    media: MediaInfo


def _fail(message: str) -> NoReturn:
    raise LightningStackError(message)
