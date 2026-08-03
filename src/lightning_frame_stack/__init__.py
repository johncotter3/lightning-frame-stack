"""Create a single composite showing every lightning strike in an animation."""

from ._media import inspect_media
from ._types import (
    AlignMode,
    LightningStackError,
    MediaInfo,
    StackConfig,
    StackMode,
    StackResult,
)
from .core import stack_media

__version__ = "0.1.0"

__all__ = [
    "AlignMode",
    "LightningStackError",
    "MediaInfo",
    "StackConfig",
    "StackMode",
    "StackResult",
    "__version__",
    "inspect_media",
    "stack_media",
]
