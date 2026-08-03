"""Create a single composite showing every lightning strike in an animation."""

from .core import (
    AlignMode,
    LightningStackError,
    MediaInfo,
    StackConfig,
    StackMode,
    StackResult,
    inspect_media,
    stack_media,
)

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
