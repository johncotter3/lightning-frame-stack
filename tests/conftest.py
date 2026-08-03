"""Shared synthetic-media fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_test_gif(path: Path) -> list[np.ndarray]:
    """Write a small GIF with two separate zig-zag lightning strikes."""

    height, width = 72, 96
    base = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(base, (0, 57), (width, height), (12, 14, 16), -1)
    cv2.line(base, (0, 56), (width, 56), (42, 42, 46), 1)

    strike_left = base.copy()
    cv2.polylines(
        strike_left,
        [np.asarray([(25, 8), (20, 23), (28, 34), (19, 50)], dtype=np.int32)],
        False,
        (245, 245, 255),
        2,
        cv2.LINE_AA,
    )

    strike_right = base.copy()
    cv2.polylines(
        strike_right,
        [np.asarray([(72, 6), (66, 21), (75, 33), (64, 52)], dtype=np.int32)],
        False,
        (235, 240, 255),
        2,
        cv2.LINE_AA,
    )

    frames = [base, strike_left, base, strike_right, base]
    images = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    return frames
