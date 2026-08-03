#!/usr/bin/env python3
"""Generate synthetic media for testing and demonstrating the stacker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lightning_frame_stack import StackConfig, stack_media

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
WIDTH = 360
HEIGHT = 220


def make_background() -> np.ndarray:
    """Create a deterministic stylized night-sky background in BGR format."""

    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    top = np.array([42, 24, 22], dtype=np.float32)
    bottom = np.array([72, 42, 34], dtype=np.float32)
    gradient = top[None, None, :] * (1.0 - y[:, :, None])
    gradient += bottom[None, None, :] * y[:, :, None]
    image = np.repeat(gradient, WIDTH, axis=1)

    cloud_layer = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for center, axes, value in [
        ((70, 72), (88, 28), 18),
        ((178, 58), (105, 34), 22),
        ((292, 82), (95, 31), 16),
        ((225, 115), (120, 26), 12),
    ]:
        cv2.ellipse(cloud_layer, center, axes, 0, 0, 360, value, -1)
    cloud_layer = cv2.GaussianBlur(cloud_layer, (0, 0), 18)
    image += cloud_layer[:, :, None]

    image = np.clip(image, 0, 255).astype(np.uint8)

    mountains = np.array(
        [
            [0, 175],
            [52, 144],
            [105, 170],
            [170, 132],
            [235, 169],
            [302, 142],
            [359, 166],
            [359, 220],
            [0, 220],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [mountains], (12, 12, 17))
    cv2.rectangle(image, (0, 184), (WIDTH, HEIGHT), (8, 9, 13), -1)

    for x, intensity in [(22, 155), (85, 190), (133, 145), (252, 180), (326, 150)]:
        cv2.circle(image, (x, 185), 1, (intensity, intensity, 215), -1)

    return image


def draw_lightning(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    branches: list[list[tuple[int, int]]] | None = None,
    flash: float = 0.0,
) -> np.ndarray:
    """Add a glowing branched lightning strike and optional broad flash."""

    result = frame.astype(np.float32)
    if flash:
        glow_center = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        cv2.circle(glow_center, points[0], 105, flash, -1)
        glow_center = cv2.GaussianBlur(glow_center, (0, 0), 55)
        result += glow_center[:, :, None] * np.array([0.75, 0.82, 1.0])

    all_paths = [points, *(branches or [])]
    glow_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    core_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for path in all_paths:
        polyline = np.asarray(path, dtype=np.int32)
        cv2.polylines(glow_mask, [polyline], False, 255, 5, cv2.LINE_AA)
        cv2.polylines(core_mask, [polyline], False, 255, 1, cv2.LINE_AA)

    glow = cv2.GaussianBlur(glow_mask, (0, 0), 4.0).astype(np.float32) / 255.0
    core = core_mask.astype(np.float32) / 255.0
    result += glow[:, :, None] * np.array([75.0, 88.0, 125.0])
    result += core[:, :, None] * np.array([190.0, 205.0, 235.0])
    return np.clip(result, 0, 255).astype(np.uint8)


def build_frames() -> list[np.ndarray]:
    """Create separate frames containing three distinct strikes."""

    base = make_background()
    strike_a = draw_lightning(
        base,
        [(92, 18), (86, 43), (95, 61), (82, 87), (89, 112), (72, 145)],
        branches=[[(91, 61), (111, 72), (119, 91)], [(83, 88), (62, 99), (53, 118)]],
        flash=72,
    )
    strike_b = draw_lightning(
        base,
        [(185, 7), (173, 34), (183, 55), (166, 76), (177, 96), (158, 133)],
        branches=[[(178, 54), (202, 68), (211, 88)], [(168, 76), (147, 84), (137, 105)]],
        flash=88,
    )
    strike_c = draw_lightning(
        base,
        [(286, 24), (275, 48), (283, 68), (267, 92), (275, 113), (254, 147)],
        branches=[[(279, 67), (302, 78), (311, 97)], [(268, 92), (247, 102), (239, 123)]],
        flash=64,
    )

    broad_flash = np.clip(base.astype(np.int16) + 18, 0, 255).astype(np.uint8)
    return [base, strike_a, base, strike_b, broad_flash, base, strike_c, base]


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    rgb_frames = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
    ]
    rgb_frames[0].save(
        path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=350,
        loop=0,
        optimize=False,
    )


def save_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    selected = [frames[1], frames[3], frames[6]]
    labeled: list[np.ndarray] = []
    for index, frame in enumerate(selected, start=1):
        tile = frame.copy()
        cv2.putText(
            tile,
            f"Strike {index}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        labeled.append(tile)
    cv2.imwrite(str(path), np.concatenate(labeled, axis=1))


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    input_path = ASSET_DIR / "synthetic_input.gif"
    output_path = ASSET_DIR / "synthetic_stacked.png"
    mask_path = ASSET_DIR / "synthetic_mask.png"

    save_gif(frames, input_path)
    save_contact_sheet(frames, ASSET_DIR / "synthetic_strikes.png")

    stack_media(
        input_path,
        output_path,
        config=StackConfig(
            base_frame=0,
            align="none",
            exposure_match=True,
            detail_sigma=3.0,
            threshold=2.5,
            min_difference=1.0,
            min_brightness=45.0,
            dilation=5,
            softness=4.0,
            lightning_gain=1.05,
        ),
        mask_output_path=mask_path,
        progress=print,
    )
    print(f"Wrote demo assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
