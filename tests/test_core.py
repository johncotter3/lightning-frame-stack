"""Tests for the lightning stacking pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lightning_frame_stack import (
    LightningStackError,
    StackConfig,
    inspect_media,
    stack_media,
)

from conftest import make_test_gif


def test_inspect_media_reads_gif(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    make_test_gif(input_path)

    info = inspect_media(input_path)

    assert info.is_gif is True
    assert info.frame_count == 5
    assert info.width == 96
    assert info.height == 72
    assert info.fps == pytest.approx(10.0)


def test_lightning_mode_combines_separate_strikes(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    output_path = tmp_path / "stacked.png"
    mask_path = tmp_path / "mask.png"
    frames = make_test_gif(input_path)

    result = stack_media(
        input_path,
        output_path,
        config=StackConfig(
            base_frame=0,
            align="none",
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
        ),
        mask_output_path=mask_path,
    )

    stacked = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    assert result.processed_frames == 5
    assert result.base_frame_index == 0
    assert stacked is not None
    assert mask is not None
    assert stacked.shape == frames[0].shape

    # Each strike should survive in the same output image.
    assert int(stacked[34, 28].max()) > 150
    assert int(stacked[33, 75].max()) > 150
    assert int(mask[34, 28]) > 0
    assert int(mask[33, 75]) > 0

    # An untouched background pixel should remain close to the base frame.
    assert np.max(np.abs(stacked[10, 48].astype(int) - frames[0][10, 48])) <= 2


def test_max_mode_keeps_brightest_pixels(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    output_path = tmp_path / "max.png"
    make_test_gif(input_path)

    stack_media(
        input_path,
        output_path,
        config=StackConfig(
            mode="max",
            base_frame=0,
            align="none",
            exposure_match=False,
        ),
    )

    stacked = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert stacked is not None
    assert int(stacked[34, 28].max()) > 200
    assert int(stacked[33, 75].max()) > 200


def test_automatic_base_selection_returns_valid_frame(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    output_path = tmp_path / "automatic.png"
    make_test_gif(input_path)

    result = stack_media(
        input_path,
        output_path,
        config=StackConfig(
            align="none",
            exposure_match=False,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
        ),
    )

    assert 0 <= result.base_frame_index < 5
    assert output_path.is_file()


def test_even_dilation_is_normalized() -> None:
    assert StackConfig(dilation=4).effective_dilation == 5
    assert StackConfig(dilation=5).effective_dilation == 5


def test_invalid_time_range_raises(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    make_test_gif(input_path)

    with pytest.raises(LightningStackError, match="end must be greater than start"):
        stack_media(
            input_path,
            tmp_path / "unused.png",
            config=StackConfig(start=2.0, end=1.0),
        )
