"""Tests for the lightning stacking pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import make_test_gif

import lightning_frame_stack.core as core_module
from lightning_frame_stack import (
    LightningStackError,
    StackConfig,
    inspect_media,
    stack_media,
)
from lightning_frame_stack._stacker import LightningStacker


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


def test_long_thin_strike_is_not_rejected_as_a_sensor_artifact() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    strike = base.copy()
    points = np.asarray(
        [
            (40, 5),
            (32, 16),
            (46, 27),
            (34, 38),
            (48, 49),
            (33, 60),
            (45, 71),
            (35, 82),
            (43, 92),
        ]
    )
    cv2.polylines(strike, [points], False, (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
        ),
    )

    stacker.add(strike.astype(np.float32), np.ones(base.shape[:2], dtype=bool))

    assert int(stacker.result()[49, 48].max()) > 150


def test_straight_flare_connected_to_a_strike_is_removed() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    frame = base.copy()
    cv2.line(frame, (18, 4), (18, 72), (180, 180, 180), 2, cv2.LINE_8)
    points = np.asarray([(18, 60), (28, 66), (36, 62), (47, 75)])
    cv2.polylines(frame, [points], False, (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
        ),
    )

    stacker.add(frame.astype(np.float32), np.ones(base.shape[:2], dtype=bool))

    result = stacker.result()
    assert np.max(np.abs(result[20, 18].astype(int) - base[20, 18])) <= 2
    assert int(result[62, 36].max()) > 150


def test_detail_repeated_across_frames_is_suppressed() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    cv2.line(base, (18, 10), (18, 90), (90, 90, 90), 2, cv2.LINE_8)
    repeated = base.copy()
    cv2.line(repeated, (19, 10), (19, 90), (220, 220, 220), 2, cv2.LINE_8)
    strike = base.copy()
    points = np.asarray(
        [(58, 8), (50, 24), (66, 40), (49, 56), (67, 72), (54, 90)]
    )
    cv2.polylines(strike, [points], False, (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            reject_straight_artifacts=False,
            persistent_artifact_frames=4,
        ),
    )
    valid = np.ones(base.shape[:2], dtype=bool)

    for _ in range(4):
        stacker.add(repeated.astype(np.float32), valid)
    stacker.add(strike.astype(np.float32), valid)

    result = stacker.result()
    assert np.max(np.abs(result[50, 19].astype(int) - base[50, 19])) <= 2
    assert int(result[40, 66].max()) > 150


def test_repeated_strike_away_from_base_edges_is_retained() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    strike = base.copy()
    points = np.asarray(
        [(58, 8), (50, 24), (66, 40), (49, 56), (67, 72), (54, 90)]
    )
    cv2.polylines(strike, [points], False, (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            persistent_artifact_frames=4,
        ),
    )
    valid = np.ones(base.shape[:2], dtype=bool)

    for _ in range(4):
        stacker.add(strike.astype(np.float32), valid)

    assert int(stacker.result()[40, 66].max()) > 150


def test_horizontal_wires_do_not_mask_a_repeated_crossing_strike() -> None:
    base = np.full((120, 90, 3), 24, dtype=np.uint8)
    for y in (42, 50, 58):
        cv2.line(base, (5, y), (84, y), (90, 90, 90), 2, cv2.LINE_8)
    strike = base.copy()
    points = np.asarray([(48, 8), (45, 30), (50, 48), (46, 70), (51, 108)])
    cv2.polylines(strike, [points], False, (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            reject_straight_artifacts=False,
            persistent_artifact_frames=4,
        ),
    )
    valid = np.ones(base.shape[:2], dtype=bool)

    for _ in range(4):
        stacker.add(strike.astype(np.float32), valid)

    result = stacker.result()
    assert int(result[46, 49].max()) > 150
    assert int(result[54, 49].max()) > 150


def test_default_keeps_foreground_reflections() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    frame = base.copy()
    cv2.line(frame, (20, 8), (28, 45), (245, 245, 255), 2, cv2.LINE_8)
    cv2.line(frame, (52, 72), (60, 92), (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            reject_straight_artifacts=False,
        ),
    )

    stacker.add(frame.astype(np.float32), np.ones(base.shape[:2], dtype=bool))

    assert int(stacker.result()[82, 56].max()) > 150


def test_frame_without_a_sky_lightning_component_is_ignored() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    cv2.line(base, (18, 48), (18, 92), (90, 90, 90), 2, cv2.LINE_8)
    shifted_structure = base.copy()
    cv2.line(
        shifted_structure,
        (20, 48),
        (20, 92),
        (220, 220, 220),
        2,
        cv2.LINE_8,
    )
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            reject_straight_artifacts=False,
        ),
    )

    stacker.add(
        shifted_structure.astype(np.float32),
        np.ones(base.shape[:2], dtype=bool),
    )

    assert np.array_equal(stacker.result(), base)


def test_failed_alignment_frame_is_not_composited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "storm.gif"
    output_path = tmp_path / "stacked.png"
    frames = make_test_gif(input_path)

    class RejectingStabilizer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def align(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            valid = np.ones(frame.shape[:2], dtype=bool)
            return frame, valid, 0.0

    monkeypatch.setattr(core_module, "Stabilizer", RejectingStabilizer)

    result = stack_media(
        input_path,
        output_path,
        config=StackConfig(base_frame=0, align="euclidean", exposure_match=False),
    )

    stacked = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert stacked is not None
    assert result.failed_alignments == len(frames)
    assert np.array_equal(stacked, frames[0])


def test_lightning_to_excludes_foreground_reflections() -> None:
    base = np.full((100, 80, 3), 24, dtype=np.uint8)
    frame = base.copy()
    cv2.line(frame, (20, 8), (28, 45), (245, 245, 255), 2, cv2.LINE_8)
    cv2.line(frame, (52, 72), (60, 92), (245, 245, 255), 2, cv2.LINE_8)
    stacker = LightningStacker(
        base,
        StackConfig(
            exposure_match=False,
            detail_sigma=2.0,
            threshold=2.0,
            min_difference=1.0,
            min_brightness=35.0,
            dilation=3,
            softness=3.0,
            lightning_to=0.60,
            reject_straight_artifacts=False,
        ),
    )

    stacker.add(frame.astype(np.float32), np.ones(base.shape[:2], dtype=bool))

    result = stacker.result()
    assert int(result[30, 25].max()) > 150
    assert np.max(np.abs(result[82, 56].astype(int) - base[82, 56])) <= 2


def test_invalid_time_range_raises(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    make_test_gif(input_path)

    with pytest.raises(LightningStackError, match="end must be greater than start"):
        stack_media(
            input_path,
            tmp_path / "unused.png",
            config=StackConfig(start=2.0, end=1.0),
        )
