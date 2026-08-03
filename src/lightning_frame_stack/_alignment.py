"""Frame stabilization and robust global exposure matching."""

from __future__ import annotations

import cv2
import numpy as np

from ._media import _resize_to_max
from ._types import AlignMode


class Stabilizer:
    """Estimate and apply frame-to-base affine transforms using OpenCV ECC."""

    def __init__(
        self,
        base_frame: np.ndarray,
        mode: AlignMode,
        scale: float,
        align_from: float,
        min_score: float,
    ) -> None:
        self.base = base_frame
        self.mode = mode
        self.scale = scale
        self.min_score = min_score
        self.height, self.width = base_frame.shape[:2]

        motion_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
        }
        self.motion_type = motion_map.get(mode)

        scaled_width = max(32, int(round(self.width * scale)))
        scaled_height = max(32, int(round(self.height * scale)))
        self.scaled_size = (scaled_width, scaled_height)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.base_prepared = self._prepare(base_frame)

        self.alignment_mask = np.zeros(self.base_prepared.shape, dtype=np.uint8)
        y0 = int(round(self.base_prepared.shape[0] * align_from))
        self.alignment_mask[y0:, :] = 255
        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            60,
            1e-5,
        )

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.scaled_size, interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(gray)
        return gray.astype(np.float32) / 255.0

    def align(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return aligned frame, valid-pixel mask, and ECC score."""

        if self.mode == "none":
            valid = np.ones((self.height, self.width), dtype=bool)
            return frame, valid, 1.0

        prepared = self._prepare(frame)
        warp = np.eye(2, 3, dtype=np.float32)
        score = 0.0

        try:
            try:
                score, warp = cv2.findTransformECC(
                    self.base_prepared,
                    prepared,
                    warp,
                    self.motion_type,
                    self.criteria,
                    self.alignment_mask,
                    5,
                )
            except TypeError:
                score, warp = cv2.findTransformECC(
                    self.base_prepared,
                    prepared,
                    warp,
                    self.motion_type,
                    self.criteria,
                    self.alignment_mask,
                )
        except cv2.error:
            score = 0.0
            warp = np.eye(2, 3, dtype=np.float32)

        if not np.isfinite(score) or score < self.min_score:
            score = 0.0
            warp = np.eye(2, 3, dtype=np.float32)

        full_warp = warp.copy()
        full_warp[:, 2] /= self.scale

        aligned = cv2.warpAffine(
            frame,
            full_warp,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT101,
        )

        valid_source = np.full((self.height, self.width), 255, dtype=np.uint8)
        valid = cv2.warpAffine(
            valid_source,
            full_warp,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned, valid > 0, float(score)


class ExposureMatcher:
    """Robustly fit a global gain and offset to the base frame exposure."""

    def __init__(self, base_frame: np.ndarray, enabled: bool = True) -> None:
        self.enabled = enabled
        base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.base_small = _resize_to_max(base_gray, 320)
        self.quantiles = np.asarray([5, 20, 35, 50, 65, 80], dtype=np.float32)
        self.base_valid = (self.base_small > 5.0) & (self.base_small < 245.0)
        if np.count_nonzero(self.base_valid) < 100:
            self.base_valid = np.ones_like(self.base_small, dtype=bool)
        self.base_quantiles = np.percentile(
            self.base_small[self.base_valid], self.quantiles
        )

    def match(
        self,
        frame: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return exposure-normalized float32 BGR data, gain, and offset."""

        if not self.enabled:
            return frame.astype(np.float32), 1.0, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        small = cv2.resize(
            gray,
            (self.base_small.shape[1], self.base_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        valid_small = cv2.resize(
            valid.astype(np.uint8),
            (self.base_small.shape[1], self.base_small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        usable = self.base_valid & valid_small & (small > 5.0) & (small < 245.0)
        if np.count_nonzero(usable) < 100:
            return frame.astype(np.float32), 1.0, 0.0

        frame_quantiles = np.percentile(small[usable], self.quantiles)
        design = np.column_stack([frame_quantiles, np.ones_like(frame_quantiles)])
        gain, offset = np.linalg.lstsq(
            design, self.base_quantiles, rcond=None
        )[0]
        gain = float(np.clip(gain, 0.35, 2.50))
        offset = float(np.clip(offset, -100.0, 100.0))

        normalized = np.clip(
            frame.astype(np.float32) * gain + offset,
            0.0,
            255.0,
        )
        return normalized, gain, offset
