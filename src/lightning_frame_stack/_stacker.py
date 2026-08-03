"""Lightning-detail extraction and per-pixel accumulation."""

from __future__ import annotations

import cv2
import numpy as np

from ._types import StackConfig


class LightningStacker:
    """Accumulate either extracted lightning detail or a normalized max stack."""

    def __init__(self, base_frame: np.ndarray, config: StackConfig) -> None:
        self.mode = config.mode
        self.base = base_frame.astype(np.float32)
        self.base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        self.height, self.width = self.base_gray.shape
        self.detail_sigma = config.detail_sigma
        self.threshold = config.threshold
        self.min_difference = config.min_difference
        self.min_brightness = config.min_brightness
        self.softness = config.softness
        self.lightning_gain = config.lightning_gain
        self.reject_straight_artifacts = config.reject_straight_artifacts
        self.base_high_pass = self.base_gray - cv2.GaussianBlur(
            self.base_gray, (0, 0), self.detail_sigma
        )
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.effective_dilation, config.effective_dilation),
        )
        self.max_delta = np.zeros_like(self.base, dtype=np.float32)
        self.max_mask = np.zeros_like(self.base_gray, dtype=np.float32)
        self.max_stack = self.base.copy()

    def _filter_seed_components(self, seed: np.ndarray) -> np.ndarray:
        neighbor_count = cv2.boxFilter(
            seed.astype(np.uint8),
            cv2.CV_16U,
            (3, 3),
            normalize=False,
        )
        seed = seed & (neighbor_count >= 2)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            seed.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(component_count, dtype=np.uint8)

        for label in range(1, component_count):
            x, _y, width, height, area = stats[label]
            if area < 2:
                continue

            if self.reject_straight_artifacts:
                fill = float(area) / float(max(1, width * height))
                aspect = float(height) / float(max(1, width))
                long_narrow_straight = (
                    height > self.height * 0.08 and aspect > 12.0 and fill > 0.20
                )
                touches_side = x <= 2 or x + width >= self.width - 2
                side_flare = (
                    touches_side and height > self.height * 0.10 and aspect > 5.0
                )
                if long_narrow_straight or side_flare:
                    continue

            keep[label] = 1

        return keep[labels].astype(bool)

    def add(self, normalized_frame: np.ndarray, valid: np.ndarray) -> None:
        """Add one aligned, exposure-normalized frame to the stack."""

        if self.mode == "max":
            candidate = normalized_frame.copy()
            candidate[~valid] = self.base[~valid]
            self.max_stack = np.maximum(self.max_stack, candidate)
            return

        frame_uint8 = np.clip(normalized_frame, 0.0, 255.0).astype(np.uint8)
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        difference = gray - self.base_gray
        high_pass = gray - cv2.GaussianBlur(gray, (0, 0), self.detail_sigma)
        transient_detail = high_pass - self.base_high_pass

        seed = (
            (transient_detail > self.threshold)
            & (difference > self.min_difference)
            & (gray > self.min_brightness)
            & valid
        )
        seed = self._filter_seed_components(seed)
        if not np.any(seed):
            return

        neighborhood = cv2.dilate(
            seed.astype(np.uint8), self.dilation_kernel
        ) > 0
        mask = np.clip(
            (transient_detail - self.threshold) / self.softness,
            0.0,
            1.0,
        )
        mask *= neighborhood.astype(np.float32)
        mask *= valid.astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.65)
        mask = np.clip(mask, 0.0, 1.0)

        positive_delta = np.maximum(normalized_frame - self.base, 0.0)
        candidate_delta = (
            positive_delta * mask[:, :, np.newaxis] * self.lightning_gain
        )
        self.max_delta = np.maximum(self.max_delta, candidate_delta)
        self.max_mask = np.maximum(self.max_mask, mask)

    def result(self) -> np.ndarray:
        """Return the final BGR uint8 composite."""

        if self.mode == "max":
            return np.clip(self.max_stack, 0.0, 255.0).astype(np.uint8)
        return np.clip(self.base + self.max_delta, 0.0, 255.0).astype(np.uint8)

    def mask_image(self) -> np.ndarray:
        """Return the accumulated lightning confidence mask as uint8."""

        return np.clip(self.max_mask * 255.0, 0.0, 255.0).astype(np.uint8)
