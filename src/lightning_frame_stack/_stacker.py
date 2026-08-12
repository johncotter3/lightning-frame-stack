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
        self.lightning_to = config.lightning_to
        self.reject_straight_artifacts = config.reject_straight_artifacts
        self.persistent_artifact_frames = config.persistent_artifact_frames
        self.base_high_pass = self.base_gray - cv2.GaussianBlur(
            self.base_gray, (0, 0), self.detail_sigma
        )
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.effective_dilation, config.effective_dilation),
        )
        base_edges = cv2.Canny(self.base_gray.astype(np.uint8), 12, 36)
        self.base_structure_mask = cv2.dilate(
            base_edges,
            self.dilation_kernel,
        ) > 0
        self.extraction_weights = self._build_extraction_weights()
        self.max_delta = np.zeros_like(self.base, dtype=np.float32)
        self.max_mask = np.zeros_like(self.base_gray, dtype=np.float32)
        self.seed_hits = np.zeros_like(self.base_gray, dtype=np.uint32)
        self.protected_lightning_mask = np.zeros_like(self.base_gray, dtype=bool)
        self.max_stack = self.base.copy()

    def _build_extraction_weights(self) -> np.ndarray:
        weights = np.ones((self.height, self.width), dtype=np.float32)
        if self.lightning_to == 1.0:
            return weights

        cutoff = int(round(self.height * self.lightning_to))
        fade = min(16, max(4, int(round(self.height * 0.0125))))
        fade_start = max(0, cutoff - fade)
        weights[cutoff:, :] = 0.0
        weights[fade_start:cutoff, :] = np.linspace(
            1.0,
            0.0,
            cutoff - fade_start,
            endpoint=False,
            dtype=np.float32,
        )[:, np.newaxis]
        return weights

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
                solid_vertical_bar = (
                    height > self.height * 0.08
                    and width >= 4
                    and aspect > 12.0
                    and fill > 0.80
                )
                touches_side = x <= 2 or x + width >= self.width - 2
                side_flare = (
                    touches_side and height > self.height * 0.10 and aspect > 5.0
                )
                if solid_vertical_bar or side_flare:
                    continue

            keep[label] = 1

        return keep[labels].astype(bool)

    def _lightning_component_mask(self, seed: np.ndarray) -> np.ndarray:
        """Return plausible sky-lightning components used to admit a frame."""

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            seed.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(component_count, dtype=np.uint8)
        min_height = max(12, int(round(self.height * 0.03)))
        min_area = max(4, min_height // 3)
        sky_limit = int(round(self.height * 0.45))

        for label in range(1, component_count):
            _x, y, width, height, area = stats[label]
            aspect = height / max(1, width)
            straight_sensor_line = (
                height > self.height * 0.08 and width <= 3 and aspect > 20.0
            )
            if (
                y < sky_limit
                and height >= min_height
                and area >= min_area
                and not straight_sensor_line
            ):
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

        lightning_components = self._lightning_component_mask(seed)
        if not np.any(lightning_components):
            return
        self.protected_lightning_mask |= cv2.dilate(
            lightning_components.astype(np.uint8), self.dilation_kernel
        ) > 0
        self.seed_hits += seed.astype(np.uint32)

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
        mask *= self.extraction_weights
        mask = cv2.GaussianBlur(mask, (0, 0), 0.65)
        mask = np.clip(mask, 0.0, 1.0)

        positive_delta = np.maximum(normalized_frame - self.base, 0.0)
        candidate_delta = (
            positive_delta * mask[:, :, np.newaxis] * self.lightning_gain
        )
        self.max_delta = np.maximum(self.max_delta, candidate_delta)
        self.max_mask = np.maximum(self.max_mask, mask)

    def _persistent_artifact_mask(self) -> np.ndarray | None:
        if self.persistent_artifact_frames == 0:
            return None

        persistent = self.seed_hits >= self.persistent_artifact_frames
        if not np.any(persistent):
            return None
        persistent = cv2.dilate(
            persistent.astype(np.uint8), self.dilation_kernel
        ) > 0
        return persistent & self.base_structure_mask

    def _vertical_artifact_mask(self) -> np.ndarray | None:
        if self.persistent_artifact_frames == 0:
            return None

        vertical_source = (self.max_mask > 0.015) & self.base_structure_mask
        vertical_length = max(12, int(round(self.height * 0.01)))
        vertical = cv2.morphologyEx(
            vertical_source.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((vertical_length, 1), dtype=np.uint8),
        )
        if not np.any(vertical):
            return None
        return cv2.dilate(vertical, self.dilation_kernel) > 0

    def _straight_artifact_mask(self) -> np.ndarray | None:
        if not self.reject_straight_artifacts:
            return None

        vertical_length = max(12, int(round(self.height * 0.12)))
        vertical_kernel = np.ones((vertical_length, 1), dtype=np.uint8)
        opened = cv2.morphologyEx(
            (self.max_mask > 0.015).astype(np.uint8),
            cv2.MORPH_OPEN,
            vertical_kernel,
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened,
            connectivity=8,
        )
        keep = np.zeros(component_count, dtype=np.uint8)
        for label in range(1, component_count):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if height / max(1, width) > 12.0:
                keep[label] = 1

        straight = keep[labels]
        if not np.any(straight):
            return None
        return cv2.dilate(straight, self.dilation_kernel) > 0

    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        scene_masks = (
            self._persistent_artifact_mask(),
            self._vertical_artifact_mask(),
        )
        present_scene_masks = [mask for mask in scene_masks if mask is not None]
        straight = self._straight_artifact_mask()
        if not present_scene_masks and straight is None:
            return image

        artifacts = np.zeros_like(self.base_gray, dtype=bool)
        if present_scene_masks:
            scene_artifacts = np.logical_or.reduce(present_scene_masks)
            scene_artifacts &= ~self.protected_lightning_mask
            artifacts |= scene_artifacts
        if straight is not None:
            # Straight sensor lines remain artifacts even when they are attached
            # to, or overlap, a genuine lightning component.
            artifacts |= straight

        cleaned = image.copy()
        cleaned[artifacts] = 0.0
        return cleaned

    def result(self) -> np.ndarray:
        """Return the final BGR uint8 composite."""

        if self.mode == "max":
            return np.clip(self.max_stack, 0.0, 255.0).astype(np.uint8)
        delta = self._remove_artifacts(self.max_delta)
        return np.clip(self.base + delta, 0.0, 255.0).astype(np.uint8)

    def mask_image(self) -> np.ndarray:
        """Return the accumulated lightning confidence mask as uint8."""

        mask = self._remove_artifacts(self.max_mask)
        return np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
