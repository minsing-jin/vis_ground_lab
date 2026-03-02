"""AutoLabeler: fuse multiple labeling signals into a single LabelResult."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from data_harvest.core.config import LabelerConfig
from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    BBoxCandidate,
    HarvestSample,
    LabelResult,
)
from data_harvest.labeler.click_proximity import click_proximity_bbox
from data_harvest.labeler.diff_detector import diff_bboxes
from data_harvest.labeler.ocr_extractor import ocr_bboxes, ocr_nearest_to_click
from data_harvest.labeler.transition_detector import is_screen_transition

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Fuses click_proximity, diff, OCR, and profile signals into a LabelResult."""

    def __init__(self, config: LabelerConfig, profile_hints: dict[str, list[BBoxCandidate]] | None = None) -> None:
        self.config = config
        self.profile_hints = profile_hints or {}

    def label_sample(self, sample: HarvestSample) -> LabelResult | None:
        """Run all signals on a sample and fuse them into a LabelResult."""
        if sample.event is None:
            return None

        pre_path = sample.pre_frame_path
        post_path = sample.post_frame_path
        if not pre_path.exists() or not post_path.exists():
            return None

        pre = cv2.imread(str(pre_path))
        post = cv2.imread(str(post_path))
        if pre is None or post is None:
            return None

        event = sample.event
        candidates: list[BBoxCandidate] = []
        weights = self.config.fusion_weights

        # 1. Click proximity
        if event.action in (ActionType.click, ActionType.drag) and event.x is not None and event.y is not None:
            cp = click_proximity_bbox(
                pre,
                event.x,
                event.y,
                crop_radius=self.config.click_crop_radius_px,
                contour_min_area=self.config.contour_min_area_px,
            )
            if cp is not None:
                candidates.append(cp)

        # 2. Diff detector
        diffs = diff_bboxes(
            pre, post,
            threshold=self.config.diff_threshold,
            contour_min_area=self.config.contour_min_area_px,
        )
        candidates.extend(diffs)

        # 3. OCR
        ocr_cands = ocr_bboxes(
            pre,
            languages=self.config.ocr_languages,
            gpu=self.config.ocr_gpu,
        )
        if event.x is not None and event.y is not None:
            nearest_ocr = ocr_nearest_to_click(ocr_cands, event.x, event.y)
            if nearest_ocr is not None:
                candidates.append(nearest_ocr)

        # 4. Transition detection
        transition = is_screen_transition(pre, post, max_diff_ratio=0.4)

        if not candidates:
            # Fallback: use a small bbox around the click
            if event.x is not None and event.y is not None:
                r = self.config.click_crop_radius_px // 2
                h, w = pre.shape[:2]
                return LabelResult(
                    bbox_x_min=max(0.0, event.x - r),
                    bbox_y_min=max(0.0, event.y - r),
                    bbox_x_max=min(float(w), event.x + r),
                    bbox_y_max=min(float(h), event.y + r),
                    confidence=0.1,
                    candidates=[],
                    transition_detected=transition,
                )
            return None

        # Weighted fusion
        return self._fuse(candidates, weights, transition)

    def _fuse(
        self,
        candidates: list[BBoxCandidate],
        weights: dict[str, float],
        transition: bool,
    ) -> LabelResult:
        """Weighted average of candidate bboxes to produce a fused result."""
        total_w = 0.0
        wx1, wy1, wx2, wy2 = 0.0, 0.0, 0.0, 0.0
        best_semantic: str | None = None
        best_conf = 0.0

        for c in candidates:
            w = weights.get(c.signal, 0.1) * c.confidence
            wx1 += c.x_min * w
            wy1 += c.y_min * w
            wx2 += c.x_max * w
            wy2 += c.y_max * w
            total_w += w
            if c.semantic_text and c.confidence > best_conf:
                best_semantic = c.semantic_text
                best_conf = c.confidence

        if total_w == 0:
            total_w = 1.0

        fused_conf = sum(
            weights.get(c.signal, 0.1) * c.confidence for c in candidates
        ) / max(sum(weights.get(c.signal, 0.1) for c in candidates), 1e-6)

        return LabelResult(
            bbox_x_min=wx1 / total_w,
            bbox_y_min=wy1 / total_w,
            bbox_x_max=wx2 / total_w,
            bbox_y_max=wy2 / total_w,
            semantic_text=best_semantic,
            confidence=min(1.0, fused_conf),
            candidates=candidates,
            transition_detected=transition,
        )
