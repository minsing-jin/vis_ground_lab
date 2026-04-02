"""Rule-based heuristic judge for label quality assessment.

Checks: bbox area ratio, aspect ratio, click containment, confidence.
"""

from __future__ import annotations

import logging
from typing import Any

from data_harvest.core.types import HarvestSample
from ralph_self_improvement.core.config import JudgeConfig

logger = logging.getLogger(__name__)


class HeuristicJudge:
    """Score auto-labels based on geometric and statistical heuristics."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def judge(self, sample: HarvestSample, image_width: int, image_height: int) -> dict[str, Any]:
        """Evaluate a single labeled sample.

        Returns a dict with:
            score (float): overall heuristic score [0, 1]
            click_inside (bool): whether click is inside the bbox
            area_ratio (float): bbox area / image area
            aspect_ratio (float): bbox width / height (or height / width, whichever >= 1)
            confidence (float): auto-labeler confidence
            penalties (list[str]): list of failed checks
        """
        label = sample.label
        event = sample.event
        if label is None or event is None:
            return {"score": 0.0, "click_inside": False, "area_ratio": 0.0,
                    "aspect_ratio": 0.0, "confidence": 0.0, "penalties": ["missing_data"]}

        penalties: list[str] = []
        score = 1.0

        # Bbox dimensions
        bw = label.bbox_x_max - label.bbox_x_min
        bh = label.bbox_y_max - label.bbox_y_min
        image_area = max(image_width * image_height, 1)
        bbox_area = max(bw * bh, 0.0)
        area_ratio = bbox_area / image_area

        # Aspect ratio (always >= 1)
        if bh > 0 and bw > 0:
            aspect_ratio = max(bw / bh, bh / bw)
        else:
            aspect_ratio = float("inf")

        # Check 1: Area ratio bounds
        if area_ratio < self.config.min_bbox_area_ratio:
            penalties.append("bbox_too_small")
            score -= 0.3
        if area_ratio > self.config.max_bbox_area_ratio:
            penalties.append("bbox_too_large")
            score -= 0.3

        # Check 2: Aspect ratio
        if aspect_ratio > self.config.max_aspect_ratio:
            penalties.append("extreme_aspect_ratio")
            score -= 0.2

        # Check 3: Click containment
        click_inside = False
        if event.x is not None and event.y is not None:
            click_inside = (
                label.bbox_x_min <= event.x <= label.bbox_x_max
                and label.bbox_y_min <= event.y <= label.bbox_y_max
            )
        if not click_inside:
            penalties.append("click_outside_bbox")
            score -= 0.3

        # Check 4: Confidence
        confidence = label.confidence
        if confidence < 0.3:
            penalties.append("low_confidence")
            score -= 0.2

        # Check 5: Degenerate bbox
        if bw <= 0 or bh <= 0:
            penalties.append("degenerate_bbox")
            score = 0.0

        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "click_inside": click_inside,
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "confidence": confidence,
            "penalties": penalties,
        }
