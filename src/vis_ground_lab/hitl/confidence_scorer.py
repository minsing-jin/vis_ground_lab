"""Uncertainty scoring for active learning routing."""

from __future__ import annotations

from vis_ground_lab.base import UIElement


def _iou(a: UIElement, b: UIElement) -> float:
    """Compute intersection-over-union between two UIElements."""
    x1 = max(a.bbox.x_min, b.bbox.x_min)
    y1 = max(a.bbox.y_min, b.bbox.y_min)
    x2 = min(a.bbox.x_max, b.bbox.x_max)
    y2 = min(a.bbox.y_max, b.bbox.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a.bbox.x_max - a.bbox.x_min) * (a.bbox.y_max - a.bbox.y_min)
    area_b = (b.bbox.x_max - b.bbox.x_min) * (b.bbox.y_max - b.bbox.y_min)
    return inter / (area_a + area_b - inter)


class ConfidenceScorer:
    """Score prediction uncertainty for HITL routing decisions."""

    def __init__(
        self,
        low_confidence_threshold: float = 0.4,
        ambiguity_iou_threshold: float = 0.3,
    ) -> None:
        self.low_confidence_threshold = low_confidence_threshold
        self.ambiguity_iou_threshold = ambiguity_iou_threshold

    def score_elements(self, elements: list[UIElement]) -> list[tuple[UIElement, float]]:
        """Return (element, uncertainty_score) pairs. Higher = more uncertain."""
        scored: list[tuple[UIElement, float]] = []
        for i, elem in enumerate(elements):
            uncertainty = 1.0 - elem.score

            # Boost uncertainty for ambiguous overlapping boxes
            for j, other in enumerate(elements):
                if i == j:
                    continue
                if _iou(elem, other) > self.ambiguity_iou_threshold:
                    uncertainty = min(1.0, uncertainty + 0.2)
                    break

            scored.append((elem, round(uncertainty, 4)))
        return scored

    def needs_review(self, elements: list[UIElement]) -> list[UIElement]:
        """Return elements that should be routed to human review."""
        scored = self.score_elements(elements)
        return [elem for elem, score in scored if score > (1.0 - self.low_confidence_threshold)]
