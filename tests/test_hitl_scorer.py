"""Tests for hitl.confidence_scorer module."""

from __future__ import annotations

from vis_ground_lab.base import BoundingBox, UIElement
from vis_ground_lab.hitl.confidence_scorer import ConfidenceScorer


def _elem(score: float, x1: float = 0, y1: float = 0, x2: float = 10, y2: float = 10) -> UIElement:
    return UIElement(class_name="button", bbox=BoundingBox(x1, y1, x2, y2), score=score)


def test_score_high_confidence():
    scorer = ConfidenceScorer(low_confidence_threshold=0.4)
    elements = [_elem(0.95)]
    scored = scorer.score_elements(elements)
    assert len(scored) == 1
    assert scored[0][1] < 0.1  # very low uncertainty


def test_score_low_confidence():
    scorer = ConfidenceScorer(low_confidence_threshold=0.4)
    elements = [_elem(0.2)]
    scored = scorer.score_elements(elements)
    assert scored[0][1] > 0.7  # high uncertainty


def test_needs_review_routes_low_confidence():
    scorer = ConfidenceScorer(low_confidence_threshold=0.4)
    elements = [_elem(0.3), _elem(0.9)]
    review = scorer.needs_review(elements)
    assert len(review) == 1
    assert review[0].score == 0.3


def test_ambiguity_boost():
    scorer = ConfidenceScorer(low_confidence_threshold=0.5, ambiguity_iou_threshold=0.3)
    # Two overlapping boxes
    e1 = _elem(0.6, 0, 0, 10, 10)
    e2 = _elem(0.6, 2, 2, 12, 12)
    scored = scorer.score_elements([e1, e2])
    # Uncertainty should be boosted due to overlap
    assert scored[0][1] > 0.4


def test_no_review_all_confident():
    scorer = ConfidenceScorer(low_confidence_threshold=0.4)
    elements = [_elem(0.8), _elem(0.9)]
    review = scorer.needs_review(elements)
    assert review == []
