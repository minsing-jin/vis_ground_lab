"""Tests for ralph_self_improvement.judge.heuristic_judge."""

from dataclasses import dataclass
from pathlib import Path

from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
)
from ralph_self_improvement.core.config import JudgeConfig
from ralph_self_improvement.judge.heuristic_judge import HeuristicJudge


def _make_sample(
    bbox: tuple[float, float, float, float] = (100, 200, 300, 400),
    click: tuple[float, float] = (200, 300),
    confidence: float = 0.8,
) -> HarvestSample:
    event = ActionEvent(timestamp_ms=0, action=ActionType.click, x=click[0], y=click[1])
    label = LabelResult(
        bbox_x_min=bbox[0], bbox_y_min=bbox[1],
        bbox_x_max=bbox[2], bbox_y_max=bbox[3],
        confidence=confidence,
    )
    return HarvestSample(sample_id="test", sample_dir=Path("/tmp/test"), event=event, label=label)


class TestHeuristicJudge:
    def setup_method(self):
        self.judge = HeuristicJudge(JudgeConfig())

    def test_good_sample(self):
        """A well-formed sample should get a high score."""
        sample = _make_sample()
        result = self.judge.judge(sample, 1920, 1080)
        assert result["score"] > 0.7
        assert result["click_inside"] is True
        assert len(result["penalties"]) == 0

    def test_click_outside(self):
        """Click outside bbox should be penalized."""
        sample = _make_sample(click=(500, 500))
        result = self.judge.judge(sample, 1920, 1080)
        assert result["click_inside"] is False
        assert "click_outside_bbox" in result["penalties"]
        assert result["score"] < 1.0

    def test_bbox_too_large(self):
        """Huge bbox covering >50% of screen should be penalized."""
        sample = _make_sample(bbox=(0, 0, 1920, 1080), click=(960, 540))
        result = self.judge.judge(sample, 1920, 1080)
        assert "bbox_too_large" in result["penalties"]

    def test_bbox_too_small(self):
        """Tiny bbox should be penalized."""
        sample = _make_sample(bbox=(100, 100, 101, 101), click=(100, 100))
        result = self.judge.judge(sample, 1920, 1080)
        assert "bbox_too_small" in result["penalties"]

    def test_extreme_aspect_ratio(self):
        """Very elongated bbox should be penalized."""
        sample = _make_sample(bbox=(0, 0, 1000, 5), click=(500, 2))
        result = self.judge.judge(sample, 1920, 1080)
        assert "extreme_aspect_ratio" in result["penalties"]

    def test_low_confidence(self):
        """Low confidence should be penalized."""
        sample = _make_sample(confidence=0.1)
        result = self.judge.judge(sample, 1920, 1080)
        assert "low_confidence" in result["penalties"]

    def test_degenerate_bbox(self):
        """Zero-area bbox should score 0."""
        sample = _make_sample(bbox=(100, 100, 100, 100))
        result = self.judge.judge(sample, 1920, 1080)
        assert result["score"] == 0.0
        assert "degenerate_bbox" in result["penalties"]

    def test_missing_data(self):
        """Sample without label should score 0."""
        sample = HarvestSample(sample_id="test", sample_dir=Path("/tmp/test"))
        result = self.judge.judge(sample, 1920, 1080)
        assert result["score"] == 0.0
        assert "missing_data" in result["penalties"]
