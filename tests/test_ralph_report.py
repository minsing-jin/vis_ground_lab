"""Tests for ralph_self_improvement.tracker.report."""

from pathlib import Path

from ralph_self_improvement.core.types import IterationResult, WeightSnapshot
from ralph_self_improvement.tracker.metrics import MetricsTracker
from ralph_self_improvement.tracker.report import generate_report


class TestGenerateReport:
    def test_empty(self, tmp_path: Path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        report = generate_report(tracker)
        assert "No iterations recorded" in report

    def test_with_iterations(self, tmp_path: Path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        ws = WeightSnapshot(weights={"vlm": 0.5, "diff": 0.3}, objective_value=0.85)
        tracker.append(IterationResult(
            iteration=1, mean_ensemble_score=0.6, mean_iou=0.5,
            mean_distance_px=15.0, n_samples=100, n_preference_pairs=50,
            weight_snapshot=ws,
        ))
        tracker.append(IterationResult(
            iteration=2, mean_ensemble_score=0.75, mean_iou=0.65,
            mean_distance_px=10.0, n_samples=100, n_preference_pairs=40,
            weight_snapshot=ws,
        ))

        report = generate_report(tracker)
        assert "RALPH Self-Improvement Report" in report
        assert "Total iterations: 2" in report
        assert "Best iteration:" in report
        assert "vlm" in report
        assert "diff" in report

    def test_trend_indicators(self, tmp_path: Path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.append(IterationResult(iteration=1, mean_ensemble_score=0.5))
        tracker.append(IterationResult(iteration=2, mean_ensemble_score=0.7))
        tracker.append(IterationResult(iteration=3, mean_ensemble_score=0.65))

        report = generate_report(tracker)
        assert "^" in report  # improvement from iter 1 to 2
        assert "v" in report  # regression from iter 2 to 3
