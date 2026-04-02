"""Tests for ralph_self_improvement.tracker.metrics."""

from pathlib import Path

from ralph_self_improvement.core.types import IterationResult, WeightSnapshot
from ralph_self_improvement.tracker.metrics import MetricsTracker


class TestMetricsTracker:
    def test_append_and_load(self, tmp_path: Path):
        path = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(path)
        assert len(tracker.results) == 0
        assert tracker.last_iteration == 0

        r1 = IterationResult(iteration=1, mean_ensemble_score=0.5, mean_iou=0.4, n_samples=10)
        tracker.append(r1)

        r2 = IterationResult(iteration=2, mean_ensemble_score=0.7, mean_iou=0.6, n_samples=10)
        tracker.append(r2)

        assert len(tracker.results) == 2
        assert tracker.last_iteration == 2

        # Reload from disk
        tracker2 = MetricsTracker(path)
        assert len(tracker2.results) == 2
        assert tracker2.results[0].mean_ensemble_score == 0.5
        assert tracker2.results[1].mean_ensemble_score == 0.7

    def test_best_iteration(self, tmp_path: Path):
        path = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(path)

        tracker.append(IterationResult(iteration=1, mean_ensemble_score=0.5))
        tracker.append(IterationResult(iteration=2, mean_ensemble_score=0.8))
        tracker.append(IterationResult(iteration=3, mean_ensemble_score=0.6))

        best = tracker.best_iteration()
        assert best is not None
        assert best.iteration == 2
        assert best.mean_ensemble_score == 0.8

    def test_best_iteration_empty(self, tmp_path: Path):
        path = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(path)
        assert tracker.best_iteration() is None

    def test_improvement_stalled(self, tmp_path: Path):
        path = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(path)

        # Add improving iterations
        tracker.append(IterationResult(iteration=1, mean_ensemble_score=0.5))
        tracker.append(IterationResult(iteration=2, mean_ensemble_score=0.8))
        assert not tracker.improvement_stalled(patience=3, threshold=0.01)

        # Add stalled iterations
        tracker.append(IterationResult(iteration=3, mean_ensemble_score=0.79))
        tracker.append(IterationResult(iteration=4, mean_ensemble_score=0.78))
        tracker.append(IterationResult(iteration=5, mean_ensemble_score=0.79))

        assert tracker.improvement_stalled(patience=3, threshold=0.01)

    def test_resume_from_existing(self, tmp_path: Path):
        path = tmp_path / "metrics.jsonl"
        tracker = MetricsTracker(path)
        tracker.append(IterationResult(iteration=1, mean_ensemble_score=0.5))
        tracker.append(IterationResult(iteration=2, mean_ensemble_score=0.6))

        # Simulate resume
        tracker2 = MetricsTracker(path)
        assert tracker2.last_iteration == 2
        tracker2.append(IterationResult(iteration=3, mean_ensemble_score=0.7))
        assert tracker2.last_iteration == 3
