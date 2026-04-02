"""Tests for ralph_self_improvement.optimizer.weight_tuner (mocked)."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from data_harvest.core.config import HarvestConfig
from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
)
from ralph_self_improvement.core.config import JudgeConfig, WeightTunerConfig


def _make_sample(sample_id: str) -> HarvestSample:
    event = ActionEvent(timestamp_ms=0, action=ActionType.click, x=200, y=300)
    label = LabelResult(
        bbox_x_min=100, bbox_y_min=200, bbox_x_max=300, bbox_y_max=400,
        confidence=0.8,
    )
    return HarvestSample(
        sample_id=sample_id,
        sample_dir=Path(f"/tmp/{sample_id}"),
        event=event,
        label=label,
    )


optuna_available = True
try:
    import optuna
except ImportError:
    optuna_available = False


@pytest.mark.skipif(not optuna_available, reason="optuna not installed")
class TestWeightTuner:
    @patch("ralph_self_improvement.optimizer.weight_tuner.WeightTuner._evaluate_weights")
    def test_tune_returns_snapshot(self, mock_eval: MagicMock):
        """WeightTuner.tune() should return a WeightSnapshot."""
        from ralph_self_improvement.optimizer.weight_tuner import WeightTuner

        mock_eval.return_value = 0.75

        config = WeightTunerConfig(n_samples=5, n_trials=5, timeout_seconds=30)
        tuner = WeightTuner(
            config=config,
            judge_config=JudgeConfig(),
            harvest_config=HarvestConfig(),
        )

        samples = [_make_sample(f"s{i}") for i in range(10)]
        snapshot = tuner.tune(samples)

        assert snapshot.objective_value > 0
        assert len(snapshot.weights) == 5
        assert snapshot.n_trials > 0
        assert snapshot.n_samples == 5  # subsampled to 5

        # Weights should be normalized to sum ~1
        total = sum(snapshot.weights.values())
        assert abs(total - 1.0) < 0.01


class TestWeightTunerConfig:
    def test_defaults(self):
        cfg = WeightTunerConfig()
        assert cfg.n_samples == 50
        assert cfg.n_trials == 30
        assert cfg.timeout_seconds == 180

    def test_custom(self):
        cfg = WeightTunerConfig(n_samples=100, n_trials=50, timeout_seconds=300)
        assert cfg.n_samples == 100
        assert cfg.n_trials == 50
