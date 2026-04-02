"""Tests for ralph_self_improvement.core.loop (mocked)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
)
from ralph_self_improvement.core.config import RalphConfig
from ralph_self_improvement.core.loop import ImprovementLoop
from ralph_self_improvement.core.types import Judgment, WeightSnapshot


def _make_sample(sample_id: str) -> HarvestSample:
    event = ActionEvent(timestamp_ms=0, action=ActionType.click, x=200, y=300)
    label = LabelResult(
        bbox_x_min=100, bbox_y_min=200, bbox_x_max=300, bbox_y_max=400,
        confidence=0.8, semantic_text="button",
    )
    return HarvestSample(
        sample_id=sample_id,
        sample_dir=Path(f"/tmp/{sample_id}"),
        event=event,
        label=label,
    )


class TestImprovementLoop:
    @patch("ralph_self_improvement.core.loop.DPOTrainer")
    @patch("ralph_self_improvement.core.loop.WeightTuner")
    @patch("ralph_self_improvement.core.loop.EnsembleJudge")
    @patch("ralph_self_improvement.core.loop.HarvestSession")
    @patch("ralph_self_improvement.core.loop.HarvestConfig.from_yaml")
    def test_single_iteration(
        self,
        mock_hconfig,
        mock_session_cls,
        mock_judge_cls,
        mock_tuner_cls,
        mock_dpo_cls,
        tmp_path: Path,
    ):
        # Setup config
        ralph_yaml = tmp_path / "ralph.yaml"
        harvest_yaml = tmp_path / "harvest.yaml"
        harvest_yaml.write_text(yaml.dump({"workdir": str(tmp_path / "runs")}))

        cfg = RalphConfig(
            harvest_config_path=str(harvest_yaml),
            output_dir=str(tmp_path / "ralph"),
            judgments_path=str(tmp_path / "judgments.jsonl"),
            preferences_path=str(tmp_path / "preferences.jsonl"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
        )
        cfg.loop.max_iterations = 1
        cfg.loop.run_weight_tuning = False
        cfg.loop.run_dpo = False

        # Mock session
        mock_session = MagicMock()
        mock_session.labeled_samples.return_value = [_make_sample(f"s{i}") for i in range(5)]
        mock_session_cls.return_value = mock_session

        # Mock judge
        mock_judge = MagicMock()
        mock_judge.judge_batch.return_value = [
            Judgment(sample_id=f"s{i}", ensemble_score=0.6 + i * 0.05, iou_with_judge=0.5 + i * 0.05)
            for i in range(5)
        ]
        mock_judge_cls.return_value = mock_judge

        loop = ImprovementLoop(cfg)
        results = loop.run()

        assert len(results) == 1
        assert results[0].iteration == 1
        assert results[0].n_samples == 5
        assert results[0].mean_ensemble_score > 0

    @patch("ralph_self_improvement.core.loop.DPOTrainer")
    @patch("ralph_self_improvement.core.loop.WeightTuner")
    @patch("ralph_self_improvement.core.loop.EnsembleJudge")
    @patch("ralph_self_improvement.core.loop.HarvestSession")
    @patch("ralph_self_improvement.core.loop.HarvestConfig.from_yaml")
    def test_no_samples(
        self,
        mock_hconfig,
        mock_session_cls,
        mock_judge_cls,
        mock_tuner_cls,
        mock_dpo_cls,
        tmp_path: Path,
    ):
        harvest_yaml = tmp_path / "harvest.yaml"
        harvest_yaml.write_text(yaml.dump({"workdir": str(tmp_path / "runs")}))

        cfg = RalphConfig(
            harvest_config_path=str(harvest_yaml),
            metrics_path=str(tmp_path / "metrics.jsonl"),
        )
        cfg.loop.max_iterations = 1

        mock_session = MagicMock()
        mock_session.labeled_samples.return_value = []
        mock_session_cls.return_value = mock_session

        loop = ImprovementLoop(cfg)
        results = loop.run()

        assert len(results) == 1
        assert results[0].n_samples == 0
