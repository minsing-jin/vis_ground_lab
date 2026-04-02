"""Tests for ralph_self_improvement.core.config."""

import tempfile
from pathlib import Path

import yaml

from ralph_self_improvement.core.config import RalphConfig


class TestRalphConfig:
    def test_defaults(self):
        cfg = RalphConfig()
        assert cfg.judge.vlm_weight == 0.6
        assert cfg.judge.heuristic_weight == 0.4
        assert cfg.weight_tuner.n_samples == 50
        assert cfg.weight_tuner.n_trials == 30
        assert cfg.dpo.method == "sft"
        assert cfg.loop.max_iterations == 5
        assert cfg.loop.patience == 3

    def test_from_yaml(self, tmp_path: Path):
        data = {
            "harvest_config_path": "configs/harvest.yaml",
            "output_dir": "runs/test_ralph",
            "judge": {
                "vlm_model_name": "microsoft/Florence-2-large",
                "vlm_weight": 0.7,
                "heuristic_weight": 0.3,
            },
            "loop": {
                "max_iterations": 10,
                "patience": 5,
            },
        }
        yaml_path = tmp_path / "ralph.yaml"
        yaml_path.write_text(yaml.dump(data))

        cfg = RalphConfig.from_yaml(yaml_path)
        assert cfg.judge.vlm_weight == 0.7
        assert cfg.judge.heuristic_weight == 0.3
        assert cfg.loop.max_iterations == 10
        assert cfg.loop.patience == 5
        assert cfg.output_dir == "runs/test_ralph"

    def test_from_yaml_minimal(self, tmp_path: Path):
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text("{}")

        cfg = RalphConfig.from_yaml(yaml_path)
        assert cfg.judge.vlm_weight == 0.6
        assert cfg.dpo.base_model_name == "microsoft/Florence-2-base"
