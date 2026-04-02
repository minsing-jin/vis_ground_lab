"""Tests for data_harvest.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data_harvest.core.config import HarvestConfig, RecorderConfig, LabelerConfig


class TestHarvestConfig:
    def test_defaults(self):
        cfg = HarvestConfig()
        assert cfg.workdir == "runs/harvest_session_01"
        assert cfg.recorder.capture_fps == 10
        assert cfg.labeler.provider == "gemini"
        assert cfg.labeler.provider_fallback_to_local is True
        assert cfg.labeler.click_crop_radius_px == 80
        assert cfg.filter.min_diff_ratio == 0.005
        assert cfg.review.auto_approve_confidence == 0.9
        assert cfg.export.normalizing_range == 1000

    def test_from_yaml(self, tmp_path: Path):
        data = {
            "workdir": "runs/test_session",
            "game_profile": "civ6",
            "recorder": {"capture_fps": 5, "buffer_seconds": 3},
            "labeler": {"provider": "local_vlm", "click_crop_radius_px": 100},
        }
        yaml_path = tmp_path / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        cfg = HarvestConfig.from_yaml(yaml_path)
        assert cfg.workdir == "runs/test_session"
        assert cfg.game_profile == "civ6"
        assert cfg.recorder.capture_fps == 5
        assert cfg.recorder.buffer_seconds == 3
        assert cfg.labeler.provider == "local_vlm"
        assert cfg.labeler.click_crop_radius_px == 100
        # Defaults should still apply for unspecified fields
        assert cfg.filter.dedup_hash_threshold == 8

    def test_validation(self):
        with pytest.raises(Exception):
            RecorderConfig(capture_fps=0)

    def test_fusion_weights_default(self):
        cfg = LabelerConfig()
        assert "click_proximity" in cfg.fusion_weights
        assert "diff" in cfg.fusion_weights
