"""Tests for profile.tool_profile module."""

from __future__ import annotations

import json

from vis_ground_lab.config.schema import DataConfig, ModelConfig
from vis_ground_lab.profile.tool_profile import ToolProfile


def test_save_and_load(tmp_path):
    profile = ToolProfile(
        tool_id="my_tool",
        tool_version="v2",
        package_dir=str(tmp_path / "pkg"),
        model_cfg=ModelConfig(backend="yolo_ultralytics", name="yolov8n.pt"),
        data_config=DataConfig(train_jsonl="data/train.jsonl"),
    )

    path = tmp_path / "profile.json"
    profile.save(path)
    assert path.exists()

    loaded = ToolProfile.load(path)
    assert loaded.tool_id == "my_tool"
    assert loaded.tool_version == "v2"
    assert loaded.model_cfg.backend == "yolo_ultralytics"
    assert loaded.data_config.train_jsonl == "data/train.jsonl"
    assert loaded.updated_at != ""


def test_from_package(tmp_path):
    pkg_dir = tmp_path / "package"
    pkg_dir.mkdir()
    meta = {
        "tool_metadata": {
            "tool_id": "t1",
            "tool_version": "v1",
            "dataset_hash": "abc123",
        }
    }
    (pkg_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    profile = ToolProfile.from_package(pkg_dir, tool_id="t1", tool_version="v1")
    assert profile.tool_id == "t1"
    assert profile.dataset_hash == "abc123"
    assert profile.created_at != ""


def test_record_training_run(tmp_path):
    profile = ToolProfile(
        tool_id="t",
        tool_version="v1",
        package_dir=str(tmp_path),
    )
    profile.record_training_run({"mAP50": 0.85, "epochs": 30})
    assert len(profile.training_history) == 1
    assert profile.training_history[0]["mAP50"] == 0.85
    assert "timestamp" in profile.training_history[0]


def test_roundtrip_with_runtime_config(tmp_path):
    profile = ToolProfile(
        tool_id="t",
        tool_version="v1",
        package_dir=str(tmp_path),
        runtime_config={"drift_hash_threshold": 10, "low_confidence_threshold": 0.4},
    )
    path = tmp_path / "p.json"
    profile.save(path)
    loaded = ToolProfile.load(path)
    assert loaded.runtime_config["drift_hash_threshold"] == 10


def test_from_package_no_metadata(tmp_path):
    pkg_dir = tmp_path / "empty_package"
    pkg_dir.mkdir()
    profile = ToolProfile.from_package(pkg_dir, tool_id="t", tool_version="v1")
    assert profile.dataset_hash == ""
