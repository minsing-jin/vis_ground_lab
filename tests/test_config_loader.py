from __future__ import annotations

import pytest

from vis_ground_lab.config.loader import load_train_config


def test_load_train_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: microsoft/Florence-2-base
  use_lora: true
trainer:
  learning_rate: 0.0001
  batch_size: 2
  epochs: 1
data:
  train_jsonl: train.jsonl
  normalize_mode: 0-1000
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_train_config(config_path)
    assert cfg.model.use_lora is True
    assert cfg.trainer.batch_size == 2
    assert cfg.data.normalize_mode == "0-1000"


def test_load_router_config(tmp_path):
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        """
task:
  name: router_classification
model:
  backend: timm_router
  name: resnet18
trainer:
  learning_rate: 0.0003
  batch_size: 4
  epochs: 2
data:
  train_csv: train.csv
  val_csv: val.csv
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_train_config(config_path)
    assert cfg.task.name == "router_classification"
    assert cfg.model.backend == "timm_router"
    assert cfg.data.train_csv == "train.csv"
    assert cfg.data.aux_label_columns == ["screen_type", "situation_id"]


def test_router_config_requires_csv_paths(tmp_path):
    config_path = tmp_path / "broken_router.yaml"
    config_path.write_text(
        """
task:
  name: router_classification
data:
  train_csv: train.csv
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_train_config(config_path)
