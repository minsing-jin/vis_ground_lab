from __future__ import annotations

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
