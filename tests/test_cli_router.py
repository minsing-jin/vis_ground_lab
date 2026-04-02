from __future__ import annotations

import json

from typer.testing import CliRunner

from vis_ground_lab.cli import app


runner = CliRunner()


def test_evaluate_dispatches_router_config(monkeypatch, tmp_path):
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        """
task:
  name: router_classification
model:
  backend: timm_router
  name: resnet18
data:
  train_csv: train.csv
  val_csv: val.csv
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "vis_ground_lab.cli._evaluate_router_task",
        lambda cfg, checkpoint_path=None: {
            "task": "router_classification",
            "primary_metric": "primitive_macro_f1",
            "metrics": {"primitive_macro_f1": 0.9},
            "artifacts": {},
            "checkpoint_path": checkpoint_path,
        },
    )

    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "router_classification"
    assert payload["metrics"]["primitive_macro_f1"] == 0.9
