from __future__ import annotations

import csv

from PIL import Image

from vis_ground_lab.experiments.autoresearch_bridge import run_router_autoresearch_experiment
from vis_ground_lab.models.timm_router import TimmRouterWrapper
from vis_ground_lab.training.router_trainer import RouterTrainer


def _write_router_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "primitive_id", "screen_type", "situation_id", "session_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_router_autoresearch_bridge_updates_tool_profile(tmp_path, monkeypatch):
    for name, color in [("a.png", (255, 0, 0)), ("b.png", (0, 255, 0))]:
        Image.new("RGB", (16, 16), color=color).save(tmp_path / name)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_router_csv(
        train_csv,
        [{"sample_id": "s1", "image_path": "a.png", "primitive_id": "religion_primitive", "screen_type": "popup", "situation_id": "religion_choice_visible", "session_id": "session_01"}],
    )
    _write_router_csv(
        val_csv,
        [{"sample_id": "s2", "image_path": "b.png", "primitive_id": "religion_primitive", "screen_type": "popup", "situation_id": "religion_choice_visible", "session_id": "session_01"}],
    )

    config_path = tmp_path / "router.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    config_path.write_text(
        f"""
task:
  name: router_classification
model:
  backend: timm_router
  name: resnet18
trainer:
  checkpoint_dir: {checkpoint_dir}
data:
  train_csv: {train_csv}
  val_csv: {val_csv}
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        RouterTrainer,
        "train",
        lambda self, train_dataset, val_dataset: {
            "checkpoint_path": str(checkpoint_dir / "best_router.pt"),
            "label_maps_path": str(checkpoint_dir / "router_label_maps.json"),
        },
    )
    monkeypatch.setattr(RouterTrainer, "evaluate", lambda self, loader: {"primitive_macro_f1": 1.0, "primitive_accuracy": 1.0})
    monkeypatch.setattr(TimmRouterWrapper, "load_model", lambda self, **kwargs: None)

    profile_path = tmp_path / "tool_profile.json"
    result = run_router_autoresearch_experiment(str(config_path), profile_path=str(profile_path))

    assert result["metrics"]["primitive_macro_f1"] == 1.0
    assert profile_path.exists()
    profile_text = profile_path.read_text(encoding="utf-8")
    assert "router_classification" in profile_text
    assert "primitive_macro_f1" in profile_text
