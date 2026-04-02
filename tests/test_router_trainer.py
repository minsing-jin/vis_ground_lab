from __future__ import annotations

import csv

from PIL import Image
import torch.nn as nn

from vis_ground_lab.config.schema import TrainerConfig
from vis_ground_lab.data_manager import RouterClassificationDataset
from vis_ground_lab.models.timm_router import TimmRouterWrapper
from vis_ground_lab.training.router_trainer import RouterTrainer


class _TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 8),
            nn.ReLU(),
        )

    def forward(self, pixel_values):
        return self.layers(pixel_values)


def _write_router_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "primitive_id", "screen_type", "situation_id", "session_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_router_trainer_saves_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(TimmRouterWrapper, "_create_backbone", lambda self: _TinyBackbone())

    for name, color in [("a.png", (255, 0, 0)), ("b.png", (0, 255, 0)), ("c.png", (255, 0, 0)), ("d.png", (0, 255, 0))]:
        Image.new("RGB", (16, 16), color=color).save(tmp_path / name)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_router_csv(
        train_csv,
        [
            {"sample_id": "s1", "image_path": "a.png", "primitive_id": "religion_primitive", "screen_type": "popup", "situation_id": "religion_choice_visible", "session_id": "session_01"},
            {"sample_id": "s2", "image_path": "b.png", "primitive_id": "unit_ops_primitive", "screen_type": "main_map", "situation_id": "noncombat_unit_ops_visible", "session_id": "session_01"},
        ],
    )
    _write_router_csv(
        val_csv,
        [
            {"sample_id": "s3", "image_path": "c.png", "primitive_id": "religion_primitive", "screen_type": "popup", "situation_id": "religion_choice_visible", "session_id": "session_01"},
            {"sample_id": "s4", "image_path": "d.png", "primitive_id": "unit_ops_primitive", "screen_type": "main_map", "situation_id": "noncombat_unit_ops_visible", "session_id": "session_01"},
        ],
    )

    train_dataset = RouterClassificationDataset(
        source=train_csv,
        image_root=tmp_path,
        image_size=16,
        aux_label_columns=["screen_type", "situation_id"],
    )
    val_dataset = RouterClassificationDataset(
        source=val_csv,
        image_root=tmp_path,
        image_size=16,
        aux_label_columns=["screen_type", "situation_id"],
        label_to_index=train_dataset.label_to_index,
        aux_label_to_index=train_dataset.aux_label_to_index,
    )

    wrapper = TimmRouterWrapper(model_name="tiny", image_size=16)
    trainer = RouterTrainer(
        wrapper,
        TrainerConfig(
            learning_rate=1.0e-3,
            batch_size=2,
            epochs=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
        aux_loss_weight=0.1,
    )

    result = trainer.train(train_dataset, val_dataset)

    assert (tmp_path / "checkpoints" / "best_router.pt").exists()
    assert result["best"]["epoch"] == 1
    assert "primitive_macro_f1" in result["best"]
    assert (tmp_path / "checkpoints" / "router_label_maps.json").exists()
