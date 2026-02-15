from __future__ import annotations

import pytest
import torch

from vis_ground_lab.base import BoundingBox, VGSample
from vis_ground_lab.config.schema import TrainerConfig

pytest.importorskip("transformers")

from vis_ground_lab.training.trainer_engine import TrainerEngine


class DummyProcessor:
    def __call__(self, text, images, return_tensors, padding, truncation):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2], [3, 4]])}


class DummyWrapper:
    def __init__(self):
        self.processor = DummyProcessor()
        self.model = object()


def test_collate_fn_builds_batch_and_labels(tmp_path):
    engine = TrainerEngine(DummyWrapper(), TrainerConfig(checkpoint_dir=str(tmp_path)))

    samples = [
        VGSample(image=object(), text="a", bbox=BoundingBox(1, 2, 3, 4)),
        VGSample(image=object(), text="b", bbox=BoundingBox(5, 6, 7, 8)),
    ]

    batch = engine.collate_fn(samples)
    assert "input_ids" in batch
    assert "labels" in batch
    assert "bbox_targets" in batch
    assert batch["bbox_targets"].shape == (2, 4)


def test_checkpoint_dir_is_created(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    TrainerEngine(DummyWrapper(), TrainerConfig(checkpoint_dir=str(checkpoint_dir)))
    assert checkpoint_dir.exists()
