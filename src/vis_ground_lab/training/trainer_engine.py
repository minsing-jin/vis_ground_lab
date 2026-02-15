"""Trainer engine wrapping Hugging Face Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from vis_ground_lab.base import VGSample
from vis_ground_lab.config.schema import TrainerConfig
from vis_ground_lab.models.florence2 import Florence2Wrapper


class TrainerEngine:
    """Thin wrapper around HF Trainer with custom collate behavior."""

    def __init__(self, model_wrapper: Florence2Wrapper, config: TrainerConfig) -> None:
        self.model_wrapper = model_wrapper
        self.config = config

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=float(self.config.epochs),
            save_strategy="epoch",
            logging_steps=10,
            remove_unused_columns=False,
            report_to=[],
        )

    def collate_fn(self, samples: list[VGSample]) -> dict[str, Any]:
        """Batch images/prompts using model processor and keep bbox targets."""
        if self.model_wrapper.processor is None:
            raise RuntimeError("Model processor is unavailable. Call load_model() first.")

        images = [sample.image for sample in samples]
        texts = [sample.text for sample in samples]
        bboxes = torch.tensor(
            [[sample.bbox.x_min, sample.bbox.y_min, sample.bbox.x_max, sample.bbox.y_max] for sample in samples],
            dtype=torch.float32,
        )

        model_inputs = self.model_wrapper.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Default CausalLM training target: next-token prediction.
        if "input_ids" in model_inputs:
            model_inputs["labels"] = model_inputs["input_ids"].clone()

        model_inputs["bbox_targets"] = bboxes
        return model_inputs

    def train(self, train_dataset: Any, eval_dataset: Any | None = None) -> Trainer:
        """Launch HF training and save resulting checkpoints locally."""
        if self.model_wrapper.model is None:
            raise RuntimeError("Model is unavailable. Call load_model() first.")

        trainer = Trainer(
            model=self.model_wrapper.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.collate_fn,
        )
        trainer.train()
        trainer.save_model(self.config.checkpoint_dir)
        return trainer
