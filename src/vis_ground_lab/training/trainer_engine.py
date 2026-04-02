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
        """Batch images/prompts using model processor."""
        if self.model_wrapper.processor is None:
            raise RuntimeError("Model processor is unavailable. Call load_model() first.")

        images = [sample.image for sample in samples]
        texts = [sample.text for sample in samples]

        model_inputs = self.model_wrapper.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Align floating tensors with model parameter dtype (CPU often needs fp32).
        model = self.model_wrapper.model
        if model is not None:
            model_dtype = None
            try:
                model_dtype = next(model.parameters()).dtype
            except (AttributeError, StopIteration, TypeError):
                model_dtype = None

            if model_dtype is not None:
                for key, value in list(model_inputs.items()):
                    if hasattr(value, "dtype") and torch.is_floating_point(value):
                        model_inputs[key] = value.to(dtype=model_dtype)

            # Florence-2 text side has a fixed positional embedding budget.
            model_config = getattr(model, "config", None)
            if model_config is not None and "input_ids" in model_inputs:
                max_pos = int(
                    getattr(
                        getattr(model_config, "text_config", model_config),
                        "max_position_embeddings",
                        1024,
                    )
                )
                if model_inputs["input_ids"].shape[1] > max_pos:
                    model_inputs["input_ids"] = model_inputs["input_ids"][:, :max_pos]
                    if "attention_mask" in model_inputs:
                        model_inputs["attention_mask"] = model_inputs["attention_mask"][:, :max_pos]

        # Default CausalLM training target: next-token prediction.
        if "input_ids" in model_inputs:
            model_inputs["labels"] = model_inputs["input_ids"].clone()
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
