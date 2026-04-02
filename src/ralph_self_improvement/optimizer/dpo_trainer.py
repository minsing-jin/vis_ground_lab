"""DPO / SFT fine-tuning for Florence-2 using preference pairs.

Falls back to SFT on chosen-only samples when trl is unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ralph_self_improvement.core.config import DPOConfig
from ralph_self_improvement.core.types import PreferencePair

logger = logging.getLogger(__name__)


class DPOTrainer:
    """Train Florence-2 with DPO or SFT fallback."""

    def __init__(self, config: DPOConfig) -> None:
        self.config = config

    def train(self, pairs: list[PreferencePair]) -> str:
        """Run DPO or SFT training and return checkpoint path."""
        if self.config.method == "dpo":
            return self._train_dpo(pairs)
        return self._train_sft(pairs)

    def _train_dpo(self, pairs: list[PreferencePair]) -> str:
        """Attempt DPO training via trl. Falls back to SFT if unavailable."""
        try:
            return self._run_trl_dpo(pairs)
        except ImportError:
            logger.warning("trl not available, falling back to SFT on chosen-only samples.")
            return self._train_sft(pairs)

    def _run_trl_dpo(self, pairs: list[PreferencePair]) -> str:
        """Run DPO training using trl library."""
        from datasets import Dataset
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer as TRLDPOTrainer
        from vis_ground_lab.models.florence2 import Florence2Wrapper

        checkpoint_dir = Path(self.config.checkpoint_dir) / "dpo"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build DPO dataset
        records = []
        for pair in pairs:
            records.append({
                "prompt": pair.chosen_prompt,
                "chosen": json.dumps(pair.chosen_bbox),
                "rejected": json.dumps(pair.rejected_bbox),
            })

        if not records:
            logger.warning("No DPO records to train on.")
            return str(checkpoint_dir)

        dataset = Dataset.from_list(records)

        # Load model
        wrapper = Florence2Wrapper(
            model_name=self.config.base_model_name,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
        )
        wrapper.load_model()

        # Reference model (frozen copy)
        ref_wrapper = Florence2Wrapper(
            model_name=self.config.base_model_name,
            use_lora=False,
        )
        ref_wrapper.load_model()

        training_args = TRLDPOConfig(
            output_dir=str(checkpoint_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            remove_unused_columns=False,
            report_to=[],
        )

        trainer = TRLDPOTrainer(
            model=wrapper.model,
            ref_model=ref_wrapper.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=wrapper.processor,
        )
        trainer.train()
        trainer.save_model(str(checkpoint_dir))

        logger.info("DPO training complete. Checkpoint: %s", checkpoint_dir)
        return str(checkpoint_dir)

    def _train_sft(self, pairs: list[PreferencePair]) -> str:
        """SFT on chosen samples using TrainerEngine."""
        from PIL import Image

        from vis_ground_lab.base import BoundingBox, VGSample
        from vis_ground_lab.config.schema import TrainerConfig
        from vis_ground_lab.models.florence2 import Florence2Wrapper
        from vis_ground_lab.training.trainer_engine import TrainerEngine

        checkpoint_dir = Path(self.config.checkpoint_dir) / "sft"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build SFT dataset from chosen samples
        vg_samples: list[VGSample] = []
        for pair in pairs:
            image_path = Path(pair.chosen_image_path)
            if not image_path.exists():
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            if len(pair.chosen_bbox) < 4:
                continue

            bbox = BoundingBox(
                x_min=pair.chosen_bbox[0],
                y_min=pair.chosen_bbox[1],
                x_max=pair.chosen_bbox[2],
                y_max=pair.chosen_bbox[3],
            )
            vg_samples.append(VGSample(
                image=image,
                text=pair.chosen_prompt,
                bbox=bbox,
                image_id=pair.chosen_sample_id,
            ))

        if not vg_samples:
            logger.warning("No SFT samples to train on.")
            return str(checkpoint_dir)

        logger.info("SFT training on %d chosen samples.", len(vg_samples))

        wrapper = Florence2Wrapper(
            model_name=self.config.base_model_name,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
        )
        wrapper.load_model()

        trainer_config = TrainerConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            checkpoint_dir=str(checkpoint_dir),
        )
        engine = TrainerEngine(wrapper, trainer_config)
        engine.train(vg_samples)

        logger.info("SFT training complete. Checkpoint: %s", checkpoint_dir)
        return str(checkpoint_dir)
