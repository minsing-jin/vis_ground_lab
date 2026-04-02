"""Pydantic configuration schema for ralph_self_improvement."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    """Configuration for the ensemble judge."""

    vlm_model_name: str = "microsoft/Florence-2-large"
    vlm_device_map: str = "auto"
    vlm_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    heuristic_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    vlm_prompts: list[str] = Field(
        default_factory=lambda: [
            "detect the UI element",
            "detect the clickable button",
        ]
    )
    min_bbox_area_ratio: float = Field(default=0.001, ge=0.0)
    max_bbox_area_ratio: float = Field(default=0.5, le=1.0)
    max_aspect_ratio: float = Field(default=10.0, ge=1.0)


class WeightTunerConfig(BaseModel):
    """Configuration for Optuna-based fusion weight tuning."""

    n_samples: int = Field(default=50, ge=5)
    n_trials: int = Field(default=30, ge=5)
    timeout_seconds: int = Field(default=180, ge=30)


class DPOConfig(BaseModel):
    """Configuration for DPO / SFT fine-tuning."""

    method: str = Field(default="sft", pattern="^(dpo|sft)$")
    base_model_name: str = "microsoft/Florence-2-base"
    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=1, ge=1)
    checkpoint_dir: str = "checkpoints/ralph"
    use_lora: bool = True
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)


class LoopConfig(BaseModel):
    """Configuration for the improvement loop orchestrator."""

    max_iterations: int = Field(default=5, ge=1)
    patience: int = Field(default=3, ge=1)
    improvement_threshold: float = Field(default=0.01, ge=0.0)
    run_dpo: bool = True
    run_weight_tuning: bool = True


class RalphConfig(BaseModel):
    """Top-level configuration for the RLAIF self-improvement loop."""

    harvest_config_path: str = "configs/harvest.yaml"
    output_dir: str = "runs/ralph"
    judgments_path: str = "runs/ralph/judgments.jsonl"
    preferences_path: str = "runs/ralph/preferences.jsonl"
    metrics_path: str = "runs/ralph/metrics.jsonl"

    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    weight_tuner: WeightTunerConfig = Field(default_factory=WeightTunerConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RalphConfig:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
