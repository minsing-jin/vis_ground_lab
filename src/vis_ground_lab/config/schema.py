"""Pydantic configuration schema definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Top-level task selector."""

    name: str = Field(default="grounding")


class TrainerConfig(BaseModel):
    """Core trainer hyperparameters."""

    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=3, ge=1)
    checkpoint_dir: str = Field(default="checkpoints")


class ModelConfig(BaseModel):
    """Model configuration for wrapper instantiation."""

    backend: str = Field(default="florence2")
    name: str = Field(default="microsoft/Florence-2-base")
    adapter_path_or_repo: str | None = None
    cache_dir: str = ".hf_cache"
    train_image_size: int = Field(default=384, ge=128)
    train_image_seq_length: int = Field(default=256, ge=64)
    use_lora: bool = Field(default=True)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    """Dataset configuration."""

    train_jsonl: str
    eval_jsonl: str | None = None
    image_root: str | None = None
    dataset_yaml: str | None = None
    val_coco: str | None = None
    normalize_mode: str = Field(default="0-1000")


class TrainRunConfig(BaseModel):
    """Top-level training config used by CLI."""

    task: TaskConfig = Field(default_factory=TaskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    data: DataConfig
