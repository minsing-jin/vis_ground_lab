"""Pydantic configuration schema definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TaskConfig(BaseModel):
    """Top-level task selector."""

    name: str = Field(default="grounding")


class TrainerConfig(BaseModel):
    """Core trainer hyperparameters."""

    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=3, ge=1)
    checkpoint_dir: str = Field(default="checkpoints")
    num_workers: int = Field(default=0, ge=0)


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
    pretrained: bool = False
    router_image_size: int = Field(default=224, ge=64)
    router_aux_loss_weight: float = Field(default=0.2, ge=0.0)
    router_dropout: float = Field(default=0.1, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    """Dataset configuration."""

    train_jsonl: str | None = None
    eval_jsonl: str | None = None
    image_root: str | None = None
    dataset_yaml: str | None = None
    val_coco: str | None = None
    normalize_mode: str = Field(default="0-1000")
    train_csv: str | None = None
    val_csv: str | None = None
    label_column: str = Field(default="primitive_id")
    aux_label_columns: list[str] = Field(default_factory=lambda: ["screen_type", "situation_id"])


class CaptureConfig(BaseModel):
    """Configuration for input log capture and action-frame correlation."""

    input_log_path: str | None = None
    input_log_format: str = "jsonl"
    video_path: str | None = None
    frame_dir: str | None = None
    fps: float = 2.0
    time_tolerance_ms: float = 200.0
    crop_radius_px: int = 64


class HITLConfig(BaseModel):
    """Configuration for human-in-the-loop review pipeline."""

    queue_dir: str = "runs/hitl_queue"
    low_confidence_threshold: float = 0.4
    ambiguity_iou_threshold: float = 0.3


class RuntimeConfig(BaseModel):
    """Configuration for runtime monitoring and drift detection."""

    confidence_window: int = 100
    drift_hash_threshold: int = 12
    low_confidence_threshold: float = 0.3
    failure_store_dir: str = "runs/failures"
    reference_frames_dir: str | None = None


class RetrainConfig(BaseModel):
    """Configuration for automated retrain triggers."""

    failure_threshold: int = 50
    correction_threshold: int = 20
    drift_threshold: float = 0.5
    cooldown_hours: float = 24.0


class TrainRunConfig(BaseModel):
    """Top-level training config used by CLI."""

    task: TaskConfig = Field(default_factory=TaskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    data: DataConfig

    @model_validator(mode="after")
    def validate_task_inputs(self) -> "TrainRunConfig":
        _validate_data_for_task(self.task.name, self.data)
        return self


class FactoryConfig(BaseModel):
    """Full lifecycle factory config encompassing all pipeline stages."""

    task: TaskConfig = Field(default_factory=TaskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    data: DataConfig
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    tool_id: str = "tool"
    tool_version: str = "v1"

    @model_validator(mode="after")
    def validate_task_inputs(self) -> "FactoryConfig":
        _validate_data_for_task(self.task.name, self.data)
        return self


def _validate_data_for_task(task_name: str, data: DataConfig) -> None:
    task = task_name.lower()
    if task == "grounding":
        if not data.train_jsonl:
            raise ValueError("task.name=grounding requires data.train_jsonl")
        return
    if task == "tool_button_detection":
        if not (data.dataset_yaml or data.train_jsonl):
            raise ValueError("task.name=tool_button_detection requires data.dataset_yaml or data.train_jsonl")
        return
    if task == "router_classification":
        if not data.train_csv or not data.val_csv:
            raise ValueError("task.name=router_classification requires data.train_csv and data.val_csv")
        return
