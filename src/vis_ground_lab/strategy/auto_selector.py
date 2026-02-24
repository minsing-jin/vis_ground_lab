"""Auto-select training strategy based on data profile."""

from __future__ import annotations

from dataclasses import dataclass

from vis_ground_lab.config.schema import DataConfig, ModelConfig, TaskConfig, TrainerConfig, TrainRunConfig
from vis_ground_lab.strategy.data_profiler import DataProfile


@dataclass(frozen=True)
class TrainingStrategy:
    """Recommended training configuration based on data profile."""

    backend: str
    model_name: str
    task_name: str
    suggested_epochs: int
    suggested_batch_size: int
    suggested_lr: float
    use_hpo: bool
    suggested_n_trials: int
    rationale: str


class AutoStrategySelector:
    """Select training strategy based on data profile characteristics."""

    def select(self, profile: DataProfile, target_latency_ms: float | None = None) -> TrainingStrategy:
        """Choose model, hyperparams, and HPO settings from the data profile."""
        # Grounding task — Florence2+LoRA
        if profile.avg_annotations_per_image <= 1.0 and profile.num_classes <= 2:
            if profile.estimated_complexity in ("trivial", "simple"):
                return TrainingStrategy(
                    backend="florence2",
                    model_name="microsoft/Florence-2-base",
                    task_name="grounding",
                    suggested_epochs=5,
                    suggested_batch_size=4,
                    suggested_lr=5e-5,
                    use_hpo=False,
                    suggested_n_trials=0,
                    rationale="Small grounding dataset, Florence2+LoRA without HPO",
                )

        # Detection task — YOLO variants by scale
        if profile.num_images < 200 and profile.num_classes <= 1:
            return TrainingStrategy(
                backend="yolo_ultralytics",
                model_name="yolov8n.pt",
                task_name="tool_button_detection",
                suggested_epochs=30,
                suggested_batch_size=16,
                suggested_lr=1e-3,
                use_hpo=False,
                suggested_n_trials=0,
                rationale="<200 images, single class → yolov8n, no HPO",
            )

        if profile.num_images < 1000 and profile.num_classes <= 5:
            return TrainingStrategy(
                backend="yolo_ultralytics",
                model_name="yolov8s.pt",
                task_name="tool_button_detection",
                suggested_epochs=50,
                suggested_batch_size=16,
                suggested_lr=1e-3,
                use_hpo=True,
                suggested_n_trials=10,
                rationale="200-1000 images → yolov8s with 10 HPO trials",
            )

        return TrainingStrategy(
            backend="yolo_ultralytics",
            model_name="yolov8m.pt",
            task_name="tool_button_detection",
            suggested_epochs=80,
            suggested_batch_size=8,
            suggested_lr=1e-3,
            use_hpo=True,
            suggested_n_trials=20,
            rationale=">1000 images or >5 classes → yolov8m with 20 HPO trials",
        )

    def to_train_run_config(self, strategy: TrainingStrategy, data_config: DataConfig) -> TrainRunConfig:
        """Convert a TrainingStrategy to a TrainRunConfig."""
        return TrainRunConfig(
            task=TaskConfig(name=strategy.task_name),
            model=ModelConfig(
                backend=strategy.backend,
                name=strategy.model_name,
                use_lora=(strategy.backend == "florence2"),
            ),
            trainer=TrainerConfig(
                learning_rate=strategy.suggested_lr,
                batch_size=strategy.suggested_batch_size,
                epochs=strategy.suggested_epochs,
            ),
            data=data_config,
        )
