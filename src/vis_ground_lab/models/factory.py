"""Model factory for backend-based wrapper selection."""

from __future__ import annotations

from vis_ground_lab.config.schema import ModelConfig
from vis_ground_lab.models.florence2 import Florence2Wrapper
from vis_ground_lab.models.timm_router import TimmRouterWrapper
from vis_ground_lab.models.yolo_ultralytics import YoloUltralyticsWrapper


def create_model_wrapper(model_cfg: ModelConfig) -> Florence2Wrapper | YoloUltralyticsWrapper | TimmRouterWrapper:
    """Create model wrapper by backend name.

    Currently supported:
    - florence2
    """
    backend = model_cfg.backend.lower()

    if backend == "florence2":
        return Florence2Wrapper(
            model_name=model_cfg.name,
            use_lora=model_cfg.use_lora,
            lora_r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            cache_dir=model_cfg.cache_dir,
            train_image_size=model_cfg.train_image_size,
            train_image_seq_length=model_cfg.train_image_seq_length,
        )
    if backend == "timm_router":
        return TimmRouterWrapper(
            model_name=model_cfg.name,
            pretrained=model_cfg.pretrained,
            image_size=model_cfg.router_image_size,
            dropout=model_cfg.router_dropout,
        )
    if backend == "yolo_ultralytics":
        return YoloUltralyticsWrapper(model_name=model_cfg.name)

    raise ValueError(
        f"Unsupported model backend: {model_cfg.backend}. "
        "Add a new wrapper and register it in vis_ground_lab.models.factory.create_model_wrapper()."
    )
