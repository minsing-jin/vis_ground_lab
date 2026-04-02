"""Pydantic configuration schema for data_harvest."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RecorderConfig(BaseModel):
    capture_fps: int = Field(default=10, ge=1, le=60)
    buffer_seconds: int = Field(default=5, ge=1)
    post_action_delay_ms: int = Field(default=500, ge=100)
    monitor_index: int = Field(default=0, ge=0)
    ui_scale: float | None = Field(default=None, gt=0.0)
    enable_hover: bool = False


class VLMConfig(BaseModel):
    enabled: bool = True
    model_name: str = "microsoft/Florence-2-base"
    device_map: str = "auto"
    crop_radius_px: int = Field(default=120, ge=20)
    prompts: list[str] = Field(
        default_factory=lambda: [
            "detect the UI element",
            "detect the clickable button",
        ]
    )


class LabelerConfig(BaseModel):
    routing_only: bool = Field(
        default=True,
        description="Active data-harvest path only labels screen_type/situation_id/primitive_id.",
    )
    provider: str = Field(
        default="gemini",
        description="Primary auto-label teacher. Supported: gemini, local_vlm.",
    )
    provider_fallback_to_local: bool = Field(
        default=True,
        description="If the primary provider fails, fall back to the local VLM/page labeler.",
    )
    click_crop_radius_px: int = Field(default=80, ge=10)
    diff_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    contour_min_area_px: int = Field(default=100, ge=1)
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])
    ocr_gpu: bool = False
    use_ocr: bool = Field(default=False, description="Use EasyOCR (fallback). Disabled by default in favour of VLM.")
    legacy_weak_signals: bool = Field(
        default=False,
        description="Enable click/diff/OCR weak supervision. Disabled by default for VLM-first page labeling.",
    )
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "click_proximity": 0.25,
            "diff": 0.35,
            "vlm": 0.3,
            "ocr": 0.0,
            "profile_hint": 0.1,
        }
    )


class FilterConfig(BaseModel):
    min_diff_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    dedup_hash_threshold: int = Field(default=8, ge=0)
    blur_laplacian_threshold: float = Field(default=50.0, ge=0.0)
    dark_overlay_threshold: float = Field(default=30.0, ge=0.0)
    transition_max_diff_ratio: float = Field(default=0.4, ge=0.0, le=1.0)


class ReviewConfig(BaseModel):
    server_port: int = Field(default=7861, ge=1024, le=65535)
    auto_approve_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    enable_auto_approve: bool = False
    legacy_mode: bool = Field(
        default=False,
        description="Expose legacy grounding/bbox editing controls in the review UI.",
    )


class ExportConfig(BaseModel):
    normalizing_range: int = Field(default=1000, ge=1)
    routing_only: bool = Field(
        default=True,
        description="Default export surface only shows routing datasets.",
    )
    router_roi_fallbacks: list[str] = Field(
        default_factory=lambda: ["bottom_right", "popup_center", "unit_panel", "main_map"],
    )


class RelabelConfig(BaseModel):
    enabled: bool = False
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash-preview"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)
    timeout_sec: int = Field(default=20, ge=1)
    ontology_strict: bool = True
    max_samples: int | None = Field(default=None, ge=1)


class HarvestConfig(BaseModel):
    workdir: str = "runs/harvest_session_01"
    game_profile: str | None = None

    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    labeler: LabelerConfig = Field(default_factory=LabelerConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    relabel: RelabelConfig = Field(default_factory=RelabelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarvestConfig:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
