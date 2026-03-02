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


class LabelerConfig(BaseModel):
    click_crop_radius_px: int = Field(default=80, ge=10)
    diff_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    contour_min_area_px: int = Field(default=100, ge=1)
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])
    ocr_gpu: bool = False
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "click_proximity": 0.3,
            "diff": 0.4,
            "ocr": 0.2,
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


class ExportConfig(BaseModel):
    normalizing_range: int = Field(default=1000, ge=1)


class HarvestConfig(BaseModel):
    workdir: str = "runs/harvest_session_01"
    game_profile: str | None = None

    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    labeler: LabelerConfig = Field(default_factory=LabelerConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarvestConfig:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
