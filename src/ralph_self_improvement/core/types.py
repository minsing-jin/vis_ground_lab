"""Core data types for the RLAIF self-improvement loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Judgment:
    """Quality assessment of a single auto-labeled sample.

    Produced by VLM judge, heuristic judge, or their ensemble.
    """

    sample_id: str
    vlm_score: float = 0.0
    heuristic_score: float = 0.0
    ensemble_score: float = 0.0
    iou_with_judge: float = 0.0
    click_inside_bbox: bool = False
    bbox_area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Judgment:
        return cls(**d)


@dataclass
class PreferencePair:
    """A chosen/rejected pair for DPO training.

    The chosen sample has a higher ensemble score than the rejected one.
    """

    chosen_sample_id: str
    rejected_sample_id: str
    chosen_score: float
    rejected_score: float
    chosen_prompt: str = ""
    chosen_bbox: list[float] = field(default_factory=list)
    rejected_prompt: str = ""
    rejected_bbox: list[float] = field(default_factory=list)
    chosen_image_path: str = ""
    rejected_image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreferencePair:
        return cls(**d)


@dataclass
class WeightSnapshot:
    """Optimized fusion weights from a single tuning run."""

    weights: dict[str, float]
    objective_value: float
    n_trials: int = 0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WeightSnapshot:
        return cls(**d)


@dataclass
class IterationResult:
    """Summary of a single self-improvement iteration."""

    iteration: int
    mean_iou: float = 0.0
    mean_distance_px: float = 0.0
    mean_ensemble_score: float = 0.0
    n_samples: int = 0
    n_preference_pairs: int = 0
    weight_snapshot: WeightSnapshot | None = None
    checkpoint_path: str | None = None
    improved: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.weight_snapshot is not None:
            d["weight_snapshot"] = self.weight_snapshot.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IterationResult:
        d = dict(d)
        ws = d.pop("weight_snapshot", None)
        if ws is not None:
            ws = WeightSnapshot.from_dict(ws)
        return cls(**d, weight_snapshot=ws)
