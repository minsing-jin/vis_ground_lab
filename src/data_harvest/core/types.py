"""Core data types for the data_harvest engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class ActionType(str, Enum):
    """Types of user actions captured during gameplay."""

    click = "click"
    press = "press"
    drag = "drag"
    type = "type"
    scroll = "scroll"


class ReviewStatus(str, Enum):
    """Review status of a harvest sample."""

    pending = "pending"
    approved = "approved"
    edited = "edited"
    rejected = "rejected"


@dataclass(frozen=True)
class ActionEvent:
    """A single user action event captured by the input listener."""

    timestamp_ms: float
    action: ActionType
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    button: str | None = None
    key: str | None = None
    text: str | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionEvent:
        d = dict(d)
        d["action"] = ActionType(d["action"])
        return cls(**d)


@dataclass(frozen=True)
class BBoxCandidate:
    """A bounding box candidate from a single labeling signal."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    signal: str
    confidence: float
    semantic_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelResult:
    """Fused labeling result from multiple signals."""

    bbox_x_min: float
    bbox_y_min: float
    bbox_x_max: float
    bbox_y_max: float
    semantic_text: str | None = None
    semantic_id: str | None = None
    confidence: float = 0.0
    candidates: list[BBoxCandidate] = field(default_factory=list)
    screen_type: str | None = None
    transition_detected: bool = False

    @property
    def bbox_xyxy(self) -> list[float]:
        return [self.bbox_x_min, self.bbox_y_min, self.bbox_x_max, self.bbox_y_max]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() if isinstance(c, BBoxCandidate) else c for c in self.candidates]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelResult:
        d = dict(d)
        d["candidates"] = [
            BBoxCandidate(**c) if isinstance(c, dict) else c
            for c in d.get("candidates", [])
        ]
        return cls(**d)


@dataclass
class HarvestSample:
    """A single harvest sample: pre/post frames + action event + label."""

    sample_id: str
    sample_dir: Path
    event: ActionEvent | None = None
    label: LabelResult | None = None
    review_status: ReviewStatus = ReviewStatus.pending
    review_corrections: dict[str, Any] | None = None

    @property
    def pre_frame_path(self) -> Path:
        return self.sample_dir / "pre.png"

    @property
    def post_frame_path(self) -> Path:
        return self.sample_dir / "post.png"

    @property
    def event_path(self) -> Path:
        return self.sample_dir / "event.json"

    @property
    def label_path(self) -> Path:
        return self.sample_dir / "label.json"

    @property
    def review_path(self) -> Path:
        return self.sample_dir / "review.json"

    def save_event(self) -> None:
        if self.event is None:
            return
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.event_path.write_text(self.event.to_json(), encoding="utf-8")

    def save_label(self) -> None:
        if self.label is None:
            return
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.label_path.write_text(self.label.to_json(), encoding="utf-8")

    def save_review(self) -> None:
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "review_status": self.review_status.value,
            "review_corrections": self.review_corrections,
        }
        self.review_path.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, sample_dir: str | Path) -> HarvestSample:
        sample_dir = Path(sample_dir)
        sample_id = sample_dir.name

        event = None
        event_path = sample_dir / "event.json"
        if event_path.exists():
            event = ActionEvent.from_dict(json.loads(event_path.read_text("utf-8")))

        label = None
        label_path = sample_dir / "label.json"
        if label_path.exists():
            label = LabelResult.from_dict(json.loads(label_path.read_text("utf-8")))

        review_status = ReviewStatus.pending
        review_corrections = None
        review_path = sample_dir / "review.json"
        if review_path.exists():
            rd = json.loads(review_path.read_text("utf-8"))
            review_status = ReviewStatus(rd["review_status"])
            review_corrections = rd.get("review_corrections")

        return cls(
            sample_id=sample_id,
            sample_dir=sample_dir,
            event=event,
            label=label,
            review_status=review_status,
            review_corrections=review_corrections,
        )
