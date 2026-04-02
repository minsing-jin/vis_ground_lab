"""Core data types for the data_harvest engine."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class ActionType(str, Enum):
    """Types of user actions captured during gameplay."""

    click = "click"
    right_click = "right_click"
    double_click = "double_click"
    hold = "hold"
    hover = "hover"
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
    duration_ms: float | None = None
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
    semantic_id: str | None = None
    function_id: str | None = None
    primitive_id: str | None = None
    roi_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageLabel:
    """Page-level routing context for the current screenshot."""

    screen_type: str | None = None
    situation_id: str | None = None
    state_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteLabel:
    """Routing target for the current page."""

    primitive_id: str | None = None
    target_element_id: str | None = None
    roi_name: str | None = None
    roi_bbox_norm: list[float] | None = None
    trigger_modality: str | None = None
    trigger_action_type: str | None = None
    trigger_mouse_button: str | None = None
    trigger_key: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionableElementLabel:
    """A single actionable UI element on the page."""

    element_id: str
    bbox_x_min: float = 0.0
    bbox_y_min: float = 0.0
    bbox_x_max: float = 1.0
    bbox_y_max: float = 1.0
    semantic_id: str | None = None
    semantic_text: str | None = None
    function_id: str | None = None
    hotkeys: list[str] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    enabled_state: str = "enabled"
    roi_name: str | None = None
    is_route_target: bool = False
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox_xyxy(self) -> list[float]:
        return [self.bbox_x_min, self.bbox_y_min, self.bbox_x_max, self.bbox_y_max]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionableElementLabel":
        data = dict(d)
        if "bbox_xyxy" in data:
            x1, y1, x2, y2 = data.pop("bbox_xyxy")
            data.setdefault("bbox_x_min", x1)
            data.setdefault("bbox_y_min", y1)
            data.setdefault("bbox_x_max", x2)
            data.setdefault("bbox_y_max", y2)
        data.setdefault("metadata", {})
        data.setdefault("hotkeys", [])
        data.setdefault("available_actions", [])
        return cls(**data)


@dataclass
class LabelResult:
    """Page-level harvest label containing routing and actionable elements."""

    bbox_x_min: float = 0.0
    bbox_y_min: float = 0.0
    bbox_x_max: float = 1.0
    bbox_y_max: float = 1.0
    semantic_text: str | None = None
    semantic_id: str | None = None
    function_id: str | None = None
    hotkeys: list[str] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    situation_id: str | None = None
    confidence: float = 0.0
    candidates: list[BBoxCandidate] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    screen_type: str | None = None
    transition_detected: bool = False
    page: PageLabel | None = None
    route_label: RouteLabel | None = None
    elements: list[ActionableElementLabel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.page is not None and not isinstance(self.page, PageLabel):
            self.page = PageLabel(**self.page)
        if self.route_label is not None and not isinstance(self.route_label, RouteLabel):
            self.route_label = RouteLabel(**self.route_label)
        self.elements = [
            e if isinstance(e, ActionableElementLabel) else ActionableElementLabel.from_dict(e)
            for e in self.elements
        ]
        self._bootstrap_nested()
        self.sync_legacy_fields()

    @property
    def bbox_xyxy(self) -> list[float]:
        primary = self.primary_element
        if primary is not None:
            return primary.bbox_xyxy
        return [self.bbox_x_min, self.bbox_y_min, self.bbox_x_max, self.bbox_y_max]

    @property
    def primary_element(self) -> ActionableElementLabel | None:
        if not self.elements:
            return None
        for element in self.elements:
            if element.is_route_target:
                return element
        return self.elements[0]

    def sync_legacy_fields(self) -> None:
        """Keep legacy single-target fields aligned with the primary element."""
        if self.page is None:
            self.page = PageLabel(
                screen_type=self.screen_type,
                situation_id=self.situation_id,
                confidence=self.confidence,
            )
        else:
            if self.screen_type is None:
                self.screen_type = self.page.screen_type
            if self.situation_id is None:
                self.situation_id = self.page.situation_id

        primary = self.primary_element
        if primary is not None:
            self.bbox_x_min = primary.bbox_x_min
            self.bbox_y_min = primary.bbox_y_min
            self.bbox_x_max = primary.bbox_x_max
            self.bbox_y_max = primary.bbox_y_max
            self.semantic_id = primary.semantic_id
            self.semantic_text = primary.semantic_text
            self.function_id = primary.function_id
            self.hotkeys = list(primary.hotkeys)
            self.available_actions = list(primary.available_actions)

        if self.route_label is None:
            self.route_label = RouteLabel(
                primitive_id=self.function_id,
                target_element_id=primary.element_id if primary is not None else None,
                confidence=self.confidence,
            )
        else:
            if self.route_label.primitive_id is None and self.function_id is not None:
                self.route_label.primitive_id = self.function_id
            if self.route_label.target_element_id is None and primary is not None:
                self.route_label.target_element_id = primary.element_id

    def to_page_sample(self, sample: "HarvestSample") -> dict[str, Any]:
        """Return the canonical page-level sample record."""
        effective = sample.effective_label() or self
        elements = [element.to_dict() for element in effective.elements]
        return {
            "id": sample.sample_id,
            "image_path": str(sample.pre_frame_path),
            "page": effective.page.to_dict() if effective.page else {},
            "route_label": effective.route_label.to_dict() if effective.route_label else {},
            "elements": elements,
            "raw_event": effective._event_payload(sample),
            "metadata": sample.metadata or {},
        }

    def to_routing_record(
        self,
        sample: "HarvestSample",
        *,
        include_legacy: bool = False,
    ) -> dict[str, Any]:
        """Return the routing-focused sample record used by active exports and LLM payloads."""
        effective = sample.effective_label() or self
        page = effective.page.to_dict() if effective.page else {}
        route = effective.route_label.to_dict() if effective.route_label else {}
        record = {
            "id": sample.sample_id,
            "image_path": str(sample.pre_frame_path),
            "page": {
                "screen_type": page.get("screen_type"),
                "situation_id": page.get("situation_id"),
            },
            "route_label": {
                "primitive_id": route.get("primitive_id"),
                "roi_name": route.get("roi_name"),
                "roi_bbox_norm": route.get("roi_bbox_norm"),
            },
            "confidence": float(effective.confidence),
            "raw_event": effective._event_payload(sample),
            "metadata": self._routing_metadata(sample),
        }
        if include_legacy:
            record["legacy"] = {
                "elements": [element.to_dict() for element in effective.elements],
                "candidates": [c.to_dict() if isinstance(c, BBoxCandidate) else c for c in effective.candidates],
                "evidence": effective.evidence,
            }
        return record

    def _event_payload(self, sample: "HarvestSample") -> dict[str, Any] | None:
        if sample.event is None:
            return None
        return {
            "action_type": sample.event.action.value,
            "mouse_button": sample.event.button,
            "key": sample.event.key,
            "x": sample.event.x,
            "y": sample.event.y,
        }

    def _routing_metadata(self, sample: "HarvestSample") -> dict[str, Any]:
        md = dict(sample.metadata or {})
        out: dict[str, Any] = {}
        for key in ("session_id", "run_config", "pipeline_version"):
            if key in md:
                out[key] = md[key]

        capture = md.get("capture")
        if isinstance(capture, dict):
            capture_out = {}
            for key in ("resolution", "ui_scale", "monitor_index", "screen_type_hint", "roi_crops"):
                if key in capture:
                    capture_out[key] = capture[key]
            if capture_out:
                out["capture"] = capture_out

        coordinates = md.get("coordinates")
        if isinstance(coordinates, dict):
            coord_out = {}
            for key in ("event_normalized_xy", "event_xy", "window_xyxy"):
                if key in coordinates:
                    coord_out[key] = coordinates[key]
            if coord_out:
                out["coordinates"] = coord_out

        filter_md = md.get("filter")
        if isinstance(filter_md, dict):
            out["filter"] = {
                key: filter_md.get(key)
                for key in ("flags", "score", "cluster_id", "cluster_representative")
                if key in filter_md
            }

        return out

    def _bootstrap_nested(self) -> None:
        if self.page is None:
            self.page = PageLabel(
                screen_type=self.screen_type,
                situation_id=self.situation_id,
                confidence=self.confidence,
            )
        if not self.elements and self._has_legacy_target():
            element = ActionableElementLabel(
                element_id="elem_001",
                bbox_x_min=self.bbox_x_min,
                bbox_y_min=self.bbox_y_min,
                bbox_x_max=self.bbox_x_max,
                bbox_y_max=self.bbox_y_max,
                semantic_id=self.semantic_id,
                semantic_text=self.semantic_text,
                function_id=self.function_id,
                hotkeys=list(self.hotkeys),
                available_actions=list(self.available_actions),
                is_route_target=True,
                confidence=self.confidence,
            )
            self.elements = [element]
        if self.route_label is None:
            primary = self.primary_element
            self.route_label = RouteLabel(
                primitive_id=self.function_id,
                target_element_id=primary.element_id if primary is not None else None,
                confidence=self.confidence,
            )

    def _has_legacy_target(self) -> bool:
        if any(v not in (0.0, 1.0) for v in (self.bbox_x_min, self.bbox_y_min, self.bbox_x_max, self.bbox_y_max)):
            return True
        if self.semantic_id or self.semantic_text:
            return True
        if self.hotkeys or self.available_actions:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        self.sync_legacy_fields()
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
        if d.get("page") is not None and isinstance(d["page"], dict):
            d["page"] = PageLabel(**d["page"])
        if d.get("route_label") is not None and isinstance(d["route_label"], dict):
            d["route_label"] = RouteLabel(**d["route_label"])
        d["elements"] = [
            ActionableElementLabel.from_dict(e) if isinstance(e, dict) else e
            for e in d.get("elements", [])
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
    metadata: dict[str, Any] | None = None

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

    @property
    def metadata_path(self) -> Path:
        return self.sample_dir / "metadata.json"

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

    def save_metadata(self) -> None:
        if self.metadata is None:
            return
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False),
            encoding="utf-8",
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

        metadata = None
        metadata_path = sample_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text("utf-8"))

        return cls(
            sample_id=sample_id,
            sample_dir=sample_dir,
            event=event,
            label=label,
            review_status=review_status,
            review_corrections=review_corrections,
            metadata=metadata,
        )

    def effective_label(self) -> LabelResult | None:
        """Return the reviewed label when edits exist, otherwise the stored label."""
        if self.label is None:
            return None
        if self.review_status != ReviewStatus.edited or not self.review_corrections:
            return self.label

        data = copy.deepcopy(self.label.to_dict())
        corrections = self.review_corrections

        if "page" in corrections and isinstance(corrections["page"], dict):
            data["page"] = corrections["page"]
        else:
            page = dict(data.get("page") or {})
            for key in ("screen_type", "situation_id", "state_flags"):
                if key in corrections:
                    page[key] = corrections[key]
            data["page"] = page

        if "route_label" in corrections and isinstance(corrections["route_label"], dict):
            data["route_label"] = corrections["route_label"]
        else:
            route = dict(data.get("route_label") or {})
            route_map = {
                "primitive_id": "primitive_id",
                "target_element_id": "target_element_id",
                "roi_name": "roi_name",
                "roi_bbox_norm": "roi_bbox_norm",
                "trigger_modality": "trigger_modality",
                "trigger_action_type": "trigger_action_type",
                "trigger_mouse_button": "trigger_mouse_button",
                "trigger_key": "trigger_key",
            }
            for src_key, dst_key in route_map.items():
                if src_key in corrections:
                    route[dst_key] = corrections[src_key]
            data["route_label"] = route

        if "elements" in corrections and isinstance(corrections["elements"], list):
            data["elements"] = corrections["elements"]
        else:
            elements = [dict(e) for e in data.get("elements", [])]
            if elements:
                idx = 0
                for i, element in enumerate(elements):
                    if element.get("is_route_target"):
                        idx = i
                        break
                element = dict(elements[idx])
                if "bbox_xyxy" in corrections:
                    x1, y1, x2, y2 = corrections["bbox_xyxy"]
                    element["bbox_x_min"] = x1
                    element["bbox_y_min"] = y1
                    element["bbox_x_max"] = x2
                    element["bbox_y_max"] = y2
                element_keys = (
                    "semantic_id",
                    "semantic_text",
                    "function_id",
                    "hotkeys",
                    "available_actions",
                    "enabled_state",
                    "roi_name",
                )
                for key in element_keys:
                    if key in corrections:
                        element[key] = corrections[key]
                elements[idx] = element
                data["elements"] = elements

        return LabelResult.from_dict(data)
