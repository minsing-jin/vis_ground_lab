"""Provider interface and typed outputs for relabeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RelabelCandidate:
    rank: int | None = None
    bbox_xyxy: list[float] | None = None
    semantic_text: str | None = None
    semantic_id: str | None = None
    function_id: str | None = None
    primitive_id: str | None = None
    screen_type: str | None = None
    situation_id: str | None = None
    roi_name: str | None = None
    roi_bbox_norm: list[float] | None = None
    action: str | None = None
    confidence: float = 0.0
    source: str = "llm"


@dataclass
class RelabelResult:
    chosen: RelabelCandidate
    candidates: list[RelabelCandidate] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Protocol for relabel providers."""

    def relabel(self, sample_payload: dict[str, Any]) -> RelabelResult:
        ...
