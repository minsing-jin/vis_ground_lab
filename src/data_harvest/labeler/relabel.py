"""Teacher-label orchestration for routing-first harvest samples."""

from __future__ import annotations

from typing import Any

from data_harvest.core.types import HarvestSample, LabelResult, PageLabel, RouteLabel
from data_harvest.llm.provider import RelabelResult
from data_harvest.profiles.base_profile import GameProfile


def build_sample_payload(
    sample: HarvestSample,
    profile: GameProfile | None = None,
    include_image_b64: str | None = None,
) -> dict[str, Any]:
    label = sample.effective_label() or sample.label
    payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "event": sample.event.to_dict() if sample.event else None,
        "metadata": sample.metadata or {},
    }
    if label is not None:
        payload["current_label"] = label.to_routing_record(sample, include_legacy=False)
    if profile is not None:
        payload["routing_taxonomy"] = {
            "screen_types": list(profile.screen_types),
            "primitives": profile.router_primitive_dict,
            "situations": profile.situation_dict,
            "rois": profile.roi_hints,
        }
        payload["ontology"] = profile.semantic_dict
    if include_image_b64:
        payload["pre_image_b64"] = include_image_b64
    return payload


def apply_relabel_result(sample: HarvestSample, result: RelabelResult) -> None:
    if sample.label is None:
        sample.label = LabelResult()

    chosen = result.chosen
    existing_page = sample.label.page or PageLabel()
    existing_route = sample.label.route_label or RouteLabel()

    screen_type = chosen.screen_type or existing_page.screen_type
    situation_id = chosen.situation_id or existing_page.situation_id
    primitive_id = chosen.primitive_id or chosen.function_id or existing_route.primitive_id
    confidence = float(chosen.confidence or sample.label.confidence or 0.0)

    sample.label.page = PageLabel(
        screen_type=screen_type,
        situation_id=situation_id,
        state_flags=[],
        confidence=confidence,
    )
    sample.label.route_label = RouteLabel(
        primitive_id=primitive_id,
        target_element_id=None,
        roi_name=chosen.roi_name,
        roi_bbox_norm=list(chosen.roi_bbox_norm) if chosen.roi_bbox_norm else None,
        trigger_action_type=sample.event.action.value if sample.event else None,
        trigger_mouse_button=sample.event.button if sample.event else None,
        trigger_key=sample.event.key if sample.event else None,
        trigger_modality="keyboard" if sample.event and sample.event.action.value in ("press", "type") else "mouse",
        confidence=confidence,
    )
    sample.label.screen_type = screen_type
    sample.label.situation_id = situation_id
    sample.label.confidence = confidence
    sample.label.evidence = {
        **result.evidence,
        "route_candidates": [
            {
                "rank": candidate.rank,
                "primitive_id": candidate.primitive_id,
                "situation_id": candidate.situation_id,
                "screen_type": candidate.screen_type,
                "roi_name": candidate.roi_name,
                "roi_bbox_norm": list(candidate.roi_bbox_norm) if candidate.roi_bbox_norm else None,
                "confidence": candidate.confidence,
                "source": candidate.source,
            }
            for candidate in result.candidates
        ],
    }

    # Active path is routing-only. Keep rich grounding fields only as legacy data.
    sample.label.elements = []
    sample.label.candidates = []
    sample.label.bbox_x_min = 0.0
    sample.label.bbox_y_min = 0.0
    sample.label.bbox_x_max = 1.0
    sample.label.bbox_y_max = 1.0
    sample.label.semantic_text = None
    sample.label.semantic_id = None
    sample.label.function_id = primitive_id
    sample.label.hotkeys = []
    sample.label.available_actions = []
    sample.label.sync_legacy_fields()


def map_semantic_id(
    proposed_id: str | None,
    proposed_text: str | None,
    ontology: dict[str, str],
    strict: bool = True,
) -> tuple[str | None, list[str]]:
    """Legacy helper retained for compatibility with older tests/callers."""
    flags: list[str] = []
    if proposed_id and proposed_id in ontology:
        return proposed_id, flags

    text = (proposed_text or "").strip().lower()
    if text:
        for sid, desc in ontology.items():
            if text == sid.lower():
                return sid, flags
            if text in desc.lower() or desc.lower() in text:
                return sid, flags

    if strict:
        flags.append("ontology_unmapped")
        return None, flags

    return proposed_id, flags
