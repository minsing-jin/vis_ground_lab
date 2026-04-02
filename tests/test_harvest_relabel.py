"""Tests for harvest relabel utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
)
from data_harvest.labeler.relabel import apply_relabel_result, build_sample_payload, map_semantic_id
from data_harvest.profiles.registry import discover_profiles, get_profile
from data_harvest.llm.provider import RelabelCandidate, RelabelResult


def _sample(tmp_path: Path, sid: str = "s1") -> HarvestSample:
    d = tmp_path / sid
    d.mkdir(parents=True)
    img = np.ones((60, 80, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(d / "pre.png"), img)
    cv2.imwrite(str(d / "post.png"), img)
    s = HarvestSample(sample_id=sid, sample_dir=d)
    s.event = ActionEvent(timestamp_ms=1.0, action=ActionType.click, x=20, y=30)
    s.label = LabelResult(bbox_x_min=10, bbox_y_min=10, bbox_x_max=30, bbox_y_max=30, confidence=0.4)
    return s


def test_build_payload_contains_event_and_routing_taxonomy(tmp_path: Path):
    s = _sample(tmp_path)
    discover_profiles()
    profile = get_profile("civ6")
    p = build_sample_payload(s, profile=profile, include_image_b64="abc")
    assert p["event"]["action"] == "click"
    assert "routing_taxonomy" in p
    assert "primitives" in p["routing_taxonomy"]
    assert p["pre_image_b64"] == "abc"


def test_map_semantic_id_strict_unmapped():
    sid, flags = map_semantic_id("unknown", "not known", {"btn_a": "A button"}, strict=True)
    assert sid is None
    assert "ontology_unmapped" in flags


def test_map_semantic_id_by_description_match():
    sid, flags = map_semantic_id(None, "End Turn button", {"btn_end_turn": "End Turn button"}, strict=True)
    assert sid == "btn_end_turn"
    assert flags == []


def test_apply_relabel_result_updates_label(tmp_path: Path):
    s = _sample(tmp_path)
    rr = RelabelResult(
        chosen=RelabelCandidate(
            bbox_xyxy=[1, 2, 40, 50],
            semantic_text="Popup confirm",
            semantic_id="btn_accept_popup",
            function_id="popup_primitive",
            primitive_id="popup_primitive",
            screen_type="popup",
            situation_id="generic_popup_or_entry_prompt_visible",
            roi_name="popup_center",
            action="click",
            confidence=0.91,
            source="gemini",
        ),
        candidates=[
            RelabelCandidate(
                rank=1,
                bbox_xyxy=[1, 2, 40, 50],
                semantic_text="Popup confirm",
                semantic_id="btn_accept_popup",
                function_id="popup_primitive",
                primitive_id="popup_primitive",
                situation_id="generic_popup_or_entry_prompt_visible",
                roi_name="popup_center",
                confidence=0.91,
                source="gemini",
            )
        ],
        evidence={
            "matched_must_have": ["generic popup visible"],
            "matched_strong_cues": ["confirm button"],
            "triggered_hard_negatives": [],
            "conflict_pair": ["popup_primitive", "policy_primitive"],
            "open_screen_detected": False,
            "reasoning": "Generic popup only.",
        },
    )
    apply_relabel_result(s, rr)
    assert s.label is not None
    assert s.label.elements == []
    assert s.label.candidates == []
    assert s.label.route_label is not None
    assert s.label.route_label.primitive_id == "popup_primitive"
    assert s.label.route_label.roi_name == "popup_center"
    assert s.label.page is not None
    assert s.label.page.situation_id == "generic_popup_or_entry_prompt_visible"
    assert s.label.page.screen_type == "popup"
    assert s.label.evidence["route_candidates"][0]["rank"] == 1
