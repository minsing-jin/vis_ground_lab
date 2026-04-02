"""Tests for review taxonomy helpers."""

from __future__ import annotations

from data_harvest.profiles.registry import discover_profiles, get_profile
from data_harvest.review.review_app import (
    _choice_pairs,
    _candidate_choices,
    _evidence_sections,
    _format_evidence,
    _friendly_value,
    _ordered_primitive_choices,
    _primitive_suggestion,
    _render_roi_editor_html,
    _taxonomy_choices,
)
from data_harvest.core.types import HarvestSample, LabelResult, PageLabel, RouteLabel


def _profile():
    discover_profiles()
    return get_profile("civ6")


def test_taxonomy_choices_loaded_from_profile_yaml():
    profile = _profile()
    choices = _taxonomy_choices(profile)

    assert "generic_popup_or_entry_prompt_visible" in choices["situations"]
    assert "btn_accept_popup" in choices["semantics"]
    assert "policy_primitive" in choices["primitives"]
    assert "main_map" in choices["screen_types"]


def test_primitive_choices_put_router_enabled_first():
    profile = _profile()
    choices = _ordered_primitive_choices(profile)

    assert choices[0] == "religion_primitive"
    assert choices.index("war_primitive") > choices.index("policy_primitive")


def test_situation_suggests_first_allowed_primitive():
    profile = _profile()

    assert _primitive_suggestion(profile, "policy_or_government_screen_open") == "policy_primitive"
    assert _primitive_suggestion(profile, "governor_management_visible") == "governor_primitive"


def test_candidate_choices_use_route_candidate_keys():
    label = LabelResult(
        page=PageLabel(screen_type="popup", situation_id="generic_popup_or_entry_prompt_visible"),
        route_label=RouteLabel(primitive_id="popup_primitive", roi_name="popup_center"),
        evidence={
            "route_candidates": [
                {"rank": 2, "primitive_id": "policy_primitive", "situation_id": "policy_or_government_screen_open", "roi_name": "popup_center", "confidence": 0.9},
                {"rank": 1, "primitive_id": "popup_primitive", "situation_id": "generic_popup_or_entry_prompt_visible", "roi_name": "bottom_right", "confidence": 0.6},
                {"rank": 3, "primitive_id": "research_select_primitive", "situation_id": "research_selection_screen_open", "roi_name": "popup_center", "confidence": 0.4},
            ]
        },
    )

    primitive_choices = _candidate_choices(
        None,
        label,
        kind="primitives",
        taxonomy_choices=["popup_primitive", "policy_primitive"],
        current_value="popup_primitive",
    )
    situation_choices = _candidate_choices(
        None,
        label,
        kind="situations",
        taxonomy_choices=["generic_popup_or_entry_prompt_visible", "policy_or_government_screen_open"],
        current_value="generic_popup_or_entry_prompt_visible",
    )

    assert primitive_choices[:3] == ["popup_primitive", "policy_primitive", "research_select_primitive"]
    assert situation_choices[:3] == [
        "generic_popup_or_entry_prompt_visible",
        "policy_or_government_screen_open",
        "research_selection_screen_open",
    ]


def test_format_evidence_includes_visual_rule_fields():
    label = LabelResult(
        evidence={
            "matched_must_have": ["open tech tree"],
            "matched_strong_cues": ["science icon"],
            "triggered_hard_negatives": ["only entry button visible"],
            "conflict_pair": ["popup_primitive", "research_select_primitive"],
            "open_screen_detected": True,
            "reasoning": "Dedicated research screen is open.",
        }
    )

    rendered = _format_evidence(label)

    assert "must-have: open tech tree" in rendered
    assert "hard negatives: only entry button visible" in rendered
    assert "conflict: popup_primitive vs research_select_primitive" in rendered


def test_evidence_sections_split_teacher_fields():
    label = LabelResult(
        evidence={
            "matched_must_have": ["open tech tree"],
            "matched_strong_cues": ["science icon"],
            "triggered_hard_negatives": ["only entry button visible"],
            "conflict_pair": ["popup_primitive", "research_select_primitive"],
            "open_screen_detected": True,
            "reasoning": "Dedicated research screen is open.",
        }
    )

    must_have, strong_cues, hard_negatives, conflict, summary = _evidence_sections(label)

    assert must_have == "- open tech tree"
    assert strong_cues == "- science icon"
    assert hard_negatives == "- only entry button visible"
    assert conflict == "- popup_primitive vs research_select_primitive"
    assert "open screen: True" in summary


def test_friendly_value_and_choice_pairs_keep_raw_values():
    profile = _profile()

    primitive_display = _friendly_value("primitives", "policy_primitive", profile)
    situation_display = _friendly_value("situations", "policy_or_government_screen_open", profile)
    roi_choices = _choice_pairs("rois", ["popup_center"], profile)

    assert primitive_display == "정책/정부"
    assert "정책 관리 화면" in situation_display
    assert roi_choices == [("중앙 팝업", "popup_center")]


def test_render_roi_editor_html_contains_bbox_and_friendly_labels(tmp_path):
    profile = _profile()
    image_path = tmp_path / "pre.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c49444154789c63f8ffff3f0005fe02fea2572f9b0000000049454e44ae426082"
        )
    )
    sample = HarvestSample(sample_id="sample_000001", sample_dir=tmp_path, metadata={})

    rendered = _render_roi_editor_html(
        sample,
        screen_type="popup",
        situation_id="policy_or_government_screen_open",
        primitive_id="policy_primitive",
        roi_name="popup_center",
        roi_bbox=[0.1, 0.2, 0.7, 0.8],
        profile=profile,
    )

    assert 'id="harvest-roi-editor"' in rendered
    assert 'data-bbox="0.100000,0.200000,0.700000,0.800000"' in rendered
    assert "정책/정부" in rendered
    assert "중앙 팝업" in rendered
    assert "이미지 위에서 사각형을 드래그" in rendered
