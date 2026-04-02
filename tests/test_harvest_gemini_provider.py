"""Tests for Gemini routing prompt construction."""

from __future__ import annotations

from data_harvest.llm.gemini_provider import GeminiProvider


def test_build_prompt_includes_visual_rubric_and_excludes_image_b64():
    prompt = GeminiProvider._build_prompt(  # noqa: SLF001
        {
            "sample_id": "sample_000001",
            "pre_image_b64": "AAAABBBB",
            "event": {"action": "click", "x": 100, "y": 200},
            "metadata": {"resolution": [1920, 1080]},
            "current_label": {"page": {"screen_type": "popup", "situation_id": "generic_popup_or_entry_prompt_visible"}},
            "routing_taxonomy": {
                "primitives": {
                    "popup_primitive": {
                        "must_have_visuals": ["generic popup only"],
                        "hard_negatives": ["tech tree open"],
                    }
                },
                "situations": {"generic_popup_or_entry_prompt_visible": {"allowed_primitives": ["popup_primitive"]}},
                "rois": {"popup_center": [0.18, 0.16, 0.82, 0.84]},
            },
            "ontology": {"btn_accept_popup": "Popup confirm button"},
        }
    )

    assert "must_have_visuals" in prompt
    assert "hard_negatives" in prompt
    assert "If a dedicated full screen or dedicated panel is actually open" in prompt
    assert "policy change popup belongs to policy_primitive" in prompt
    assert "Return up to 3 ranked candidates" in prompt
    assert "AAAABBBB" not in prompt
    assert "\"ontology\"" not in prompt


def test_to_result_preserves_ranked_candidates():
    parsed = {
        "chosen": {
            "screen_type": "popup",
            "situation_id": "generic_popup_or_entry_prompt_visible",
            "primitive_id": "popup_primitive",
            "roi_name": "popup_center",
            "confidence": 0.81,
        },
        "candidates": [
            {
                "rank": 2,
                "screen_type": "government",
                "situation_id": "policy_or_government_screen_open",
                "primitive_id": "policy_primitive",
                "roi_name": "popup_center",
                "confidence": 0.74,
                "source": "gemini",
            },
            {
                "rank": 1,
                "screen_type": "popup",
                "situation_id": "generic_popup_or_entry_prompt_visible",
                "primitive_id": "popup_primitive",
                "roi_name": "bottom_right",
                "confidence": 0.81,
                "source": "gemini",
            },
            {
                "rank": 3,
                "screen_type": "tech_tree",
                "situation_id": "research_selection_screen_open",
                "primitive_id": "research_select_primitive",
                "roi_name": "popup_center",
                "confidence": 0.51,
                "source": "gemini",
            },
        ],
        "evidence": {},
    }

    result = GeminiProvider._to_result(parsed, raw_text="{}")  # noqa: SLF001

    assert result.chosen.rank == 1
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert [candidate.primitive_id for candidate in result.candidates] == [
        "popup_primitive",
        "policy_primitive",
        "research_select_primitive",
    ]
