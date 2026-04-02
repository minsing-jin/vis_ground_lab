"""Civilization VI game profile backed by a single taxonomy YAML."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from data_harvest.profiles.base_profile import GameProfile
from data_harvest.profiles.registry import register_profile

logger = logging.getLogger(__name__)

_FALLBACK_DISPLAY_NAME = "Civilization VI"
_FALLBACK_OCR_LANGUAGES = ["en"]
_FALLBACK_SCREEN_TYPES = [
    "main_map",
    "city_view",
    "tech_tree",
    "civic_tree",
    "diplomacy",
    "government",
    "religion_screen",
    "governor_screen",
    "world_congress",
    "popup",
    "loading",
    "main_menu",
]
_FALLBACK_ELEMENTS: dict[str, dict[str, object]] = {
    "btn_end_turn": {
        "semantic_text": "End Turn / Next Turn button",
        "function_id": "popup_primitive",
        "hotkeys": ["SHIFT+ENTER"],
        "available_actions": ["click", "press"],
        "roi_hint": "bottom_right",
        "aliases": ["End Turn", "Next Turn"],
    },
    "btn_choose_research": {
        "semantic_text": "Choose Research button",
        "function_id": "popup_primitive",
        "hotkeys": [],
        "available_actions": ["click"],
        "roi_hint": "bottom_right",
        "aliases": ["Choose Research"],
    },
    "btn_choose_civic": {
        "semantic_text": "Choose Civic button",
        "function_id": "culture_decision_primitive",
        "hotkeys": [],
        "available_actions": ["click"],
        "roi_hint": "bottom_right",
        "aliases": ["Choose Civic"],
    },
    "btn_choose_production": {
        "semantic_text": "Choose Production button",
        "function_id": "popup_primitive",
        "hotkeys": [],
        "available_actions": ["click"],
        "roi_hint": "unit_panel",
        "aliases": ["Choose Production"],
    },
    "unit_action_button": {
        "semantic_text": "Unit action button",
        "function_id": "unit_ops_primitive",
        "hotkeys": [],
        "available_actions": ["click"],
        "roi_hint": "unit_panel",
        "aliases": ["Unit Action"],
    },
}
_FALLBACK_PRIMITIVES: dict[str, dict[str, object]] = {
    "religion_primitive": {"router_enabled": True, "criteria": "종교관 선택 화면"},
    "governor_primitive": {"router_enabled": True, "criteria": "총독 관리 화면"},
    "voting_primitive": {"router_enabled": True, "criteria": "세계의회 투표 화면"},
    "era_primitive": {"router_enabled": True, "criteria": "시대 전략 선택 화면"},
    "unit_ops_primitive": {"router_enabled": True, "criteria": "비전투 일반 유닛 조작 화면"},
    "research_select_primitive": {"router_enabled": True, "criteria": "기술 트리 또는 연구 목록 화면"},
    "city_production_primitive": {"router_enabled": True, "criteria": "도시 생산 목록 화면"},
    "culture_decision_primitive": {"router_enabled": True, "criteria": "사회 제도 트리 화면"},
    "diplomatic_primitive": {"router_enabled": True, "criteria": "도시국가 사절 화면"},
    "combat_primitive": {"router_enabled": True, "criteria": "즉시 전투 판단 화면"},
    "policy_primitive": {"router_enabled": True, "criteria": "정책 또는 정부 관리 화면"},
    "popup_primitive": {"router_enabled": True, "criteria": "일반 팝업 또는 진입 버튼 화면"},
    "war_primitive": {"router_enabled": True, "criteria": "전쟁 선포 화면"},
    "deal_primitive": {"router_enabled": True, "criteria": "거래 화면"},
}
_FALLBACK_SITUATIONS: dict[str, dict[str, object]] = {
    "religion_choice_visible": {"allowed_primitives": ["religion_primitive"], "screen_types": ["popup", "religion_screen"]},
    "governor_management_visible": {"allowed_primitives": ["governor_primitive"], "screen_types": ["popup", "governor_screen"]},
    "world_congress_vote_visible": {"allowed_primitives": ["voting_primitive"], "screen_types": ["world_congress", "popup"]},
    "era_dedication_visible": {"allowed_primitives": ["era_primitive"], "screen_types": ["popup", "era_screen"]},
    "noncombat_unit_ops_visible": {"allowed_primitives": ["unit_ops_primitive"], "screen_types": ["main_map"]},
    "research_selection_screen_open": {"allowed_primitives": ["research_select_primitive"], "screen_types": ["main_map", "tech_tree"]},
    "city_production_screen_open": {"allowed_primitives": ["city_production_primitive"], "screen_types": ["main_map", "city_view"]},
    "civic_selection_screen_open": {"allowed_primitives": ["culture_decision_primitive"], "screen_types": ["main_map", "civic_tree"]},
    "diplomatic_envoy_screen_open": {"allowed_primitives": ["diplomatic_primitive"], "screen_types": ["diplomacy", "city_state_screen"]},
    "combat_decision_visible": {"allowed_primitives": ["combat_primitive"], "screen_types": ["main_map"]},
    "policy_or_government_screen_open": {"allowed_primitives": ["policy_primitive"], "screen_types": ["government", "popup"]},
    "generic_popup_or_entry_prompt_visible": {"allowed_primitives": ["popup_primitive"], "screen_types": ["main_map", "popup"]},
    "war_declaration_screen_open": {"allowed_primitives": ["war_primitive"], "screen_types": ["diplomacy", "war_screen"]},
    "deal_negotiation_screen_open": {"allowed_primitives": ["deal_primitive"], "screen_types": ["diplomacy", "deal_screen"]},
}
_FALLBACK_ROIS = {
    "bottom_right": [0.78, 0.72, 1.0, 1.0],
    "unit_panel": [0.0, 0.52, 0.32, 0.86],
    "popup_center": [0.18, 0.16, 0.82, 0.84],
    "main_map": [0.0, 0.08, 1.0, 1.0],
}


@register_profile
class Civ6Profile(GameProfile):
    """Profile for Sid Meier's Civilization VI."""

    @property
    def name(self) -> str:
        return "civ6"

    @property
    def display_name(self) -> str:
        profile = self._taxonomy.get("profile", {})
        if isinstance(profile, dict) and profile.get("display_name"):
            return str(profile["display_name"])
        return _FALLBACK_DISPLAY_NAME

    @cached_property
    def _taxonomy_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "configs" / "harvest_taxonomy" / f"{self.name}.yaml"

    @cached_property
    def _taxonomy(self) -> dict[str, Any]:
        path = self._taxonomy_path
        if not path.exists():
            logger.warning("Taxonomy file not found for profile '%s': %s", self.name, path)
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Failed to load taxonomy file: %s", path, exc_info=True)
            return {}
        if not isinstance(data, dict):
            logger.warning("Taxonomy file has unexpected top-level type: %s", path)
            return {}
        return data

    @property
    def semantic_dict(self) -> dict[str, str]:
        return {
            semantic_id: str(spec.get("semantic_text") or semantic_id)
            for semantic_id, spec in self.element_catalog.items()
        }

    @property
    def element_catalog(self) -> dict[str, dict[str, object]]:
        items = self._taxonomy.get("elements", [])
        if not isinstance(items, list):
            return dict(_FALLBACK_ELEMENTS)

        catalog: dict[str, dict[str, object]] = {}
        for raw in items:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            semantic_id = str(raw["id"])
            semantic_text = str(raw.get("display_name") or semantic_id)
            aliases = [str(v) for v in raw.get("text_aliases", []) if v is not None]
            aliases.extend(str(v) for v in raw.get("icon_aliases", []) if v is not None)
            function_id = raw.get("function_id") or raw.get("function_id_or_related_primitive")
            catalog[semantic_id] = {
                "semantic_text": semantic_text,
                "function_id": str(function_id) if function_id is not None else None,
                "hotkeys": [str(v) for v in raw.get("hotkeys", []) if v is not None],
                "available_actions": [str(v) for v in raw.get("available_actions", ["click"]) if v is not None],
                "roi_hint": str(raw["roi_hint"]) if raw.get("roi_hint") is not None else None,
                "aliases": aliases,
            }
        return catalog or dict(_FALLBACK_ELEMENTS)

    @property
    def primitive_dict(self) -> dict[str, dict[str, object]]:
        items = self._taxonomy.get("primitives", [])
        if not isinstance(items, list):
            return dict(_FALLBACK_PRIMITIVES)

        primitives: dict[str, dict[str, object]] = {}
        for raw in items:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            primitive_id = str(raw["id"])
            primitives[primitive_id] = {k: v for k, v in raw.items() if k != "id"}
        return primitives or dict(_FALLBACK_PRIMITIVES)

    @property
    def situation_dict(self) -> dict[str, dict[str, object]]:
        items = self._taxonomy.get("situations", [])
        if not isinstance(items, list):
            return dict(_FALLBACK_SITUATIONS)

        situations: dict[str, dict[str, object]] = {}
        for raw in items:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            situation_id = str(raw["id"])
            situations[situation_id] = {k: v for k, v in raw.items() if k != "id"}
        return situations or dict(_FALLBACK_SITUATIONS)

    @property
    def screen_types(self) -> list[str]:
        profile = self._taxonomy.get("profile", {})
        if isinstance(profile, dict):
            declared = profile.get("screen_types", [])
            if isinstance(declared, list) and declared:
                return [str(v) for v in declared if v is not None]

        discovered: list[str] = []
        for spec in self.situation_dict.values():
            for screen_type in spec.get("screen_types", []):
                if screen_type is not None:
                    discovered.append(str(screen_type))
        return sorted(set(discovered or _FALLBACK_SCREEN_TYPES))

    @property
    def ocr_languages(self) -> list[str]:
        profile = self._taxonomy.get("profile", {})
        if isinstance(profile, dict):
            langs = profile.get("ocr_languages", [])
            if isinstance(langs, list) and langs:
                return [str(v) for v in langs if v is not None]
        return list(_FALLBACK_OCR_LANGUAGES)

    @property
    def roi_hints(self) -> dict[str, list[float]]:
        items = self._taxonomy.get("rois", [])
        if not isinstance(items, list):
            return dict(_FALLBACK_ROIS)

        rois: dict[str, list[float]] = {}
        for raw in items:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            roi_id = str(raw["id"])
            norm_xyxy = raw.get("norm_xyxy", [])
            if not isinstance(norm_xyxy, list) or len(norm_xyxy) != 4:
                continue
            rois[roi_id] = [float(v) for v in norm_xyxy]
        return rois or dict(_FALLBACK_ROIS)
