"""Civilization VI game profile."""

from __future__ import annotations

from data_harvest.profiles.base_profile import GameProfile
from data_harvest.profiles.registry import register_profile


@register_profile
class Civ6Profile(GameProfile):
    """Profile for Sid Meier's Civilization VI."""

    @property
    def name(self) -> str:
        return "civ6"

    @property
    def display_name(self) -> str:
        return "Civilization VI"

    @property
    def semantic_dict(self) -> dict[str, str]:
        return {
            "btn_end_turn": "End Turn button",
            "btn_research": "Research (tech/civic) button",
            "btn_production": "Production queue button",
            "btn_gold": "Gold/Treasury display",
            "btn_faith": "Faith display",
            "btn_culture": "Culture display",
            "btn_science": "Science display",
            "btn_diplomacy": "Diplomacy ribbon button",
            "btn_great_people": "Great People button",
            "btn_government": "Government button",
            "btn_religion": "Religion lens button",
            "unit_action": "Unit action panel button",
            "minimap_click": "Minimap interaction",
            "city_banner": "City banner click",
            "notification": "Notification popup",
            "menu_item": "Menu item selection",
        }

    @property
    def screen_types(self) -> list[str]:
        return [
            "main_map",
            "city_view",
            "tech_tree",
            "civic_tree",
            "diplomacy",
            "great_people",
            "government",
            "religion",
            "espionage",
            "world_congress",
            "loading",
            "main_menu",
        ]

    @property
    def ocr_languages(self) -> list[str]:
        return ["en"]

    @property
    def roi_hints(self) -> dict[str, list[float]]:
        return {
            "minimap": [0.0, 0.75, 0.2, 1.0],
            "top_bar": [0.0, 0.0, 1.0, 0.04],
            "end_turn": [0.85, 0.85, 1.0, 1.0],
            "unit_panel": [0.0, 0.55, 0.25, 0.75],
            "notifications": [0.85, 0.0, 1.0, 0.5],
        }
