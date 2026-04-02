"""GameProfile ABC: defines game-specific semantics, screen types, OCR config, ROI hints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GameProfile(ABC):
    """Abstract base class for game-specific configuration profiles."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique profile identifier (e.g., 'civ6')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name (e.g., 'Civilization VI')."""

    @property
    @abstractmethod
    def semantic_dict(self) -> dict[str, str]:
        """Mapping of semantic IDs to human-readable descriptions.

        Example: {"btn_end_turn": "End Turn button", "btn_research": "Research button"}
        """

    @property
    @abstractmethod
    def screen_types(self) -> list[str]:
        """Known screen types for this game.

        Example: ["main_map", "city_view", "tech_tree", "diplomacy", "loading"]
        """

    @property
    def ocr_languages(self) -> list[str]:
        """OCR languages for this game (default: English)."""
        return ["en"]

    @property
    def roi_hints(self) -> dict[str, list[float]]:
        """Region-of-interest hints: named_region → [x_min, y_min, x_max, y_max] (normalized 0-1).

        Example: {"minimap": [0.0, 0.75, 0.2, 1.0], "top_bar": [0.0, 0.0, 1.0, 0.05]}
        """
        return {}

    @property
    def element_catalog(self) -> dict[str, dict[str, Any]]:
        """Actionable element catalog keyed by semantic_id."""
        return {
            semantic_id: {
                "semantic_text": desc,
                "function_id": None,
                "hotkeys": [],
                "available_actions": ["click"],
                "roi_hint": None,
                "aliases": [],
            }
            for semantic_id, desc in self.semantic_dict.items()
        }

    @property
    def primitive_dict(self) -> dict[str, dict[str, Any]]:
        """Primitive catalog keyed by primitive_id."""
        return {}

    @property
    def router_primitive_dict(self) -> dict[str, dict[str, Any]]:
        """Router-enabled primitive catalog preserving declared order."""
        return {
            primitive_id: spec
            for primitive_id, spec in self.primitive_dict.items()
            if bool(spec.get("router_enabled", True))
        }

    @property
    def situation_dict(self) -> dict[str, dict[str, Any]]:
        """Situation catalog keyed by situation_id."""
        return {}

    def situation_primary_roi(self, situation_id: str | None) -> str | None:
        """Return the highest-priority ROI for a situation, if declared."""
        if not situation_id:
            return None
        spec = self.situation_dict.get(situation_id, {})
        roi_priority = spec.get("roi_priority", [])
        if isinstance(roi_priority, list) and roi_priority:
            return str(roi_priority[0])
        likely_rois = spec.get("likely_rois", [])
        if isinstance(likely_rois, list) and likely_rois:
            return str(likely_rois[0])
        return None

    def infer_primitive_from_semantic(self, semantic_id: str | None) -> str | None:
        """Infer route primitive from the actionable element semantic ID."""
        if not semantic_id:
            return None
        element = self.element_catalog.get(semantic_id, {})
        function_id = element.get("function_id")
        return str(function_id) if function_id else None

    def infer_situation_from_primitive(self, primitive_id: str | None) -> str | None:
        """Infer page situation from the active primitive."""
        if not primitive_id:
            return None
        for situation_id, spec in self.situation_dict.items():
            allowed = spec.get("allowed_primitives", [])
            if primitive_id in allowed:
                return situation_id
        return None

    def classify_screen(self, frame: Any) -> str | None:
        """Optionally classify the current screen type. Override for game-specific logic."""
        return None
