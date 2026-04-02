"""Routing-first local fallback labeler."""

from __future__ import annotations

from typing import Any

import cv2

from data_harvest.core.config import LabelerConfig
from data_harvest.core.types import ActionEvent, ActionType, HarvestSample, LabelResult, PageLabel, RouteLabel
from data_harvest.profiles.base_profile import GameProfile


class AutoLabeler:
    """Build a minimal routing label when the primary teacher is unavailable."""

    def __init__(self, config: LabelerConfig, profile: GameProfile | None = None) -> None:
        self.config = config
        self.profile = profile

    def label_sample(self, sample: HarvestSample) -> LabelResult | None:
        if sample.event is None or not sample.pre_frame_path.exists():
            return None

        frame = cv2.imread(str(sample.pre_frame_path))
        if frame is None:
            return None

        event = sample.event
        primitive_id, primitive_signals = self._infer_primitive(sample, frame, event)
        situation_id, situation_signals = self._infer_situation(primitive_id, sample, frame, event)
        screen_type = self._infer_screen_type(situation_id, frame)

        signals = primitive_signals + situation_signals
        confidence = 0.0
        if primitive_id:
            confidence += 0.35
        if situation_id:
            confidence += 0.25
        if screen_type:
            confidence += 0.10
        if "hotkey_match" in signals:
            confidence += 0.20
        if "roi_match" in signals:
            confidence += 0.15
        confidence = min(confidence, 0.75)

        page = PageLabel(
            screen_type=screen_type,
            situation_id=situation_id,
            state_flags=[],
            confidence=confidence,
        )
        route = RouteLabel(
            primitive_id=primitive_id,
            target_element_id=None,
            roi_name=self.profile.situation_primary_roi(situation_id) if self.profile and situation_id else None,
            trigger_modality="keyboard" if event.action in (ActionType.press, ActionType.type) else "mouse",
            trigger_action_type=event.action.value,
            trigger_mouse_button=event.button,
            trigger_key=event.key,
            confidence=confidence,
        )
        return LabelResult(
            screen_type=screen_type,
            situation_id=situation_id,
            function_id=primitive_id,
            confidence=confidence,
            evidence={
                "mode": "local_routing_fallback",
                "signals_used": signals,
                "event": event.to_dict(),
                "matched_rois": sorted(self._event_rois(sample, frame)),
            },
            page=page,
            route_label=route,
            elements=[],
            candidates=[],
        )

    def _infer_primitive(
        self,
        sample: HarvestSample,
        frame,
        event: ActionEvent,
    ) -> tuple[str | None, list[str]]:
        signals: list[str] = []
        if self.profile is None:
            return None, signals

        hotkey_primitive = self._infer_from_hotkey(event)
        if hotkey_primitive:
            signals.append("hotkey_match")
            return hotkey_primitive, signals

        roi_hits = self._event_rois(sample, frame)
        if roi_hits:
            candidates: list[str] = []
            for situation_id, spec in self.profile.situation_dict.items():
                allowed = [str(value) for value in spec.get("allowed_primitives", []) if value is not None]
                if not allowed:
                    continue
                roi_priority = [str(value) for value in spec.get("roi_priority", []) if value is not None]
                if roi_priority and any(roi in roi_hits for roi in roi_priority):
                    candidates.extend(allowed)
            if len(candidates) == 1:
                signals.append("roi_match")
                return candidates[0], signals
            if candidates:
                signals.append("roi_ambiguous")
                return candidates[0], signals

        if event.action in (ActionType.press, ActionType.type):
            key = (event.key or "").strip().upper()
            if key in {"ENTER", "RETURN", "ESC", "ESCAPE"} and "popup_primitive" in self.profile.router_primitive_dict:
                signals.append("default_popup_key")
                return "popup_primitive", signals

        return None, signals

    def _infer_situation(
        self,
        primitive_id: str | None,
        sample: HarvestSample,
        frame,
        event: ActionEvent,
    ) -> tuple[str | None, list[str]]:
        signals: list[str] = []
        if self.profile is None or not primitive_id:
            return None, signals

        roi_hits = self._event_rois(sample, frame)
        matching: list[str] = []
        for situation_id, spec in self.profile.situation_dict.items():
            allowed = [str(value) for value in spec.get("allowed_primitives", []) if value is not None]
            if primitive_id not in allowed:
                continue
            roi_priority = [str(value) for value in spec.get("roi_priority", []) if value is not None]
            if roi_priority and roi_hits and any(roi in roi_hits for roi in roi_priority):
                matching.append(situation_id)

        if matching:
            signals.append("roi_match")
            return matching[0], signals

        inferred = self.profile.infer_situation_from_primitive(primitive_id)
        if inferred:
            signals.append("primitive_default")
        return inferred, signals

    def _infer_screen_type(self, situation_id: str | None, frame) -> str | None:
        if self.profile is None:
            return None
        classified = self.profile.classify_screen(frame)
        if classified:
            return classified
        if not situation_id:
            return None
        spec = self.profile.situation_dict.get(situation_id, {})
        screen_types = spec.get("screen_types", [])
        if isinstance(screen_types, list) and screen_types:
            return str(screen_types[0])
        return None

    def _infer_from_hotkey(self, event: ActionEvent) -> str | None:
        if self.profile is None or not event.key:
            return None
        key_upper = event.key.upper()
        for element in self.profile.element_catalog.values():
            for hotkey in element.get("hotkeys", []):
                hotkey_text = str(hotkey).upper()
                if key_upper == hotkey_text or key_upper in hotkey_text:
                    function_id = element.get("function_id")
                    return str(function_id) if function_id else None
        return None

    def _event_rois(self, sample: HarvestSample, frame) -> set[str]:
        if self.profile is None:
            return set()
        metadata = sample.metadata or {}
        coords = metadata.get("coordinates", {}) if isinstance(metadata, dict) else {}
        norm = coords.get("event_normalized_xy", {}) if isinstance(coords, dict) else {}

        x = norm.get("x") if isinstance(norm, dict) else None
        y = norm.get("y") if isinstance(norm, dict) else None
        if x is None or y is None:
            event = sample.event
            if event is None or event.x is None or event.y is None:
                return set()
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                return set()
            x = float(event.x) / float(w)
            y = float(event.y) / float(h)

        hits: set[str] = set()
        for roi_name, roi_xyxy in self.profile.roi_hints.items():
            x1, y1, x2, y2 = roi_xyxy
            if x1 <= float(x) <= x2 and y1 <= float(y) <= y2:
                hits.add(roi_name)
        return hits
