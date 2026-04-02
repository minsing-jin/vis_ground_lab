"""Tests for data_harvest.core.types."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data_harvest.core.types import (
    ActionType,
    ActionEvent,
    ActionableElementLabel,
    BBoxCandidate,
    LabelResult,
    HarvestSample,
    PageLabel,
    ReviewStatus,
    RouteLabel,
)


class TestActionEvent:
    def test_create_click(self):
        e = ActionEvent(timestamp_ms=1000.0, action=ActionType.click, x=100.0, y=200.0, button="left")
        assert e.action == ActionType.click
        assert e.x == 100.0
        assert e.button == "left"

    def test_frozen(self):
        e = ActionEvent(timestamp_ms=1000.0, action=ActionType.click)
        with pytest.raises(AttributeError):
            e.x = 50.0  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        e = ActionEvent(timestamp_ms=1234.5, action=ActionType.drag, x=10.0, y=20.0, end_x=30.0, end_y=40.0)
        d = e.to_dict()
        assert d["action"] == "drag"
        e2 = ActionEvent.from_dict(d)
        assert e2 == e

    def test_to_json(self):
        e = ActionEvent(timestamp_ms=100.0, action=ActionType.press, key="space")
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["action"] == "press"
        assert parsed["key"] == "space"


class TestBBoxCandidate:
    def test_create(self):
        b = BBoxCandidate(x_min=0, y_min=0, x_max=100, y_max=100, signal="click_proximity", confidence=0.8)
        assert b.signal == "click_proximity"
        assert b.confidence == 0.8

    def test_to_dict(self):
        b = BBoxCandidate(x_min=10, y_min=20, x_max=30, y_max=40, signal="diff", confidence=0.5, semantic_text="OK")
        d = b.to_dict()
        assert d["semantic_text"] == "OK"


class TestLabelResult:
    def test_bbox_xyxy(self):
        lr = LabelResult(bbox_x_min=10, bbox_y_min=20, bbox_x_max=30, bbox_y_max=40, confidence=0.7)
        assert lr.bbox_xyxy == [10, 20, 30, 40]
        assert lr.primary_element is not None
        assert lr.primary_element.bbox_xyxy == [10, 20, 30, 40]

    def test_roundtrip(self):
        c = BBoxCandidate(x_min=10, y_min=20, x_max=30, y_max=40, signal="diff", confidence=0.6)
        lr = LabelResult(
            bbox_x_min=10, bbox_y_min=20, bbox_x_max=30, bbox_y_max=40,
            semantic_text="button", confidence=0.8, candidates=[c],
            transition_detected=True,
        )
        d = lr.to_dict()
        lr2 = LabelResult.from_dict(d)
        assert lr2.bbox_xyxy == lr.bbox_xyxy
        assert lr2.transition_detected is True
        assert len(lr2.candidates) == 1
        assert len(lr2.elements) == 1

    def test_page_level_fields_roundtrip(self):
        lr = LabelResult(
            page=PageLabel(screen_type="main_map", situation_id="waiting_for_next_turn", confidence=0.8),
            route_label=RouteLabel(primitive_id="END_TURN", target_element_id="elem_001", confidence=0.8),
            elements=[
                ActionableElementLabel(
                    element_id="elem_001",
                    bbox_x_min=1,
                    bbox_y_min=2,
                    bbox_x_max=30,
                    bbox_y_max=40,
                    semantic_id="btn_end_turn",
                    semantic_text="End Turn button",
                    function_id="END_TURN",
                    available_actions=["click", "press"],
                    hotkeys=["SHIFT+ENTER"],
                    is_route_target=True,
                    confidence=0.8,
                )
            ],
            confidence=0.8,
        )
        loaded = LabelResult.from_dict(lr.to_dict())
        assert loaded.page is not None
        assert loaded.page.situation_id == "waiting_for_next_turn"
        assert loaded.route_label is not None
        assert loaded.route_label.primitive_id == "END_TURN"
        assert loaded.primary_element is not None
        assert loaded.primary_element.semantic_id == "btn_end_turn"


class TestHarvestSample:
    def test_paths(self):
        s = HarvestSample(sample_id="sample_000001", sample_dir=Path("/tmp/test_sample"))
        assert s.pre_frame_path == Path("/tmp/test_sample/pre.png")
        assert s.post_frame_path == Path("/tmp/test_sample/post.png")
        assert s.event_path == Path("/tmp/test_sample/event.json")

    def test_save_and_load_event(self, tmp_path: Path):
        sample_dir = tmp_path / "sample_000001"
        s = HarvestSample(sample_id="sample_000001", sample_dir=sample_dir)
        s.event = ActionEvent(timestamp_ms=500.0, action=ActionType.click, x=50.0, y=60.0)
        s.save_event()
        assert s.event_path.exists()

        loaded = HarvestSample.load(sample_dir)
        assert loaded.event is not None
        assert loaded.event.action == ActionType.click
        assert loaded.event.x == 50.0

    def test_save_and_load_label(self, tmp_path: Path):
        sample_dir = tmp_path / "sample_000002"
        s = HarvestSample(sample_id="sample_000002", sample_dir=sample_dir)
        s.label = LabelResult(bbox_x_min=10, bbox_y_min=20, bbox_x_max=30, bbox_y_max=40, confidence=0.9)
        s.save_label()
        assert s.label_path.exists()

        loaded = HarvestSample.load(sample_dir)
        assert loaded.label is not None
        assert loaded.label.confidence == 0.9

    def test_save_and_load_review(self, tmp_path: Path):
        sample_dir = tmp_path / "sample_000003"
        s = HarvestSample(sample_id="sample_000003", sample_dir=sample_dir)
        s.review_status = ReviewStatus.edited
        s.review_corrections = {
            "page": {"screen_type": "main_map", "situation_id": "waiting_for_next_turn", "state_flags": []},
            "route_label": {"primitive_id": "END_TURN", "target_element_id": "elem_001"},
            "elements": [
                {
                    "element_id": "elem_001",
                    "bbox_x_min": 1,
                    "bbox_y_min": 2,
                    "bbox_x_max": 3,
                    "bbox_y_max": 4,
                    "semantic_id": "btn_end_turn",
                    "semantic_text": "End Turn button",
                    "function_id": "END_TURN",
                    "available_actions": ["click"],
                    "hotkeys": ["SHIFT+ENTER"],
                    "is_route_target": True,
                }
            ],
        }
        s.save_review()

        loaded = HarvestSample.load(sample_dir)
        assert loaded.review_status == ReviewStatus.edited
        assert loaded.review_corrections is not None
        assert loaded.review_corrections["route_label"]["primitive_id"] == "END_TURN"

    def test_save_and_load_metadata(self, tmp_path: Path):
        sample_dir = tmp_path / "sample_000004"
        s = HarvestSample(sample_id="sample_000004", sample_dir=sample_dir)
        s.metadata = {
            "resolution": {"pre": {"width": 100, "height": 80}},
            "coordinates": {"event_normalized_xy": {"x": 0.5, "y": 0.25}},
        }
        s.save_metadata()

        loaded = HarvestSample.load(sample_dir)
        assert loaded.metadata is not None
        assert loaded.metadata["resolution"]["pre"]["width"] == 100

    def test_effective_label_uses_reviewed_page_structure(self, tmp_path: Path):
        sample_dir = tmp_path / "sample_000005"
        s = HarvestSample(sample_id="sample_000005", sample_dir=sample_dir)
        s.label = LabelResult(
            bbox_x_min=10,
            bbox_y_min=20,
            bbox_x_max=30,
            bbox_y_max=40,
            semantic_id="btn_end_turn",
            function_id="END_TURN",
            confidence=0.5,
        )
        s.review_status = ReviewStatus.edited
        s.review_corrections = {
            "page": {"screen_type": "main_map", "situation_id": "waiting_for_next_turn", "state_flags": []},
            "route_label": {"primitive_id": "END_TURN", "target_element_id": "elem_001"},
            "elements": [
                {
                    "element_id": "elem_001",
                    "bbox_x_min": 11,
                    "bbox_y_min": 22,
                    "bbox_x_max": 33,
                    "bbox_y_max": 44,
                    "semantic_id": "btn_end_turn",
                    "semantic_text": "End Turn button",
                    "function_id": "END_TURN",
                    "available_actions": ["click", "press"],
                    "hotkeys": ["SHIFT+ENTER"],
                    "is_route_target": True,
                }
            ],
        }
        effective = s.effective_label()
        assert effective is not None
        assert effective.page is not None
        assert effective.page.situation_id == "waiting_for_next_turn"
        assert effective.primary_element is not None
        assert effective.primary_element.bbox_xyxy == [11, 22, 33, 44]
