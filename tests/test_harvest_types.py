"""Tests for data_harvest.core.types."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data_harvest.core.types import (
    ActionType,
    ActionEvent,
    BBoxCandidate,
    LabelResult,
    HarvestSample,
    ReviewStatus,
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
        s.review_corrections = {"bbox_xyxy": [1, 2, 3, 4]}
        s.save_review()

        loaded = HarvestSample.load(sample_dir)
        assert loaded.review_status == ReviewStatus.edited
        assert loaded.review_corrections == {"bbox_xyxy": [1, 2, 3, 4]}
