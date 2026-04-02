"""Tests for data_harvest export modules."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    ActionableElementLabel,
    HarvestSample,
    LabelResult,
    PageLabel,
    ReviewStatus,
    RouteLabel,
)
from data_harvest.export.stats import compute_stats


def _make_labeled_sample(
    tmp_path: Path,
    sample_id: str,
    confidence: float = 0.8,
    action: ActionType = ActionType.click,
    review_status: ReviewStatus = ReviewStatus.pending,
) -> HarvestSample:
    sample_dir = tmp_path / sample_id
    sample_dir.mkdir(parents=True)
    # Create pre/post frames
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(sample_dir / "pre.png"), frame)
    cv2.imwrite(str(sample_dir / "post.png"), frame)

    s = HarvestSample(sample_id=sample_id, sample_dir=sample_dir)
    s.event = ActionEvent(timestamp_ms=100.0, action=action, x=50.0, y=50.0)
    s.save_event()
    s.label = LabelResult(
        bbox_x_min=20, bbox_y_min=20, bbox_x_max=80, bbox_y_max=80,
        confidence=confidence,
        semantic_text="button",
        semantic_id="btn_end_turn",
        function_id="END_TURN",
        page=PageLabel(screen_type="main_map", situation_id="waiting_for_next_turn", confidence=confidence),
        route_label=RouteLabel(primitive_id="END_TURN", target_element_id="elem_001", confidence=confidence),
        elements=[
            ActionableElementLabel(
                element_id="elem_001",
                bbox_x_min=20,
                bbox_y_min=20,
                bbox_x_max=80,
                bbox_y_max=80,
                semantic_id="btn_end_turn",
                semantic_text="End Turn button",
                function_id="END_TURN",
                hotkeys=["SHIFT+ENTER"],
                available_actions=["click", "press"],
                is_route_target=True,
                confidence=confidence,
            )
        ],
    )
    s.save_label()
    s.review_status = review_status
    return s


class TestStats:
    def test_compute_stats(self, tmp_path: Path):
        samples = [
            _make_labeled_sample(tmp_path, "s1", confidence=0.9, review_status=ReviewStatus.approved),
            _make_labeled_sample(tmp_path, "s2", confidence=0.5, review_status=ReviewStatus.pending),
            _make_labeled_sample(tmp_path, "s3", confidence=0.3, action=ActionType.drag, review_status=ReviewStatus.rejected),
        ]
        stats = compute_stats(samples)
        assert stats.total_samples == 3
        assert stats.labeled_samples == 3
        assert stats.approved == 1
        assert stats.rejected == 1
        assert stats.pending_review == 1
        assert stats.avg_confidence == pytest.approx((0.9 + 0.5 + 0.3) / 3, abs=0.01)
        assert stats.primitive_counts is not None
        assert stats.primitive_counts["END_TURN"] == 3
        assert stats.situation_counts is not None
        assert stats.situation_counts["waiting_for_next_turn"] == 3

    def test_report_output(self, tmp_path: Path):
        samples = [_make_labeled_sample(tmp_path, "s1")]
        stats = compute_stats(samples)
        report = stats.to_report()
        assert "Total samples" in report
        assert "Primitive distribution" in report


class TestGroundingExporter:
    def test_export_grounding(self, tmp_path: Path):
        from data_harvest.export.grounding_exporter import export_grounding

        samples = [
            _make_labeled_sample(tmp_path, "s1", confidence=0.8, review_status=ReviewStatus.approved),
            _make_labeled_sample(tmp_path, "s2", confidence=0.3, review_status=ReviewStatus.rejected),
        ]
        out = export_grounding(samples, tmp_path / "grounding.jsonl", normalizing_range=1000)
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1  # Rejected sample excluded
        rec = json.loads(lines[0])
        assert "image_path" in rec
        assert "bbox" in rec
        assert len(rec["bbox"]) == 4
        assert rec["confidence"] == pytest.approx(0.8, abs=1e-6)
        assert rec["function_id"] == "END_TURN"

    def test_export_grounding_uses_review_corrections(self, tmp_path: Path):
        from data_harvest.export.grounding_exporter import export_grounding

        s = _make_labeled_sample(tmp_path, "s1", confidence=0.8, review_status=ReviewStatus.edited)
        s.review_corrections = {
            "page": {"screen_type": "main_map", "situation_id": "needs_research_choice", "state_flags": []},
            "route_label": {"primitive_id": "CHOOSE_RESEARCH", "target_element_id": "elem_001"},
            "elements": [
                {
                    "element_id": "elem_001",
                    "bbox_x_min": 10,
                    "bbox_y_min": 10,
                    "bbox_x_max": 50,
                    "bbox_y_max": 50,
                    "semantic_text": "Choose Research",
                    "semantic_id": "btn_choose_research",
                    "function_id": "CHOOSE_RESEARCH",
                    "available_actions": ["click"],
                    "hotkeys": [],
                    "is_route_target": True,
                }
            ],
        }
        s.save_review()

        out = export_grounding([s], tmp_path / "grounding.jsonl", normalizing_range=1000)
        rec = json.loads(out.read_text().strip())
        assert rec["bbox"] == [100, 100, 500, 500]
        assert rec["semantic_id"] == "btn_choose_research"
        assert rec["primitive_id"] == "CHOOSE_RESEARCH"


class TestYoloExporter:
    def test_export_yolo(self, tmp_path: Path):
        from data_harvest.export.yolo_exporter import export_yolo

        samples = [
            _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved),
            _make_labeled_sample(tmp_path, "s2", review_status=ReviewStatus.approved),
        ]
        out_dir = tmp_path / "yolo_out"
        export_yolo(samples, out_dir)
        assert (out_dir / "data.yaml").exists()
        assert (out_dir / "images").is_dir()
        assert (out_dir / "labels").is_dir()
        # Check label file format
        label_files = list((out_dir / "labels").glob("*.txt"))
        assert len(label_files) == 2
        content = label_files[0].read_text().strip()
        parts = content.split()
        assert len(parts) == 5  # class_id cx cy w h

    def test_export_yolo_uses_corrected_semantic_id(self, tmp_path: Path):
        from data_harvest.export.yolo_exporter import export_yolo

        s = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.edited)
        s.review_corrections = {
            "elements": [
                {
                    "element_id": "elem_001",
                    "bbox_x_min": 20,
                    "bbox_y_min": 20,
                    "bbox_x_max": 40,
                    "bbox_y_max": 40,
                    "semantic_id": "save_button",
                    "semantic_text": "Save button",
                    "function_id": "SAVE",
                    "available_actions": ["click"],
                    "hotkeys": [],
                    "is_route_target": True,
                }
            ]
        }
        s.save_review()

        out_dir = tmp_path / "yolo_out"
        export_yolo([s], out_dir, category_names=["ui_element", "save_button"])
        label = (out_dir / "labels" / "s1.txt").read_text().strip()
        class_id = int(label.split()[0])
        assert class_id == 1


class TestCocoExporter:
    def test_export_coco(self, tmp_path: Path):
        from data_harvest.export.coco_exporter import export_coco

        samples = [
            _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved),
        ]
        out = export_coco(samples, tmp_path / "coco.json")
        assert out.exists()
        coco = json.loads(out.read_text())
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
        assert len(coco["categories"]) == 1

    def test_export_coco_uses_corrected_semantic_id(self, tmp_path: Path):
        from data_harvest.export.coco_exporter import export_coco

        s = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.edited)
        s.review_corrections = {
            "elements": [
                {
                    "element_id": "elem_001",
                    "bbox_x_min": 20,
                    "bbox_y_min": 20,
                    "bbox_x_max": 80,
                    "bbox_y_max": 80,
                    "semantic_id": "save_button",
                    "semantic_text": "Save button",
                    "function_id": "SAVE",
                    "available_actions": ["click"],
                    "hotkeys": [],
                    "is_route_target": True,
                }
            ]
        }
        s.save_review()

        out = export_coco([s], tmp_path / "coco.json", category_names=["ui_element", "save_button"])
        coco = json.loads(out.read_text())
        assert coco["annotations"][0]["category_id"] == 2


class TestRouterExporter:
    def test_single_session_split_uses_time_block_holdout(self, tmp_path: Path):
        from data_harvest.export.router_exporter import export_router_full

        samples = [
            _make_labeled_sample(tmp_path, "sample_000001", review_status=ReviewStatus.approved),
            _make_labeled_sample(tmp_path, "sample_000002", review_status=ReviewStatus.approved),
            _make_labeled_sample(tmp_path, "sample_000003", review_status=ReviewStatus.approved),
        ]

        out_dir = tmp_path / "router_full"
        export_router_full(samples, out_dir)

        train_lines = (out_dir / "train.csv").read_text(encoding="utf-8").strip().splitlines()
        val_lines = (out_dir / "val.csv").read_text(encoding="utf-8").strip().splitlines()

        assert len(train_lines) == 3  # header + 2 rows
        assert len(val_lines) == 2  # header + 1 row
        assert "sample_000003" in val_lines[-1]


class TestRoiStateExporter:
    def test_export_roi_state(self, tmp_path: Path):
        from data_harvest.export.roi_state_exporter import export_roi_state

        s = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved)
        s.review_corrections = {"turn_state": "NEXT_TURN"}
        s.save_review()

        out_dir = tmp_path / "roi_state"
        out = export_roi_state([s], out_dir)
        assert out.exists()
        assert (out_dir / "images" / "s1.png").exists()
        assert (out_dir / "labels.csv").exists()


class TestRouterAndUnifiedExporter:
    def test_export_router(self, tmp_path: Path):
        from data_harvest.export.router_exporter import export_router

        s = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved)
        out_dir = export_router([s], tmp_path / "router")
        assert (out_dir / "full" / "images" / "s1.png").exists()
        assert (out_dir / "full" / "labels.csv").exists()
        assert (out_dir / "roi" / "images" / "s1.png").exists()
        assert (out_dir / "roi" / "labels.csv").exists()

    def test_export_unified(self, tmp_path: Path):
        from data_harvest.export.unified_exporter import export_unified

        s = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved)
        out = export_unified([s], tmp_path / "unified.jsonl")
        record = json.loads(out.read_text().strip())
        assert record["route_label"]["primitive_id"] == "END_TURN"
        assert record["page"]["situation_id"] == "waiting_for_next_turn"
        assert "legacy" not in record

    def test_export_router_excludes_duplicate_non_representative(self, tmp_path: Path):
        from data_harvest.export.router_exporter import export_router_full

        representative = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved)
        duplicate = _make_labeled_sample(tmp_path, "s2", review_status=ReviewStatus.approved)
        duplicate.metadata = {
            "filter": {
                "flags": ["duplicate_non_representative"],
                "cluster_id": "cluster_1",
                "cluster_representative": False,
            }
        }
        duplicate.save_metadata()

        out_dir = export_router_full([representative, duplicate], tmp_path / "router_full")
        labels = (out_dir / "labels.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(labels) == 2
        assert "s1.png" in labels[1]
        assert "s2.png" not in labels[1]

    def test_export_unified_excludes_duplicate_non_representative(self, tmp_path: Path):
        from data_harvest.export.unified_exporter import export_unified

        representative = _make_labeled_sample(tmp_path, "s1", review_status=ReviewStatus.approved)
        duplicate = _make_labeled_sample(tmp_path, "s2", review_status=ReviewStatus.approved)
        duplicate.metadata = {
            "filter": {
                "flags": ["duplicate_non_representative"],
                "cluster_id": "cluster_1",
                "cluster_representative": False,
            }
        }
        duplicate.save_metadata()

        out = export_unified([representative, duplicate], tmp_path / "unified.jsonl")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["id"] == "s1"
