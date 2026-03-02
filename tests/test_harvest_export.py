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
    HarvestSample,
    LabelResult,
    ReviewStatus,
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
        confidence=confidence, semantic_text="button",
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
        assert stats.action_counts is not None
        assert stats.action_counts["click"] == 2
        assert stats.action_counts["drag"] == 1
        assert stats.avg_confidence == pytest.approx((0.9 + 0.5 + 0.3) / 3, abs=0.01)

    def test_report_output(self, tmp_path: Path):
        samples = [_make_labeled_sample(tmp_path, "s1")]
        stats = compute_stats(samples)
        report = stats.to_report()
        assert "Total samples" in report
        assert "Avg confidence" in report


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
