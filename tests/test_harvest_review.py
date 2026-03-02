"""Tests for data_harvest review queue."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_harvest.core.config import ReviewConfig
from data_harvest.core.types import (
    ActionEvent,
    ActionType,
    HarvestSample,
    LabelResult,
    ReviewStatus,
)
from data_harvest.review.queue import HarvestReviewQueue


def _make_sample(tmp_path: Path, sid: str, confidence: float) -> HarvestSample:
    d = tmp_path / sid
    d.mkdir(parents=True)
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(d / "pre.png"), frame)
    cv2.imwrite(str(d / "post.png"), frame)

    s = HarvestSample(sample_id=sid, sample_dir=d)
    s.event = ActionEvent(timestamp_ms=100.0, action=ActionType.click, x=25.0, y=25.0)
    s.save_event()
    s.label = LabelResult(bbox_x_min=10, bbox_y_min=10, bbox_x_max=40, bbox_y_max=40, confidence=confidence)
    s.save_label()
    return s


class TestReviewQueue:
    def test_auto_approve_high_confidence(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.9)
        q = HarvestReviewQueue(cfg)

        high = _make_sample(tmp_path, "high", 0.95)
        low = _make_sample(tmp_path, "low", 0.5)
        q.load([high, low])

        assert high.review_status == ReviewStatus.approved
        assert q.pending_count == 1

    def test_sorted_by_confidence(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.99)
        q = HarvestReviewQueue(cfg)

        s1 = _make_sample(tmp_path, "s1", 0.7)
        s2 = _make_sample(tmp_path, "s2", 0.3)
        s3 = _make_sample(tmp_path, "s3", 0.5)
        q.load([s1, s2, s3])

        # Should come out lowest confidence first
        first = q.next_sample()
        assert first is not None
        assert first.sample_id == "s2"

    def test_approve_reject_edit(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.99)
        q = HarvestReviewQueue(cfg)

        s = _make_sample(tmp_path, "s1", 0.5)
        q.load([s])

        sample = q.next_sample()
        assert sample is not None

        q.approve(sample)
        assert sample.review_status == ReviewStatus.approved

        s2 = _make_sample(tmp_path, "s2", 0.4)
        q.load([s2])
        sample2 = q.next_sample()
        assert sample2 is not None
        q.reject(sample2)
        assert sample2.review_status == ReviewStatus.rejected

        s3 = _make_sample(tmp_path, "s3", 0.3)
        q.load([s3])
        sample3 = q.next_sample()
        assert sample3 is not None
        q.edit(sample3, {"bbox_xyxy": [5, 5, 45, 45]})
        assert sample3.review_status == ReviewStatus.edited
        assert sample3.review_corrections == {"bbox_xyxy": [5, 5, 45, 45]}
