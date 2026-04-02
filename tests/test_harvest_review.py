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
        cfg = ReviewConfig(auto_approve_confidence=0.9, enable_auto_approve=True)
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
        cfg = ReviewConfig(auto_approve_confidence=0.99, enable_auto_approve=False)
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
        q.edit(
            sample3,
            {
                "page": {"screen_type": "main_map", "situation_id": "waiting_for_next_turn", "state_flags": []},
                "route_label": {"primitive_id": "END_TURN", "target_element_id": "elem_001"},
                "elements": [
                    {
                        "element_id": "elem_001",
                        "bbox_x_min": 5,
                        "bbox_y_min": 5,
                        "bbox_x_max": 45,
                        "bbox_y_max": 45,
                        "semantic_id": "btn_end_turn",
                        "semantic_text": "End Turn button",
                        "function_id": "END_TURN",
                        "available_actions": ["click"],
                        "hotkeys": ["SHIFT+ENTER"],
                        "is_route_target": True,
                    }
                ],
            },
        )
        assert sample3.review_status == ReviewStatus.edited
        assert sample3.review_corrections is not None
        assert sample3.review_corrections["route_label"]["primitive_id"] == "END_TURN"

    def test_priority_prefers_conflict_flag(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.99)
        q = HarvestReviewQueue(cfg)

        normal = _make_sample(tmp_path, "normal", 0.4)
        flagged = _make_sample(tmp_path, "flagged", 0.6)
        flagged.metadata = {
            "filter": {
                "flags": ["candidate_conflict"],
                "cluster_representative": True,
            }
        }

        q.load([normal, flagged])
        first = q.next_sample()
        assert first is not None
        assert first.sample_id == "flagged"

    def test_priority_prefers_evidence_conflict(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.99)
        q = HarvestReviewQueue(cfg)

        normal = _make_sample(tmp_path, "normal", 0.45)
        conflicted = _make_sample(tmp_path, "conflicted", 0.6)
        conflicted.label = LabelResult(
            confidence=0.6,
            evidence={
                "conflict_pair": ["popup_primitive", "policy_primitive"],
                "triggered_hard_negatives": ["policy cards visible"],
                "open_screen_detected": True,
            },
            page={"screen_type": "popup", "situation_id": "generic_popup_or_entry_prompt_visible"},
            route_label={"primitive_id": "popup_primitive"},
        )

        q.load([normal, conflicted])
        first = q.next_sample()
        assert first is not None
        assert first.sample_id == "conflicted"

    def test_duplicate_non_representative_is_excluded_from_queue(self, tmp_path: Path):
        cfg = ReviewConfig(auto_approve_confidence=0.99)
        q = HarvestReviewQueue(cfg)

        representative = _make_sample(tmp_path, "representative", 0.4)
        duplicate = _make_sample(tmp_path, "duplicate", 0.2)
        duplicate.metadata = {
            "filter": {
                "flags": ["duplicate_non_representative"],
                "cluster_representative": False,
            }
        }

        q.load([representative, duplicate])
        assert q.pending_count == 1
        first = q.next_sample()
        assert first is not None
        assert first.sample_id == "representative"
