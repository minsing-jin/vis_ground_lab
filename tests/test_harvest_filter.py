"""Tests for data_harvest filter modules."""

from __future__ import annotations

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
from data_harvest.filter.invalid_action import is_invalid_action
from data_harvest.filter.quality import is_blurry, is_dark_overlay, has_quality_issue


def _make_sample_with_frames(
    tmp_path: Path,
    sample_id: str,
    pre: np.ndarray,
    post: np.ndarray,
) -> HarvestSample:
    sample_dir = tmp_path / sample_id
    sample_dir.mkdir(parents=True)
    cv2.imwrite(str(sample_dir / "pre.png"), pre)
    cv2.imwrite(str(sample_dir / "post.png"), post)

    s = HarvestSample(sample_id=sample_id, sample_dir=sample_dir)
    s.event = ActionEvent(timestamp_ms=100.0, action=ActionType.click, x=50.0, y=50.0)
    s.save_event()
    return s


class TestInvalidAction:
    def test_identical_frames_invalid(self, tmp_path: Path):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        s = _make_sample_with_frames(tmp_path, "s1", frame, frame.copy())
        assert is_invalid_action(s, min_diff_ratio=0.005) is True

    def test_different_frames_valid(self, tmp_path: Path):
        pre = np.zeros((100, 100, 3), dtype=np.uint8)
        post = np.zeros((100, 100, 3), dtype=np.uint8)
        post[20:80, 20:80] = 255
        s = _make_sample_with_frames(tmp_path, "s2", pre, post)
        assert is_invalid_action(s, min_diff_ratio=0.005) is False


class TestQuality:
    def test_blurry_detection(self):
        # Very blurry (uniform)
        blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assert is_blurry(blurry, laplacian_threshold=50.0) is True

        # Sharp edges
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[40:60, :] = 255
        assert is_blurry(sharp, laplacian_threshold=50.0) is False

    def test_dark_overlay(self):
        dark = np.zeros((100, 100, 3), dtype=np.uint8)
        assert is_dark_overlay(dark, threshold=30.0) is True

        bright = np.ones((100, 100, 3), dtype=np.uint8) * 200
        assert is_dark_overlay(bright, threshold=30.0) is False

    def test_has_quality_issue(self, tmp_path: Path):
        dark = np.zeros((100, 100, 3), dtype=np.uint8)
        s = _make_sample_with_frames(tmp_path, "q1", dark, dark)
        assert has_quality_issue(s) is True
