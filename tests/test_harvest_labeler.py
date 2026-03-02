"""Tests for data_harvest labeler modules (click_proximity, diff_detector, fusion)."""

from __future__ import annotations

import numpy as np
import pytest

from data_harvest.labeler.click_proximity import click_proximity_bbox
from data_harvest.labeler.diff_detector import diff_bboxes, diff_ratio
from data_harvest.labeler.transition_detector import is_screen_transition


class TestClickProximity:
    def test_finds_bbox_near_rectangle(self):
        # Create a frame with a white rectangle on black background
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[80:120, 120:180] = 255  # White rectangle

        result = click_proximity_bbox(frame, click_x=150.0, click_y=100.0, crop_radius=80, contour_min_area=50)
        assert result is not None
        assert result.signal == "click_proximity"
        # The bbox should roughly cover the white rectangle area
        assert result.x_min >= 100
        assert result.x_max <= 200

    def test_returns_none_for_empty(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = click_proximity_bbox(frame, click_x=50.0, click_y=50.0, crop_radius=40, contour_min_area=50)
        assert result is None


class TestDiffDetector:
    def test_detects_changed_region(self):
        pre = np.zeros((200, 300, 3), dtype=np.uint8)
        post = np.zeros((200, 300, 3), dtype=np.uint8)
        post[50:100, 100:200] = 255  # Changed region

        bboxes = diff_bboxes(pre, post, threshold=0.02, contour_min_area=50)
        assert len(bboxes) >= 1
        assert bboxes[0].signal == "diff"

    def test_no_diff_returns_empty(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bboxes = diff_bboxes(frame, frame.copy(), threshold=0.02)
        assert len(bboxes) == 0

    def test_diff_ratio_identical(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assert diff_ratio(frame, frame.copy()) == 0.0

    def test_diff_ratio_different(self):
        pre = np.zeros((100, 100, 3), dtype=np.uint8)
        post = np.ones((100, 100, 3), dtype=np.uint8) * 255
        ratio = diff_ratio(pre, post)
        assert ratio > 0.9

    def test_diff_shape_mismatch(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((200, 200, 3), dtype=np.uint8)
        assert diff_bboxes(a, b) == []
        assert diff_ratio(a, b) == 1.0


class TestTransitionDetector:
    def test_detects_transition(self):
        pre = np.zeros((100, 100, 3), dtype=np.uint8)
        post = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert is_screen_transition(pre, post, max_diff_ratio=0.4) is True

    def test_no_transition(self):
        pre = np.zeros((100, 100, 3), dtype=np.uint8)
        post = pre.copy()
        post[5:10, 5:10] = 200  # Small change
        assert is_screen_transition(pre, post, max_diff_ratio=0.4) is False
