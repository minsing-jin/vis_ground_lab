from __future__ import annotations

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.cli import _to_pixel_bbox


def test_to_pixel_bbox_none():
    bbox = BoundingBox(10, 20, 30, 40)
    out = _to_pixel_bbox(bbox, normalize_mode="none", width=100, height=200)
    assert out == bbox


def test_to_pixel_bbox_zero_one():
    bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
    out = _to_pixel_bbox(bbox, normalize_mode="0-1", width=100, height=200)
    assert out.x_min == 10
    assert out.y_min == 40
    assert out.x_max == 30
    assert out.y_max == 80


def test_to_pixel_bbox_zero_thousand():
    bbox = BoundingBox(100, 100, 500, 600)
    out = _to_pixel_bbox(bbox, normalize_mode="0-1000", width=100, height=200)
    assert out.x_min == 10
    assert out.y_min == 20
    assert out.x_max == 50
    assert out.y_max == 120
