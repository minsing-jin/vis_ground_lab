"""Tests for runtime.failure_store module."""

from __future__ import annotations

from PIL import Image

from vis_ground_lab.base import ActionableElement, BoundingBox
from vis_ground_lab.runtime.failure_store import FailureSample, FailureStore


def _make_sample(frame_id: str = "fail_001") -> FailureSample:
    return FailureSample(
        frame_id=frame_id,
        image_path=f"/tmp/{frame_id}.png",
        timestamp_ms=1000.0,
        elements=(
            ActionableElement(
                class_name="button",
                bbox=BoundingBox(10, 20, 50, 60),
                score=0.2,
                center=(30.0, 40.0),
                semantic_id="button_r0c0",
                affordances=("click",),
                element_type="button",
            ),
        ),
        failure_reason="low_confidence",
        observed_at="2026-01-01T00:00:00Z",
    )


def test_save_and_load(tmp_path):
    store = FailureStore(tmp_path / "failures")
    sample = _make_sample()
    store.save_failure(sample)

    loaded = store.load_failures()
    assert len(loaded) == 1
    assert loaded[0].frame_id == "fail_001"
    assert loaded[0].failure_reason == "low_confidence"
    assert len(loaded[0].elements) == 1


def test_save_with_image(tmp_path):
    store = FailureStore(tmp_path / "failures")
    sample = _make_sample("img_fail")
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    store.save_failure(sample, image=img)

    assert (tmp_path / "failures" / "images" / "img_fail.png").exists()


def test_count(tmp_path):
    store = FailureStore(tmp_path / "failures")
    assert store.count() == 0

    store.save_failure(_make_sample("f1"))
    store.save_failure(_make_sample("f2"))
    assert store.count() == 2


def test_load_with_limit(tmp_path):
    store = FailureStore(tmp_path / "failures")
    for i in range(5):
        store.save_failure(_make_sample(f"f{i}"))

    loaded = store.load_failures(limit=3)
    assert len(loaded) == 3


def test_export_for_review(tmp_path):
    from vis_ground_lab.hitl.review_queue import ReviewQueue

    store = FailureStore(tmp_path / "failures")
    store.save_failure(_make_sample("f1"))
    store.save_failure(_make_sample("f2"))

    queue = ReviewQueue(tmp_path / "queue")
    exported = store.export_for_review(queue)
    assert exported == 2
    assert queue.pending_count() == 2


def test_empty_store(tmp_path):
    store = FailureStore(tmp_path / "failures")
    assert store.load_failures() == []
    assert store.count() == 0
