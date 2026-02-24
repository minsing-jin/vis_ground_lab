"""Tests for hitl.review_queue module."""

from __future__ import annotations

from vis_ground_lab.hitl.review_queue import ReviewItem, ReviewQueue


def _make_item(frame_id: str, score: float, source: str = "auto_capture") -> ReviewItem:
    return ReviewItem(
        image_path=f"/tmp/{frame_id}.png",
        frame_id=frame_id,
        elements=[{"class_name": "button", "bbox": [0, 0, 10, 10], "score": 0.5}],
        uncertainty_score=score,
        source=source,
        timestamp="2026-01-01T00:00:00Z",
    )


def test_enqueue_and_peek(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    queue.enqueue(_make_item("f1", 0.9))
    queue.enqueue(_make_item("f2", 0.5))
    queue.enqueue(_make_item("f3", 0.7))

    items = queue.peek(n=2)
    assert len(items) == 2
    # Sorted by uncertainty desc
    assert items[0].frame_id == "f1"
    assert items[1].frame_id == "f3"


def test_mark_reviewed(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    queue.enqueue(_make_item("f1", 0.9))
    queue.enqueue(_make_item("f2", 0.5))

    corrections = {"boxes": [{"class_name": "button", "bbox": [0, 0, 20, 20]}]}
    queue.mark_reviewed("f1", corrections)

    items = queue.peek(n=10)
    assert len(items) == 1
    assert items[0].frame_id == "f2"


def test_stats(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    queue.enqueue(_make_item("f1", 0.9))
    queue.enqueue(_make_item("f2", 0.5))
    queue.mark_reviewed("f1", {"boxes": []})

    stats = queue.stats()
    assert stats["total"] == 2
    assert stats["reviewed"] == 1
    assert stats["pending"] == 1
    assert stats["with_corrections"] == 1


def test_pending_count(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    queue.enqueue(_make_item("f1", 0.9))
    queue.enqueue(_make_item("f2", 0.5))
    assert queue.pending_count() == 2

    queue.mark_reviewed("f1")
    assert queue.pending_count() == 1


def test_empty_queue(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    assert queue.peek() == []
    assert queue.stats() == {"total": 0, "reviewed": 0, "pending": 0, "with_corrections": 0}


def test_export_corrections_as_coco(tmp_path):
    queue = ReviewQueue(tmp_path / "queue")
    queue.enqueue(_make_item("f1", 0.9))
    queue.mark_reviewed("f1", {"boxes": [{"class_name": "button", "bbox": [0, 0, 10, 10]}]})

    # Create a dummy image for the image path so add_image_entry can read it
    from PIL import Image

    img_path = tmp_path / "f1.png"
    Image.new("RGB", (100, 100)).save(img_path)

    # Update the queue item to point to the real image
    items = queue._load_all()
    items[0]["image_path"] = str(img_path)
    queue._save_all(items)

    out = tmp_path / "corrections.coco.json"
    queue.export_corrections_as_coco(out, class_names=["button"])
    assert out.exists()

    import json
    coco = json.loads(out.read_text(encoding="utf-8"))
    assert len(coco["annotations"]) == 1
