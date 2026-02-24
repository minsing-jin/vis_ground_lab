"""Tests for strategy.data_profiler and strategy.auto_selector modules."""

from __future__ import annotations

import json

from vis_ground_lab.config.schema import DataConfig
from vis_ground_lab.strategy.auto_selector import AutoStrategySelector
from vis_ground_lab.strategy.data_profiler import DataProfile, DataProfiler


def _make_coco(tmp_path, num_images=10, num_anns_per_image=2, num_classes=2):
    """Create a minimal COCO file."""
    categories = [{"id": i + 1, "name": f"cls_{i}"} for i in range(num_classes)]
    images = []
    annotations = []
    ann_id = 1
    for img_id in range(1, num_images + 1):
        images.append({"id": img_id, "file_name": f"img_{img_id}.png", "width": 640, "height": 480})
        for _ in range(num_anns_per_image):
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": (ann_id % num_classes) + 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }), encoding="utf-8")
    return coco_path


def _make_jsonl(tmp_path, num_samples=10, num_classes=2):
    """Create a minimal JSONL file."""
    jsonl_path = tmp_path / "data.jsonl"
    lines = []
    for i in range(num_samples):
        lines.append(json.dumps({
            "image_path": f"img_{i}.png",
            "text": f"cls_{i % num_classes}",
            "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
            "width": 640,
            "height": 480,
        }))
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    return jsonl_path


def test_profile_coco(tmp_path):
    coco_path = _make_coco(tmp_path, num_images=10, num_anns_per_image=3, num_classes=2)
    profiler = DataProfiler()
    profile = profiler.profile_coco(coco_path)

    assert profile.num_images == 10
    assert profile.num_annotations == 30
    assert profile.num_classes == 2
    assert profile.avg_annotations_per_image == 3.0
    assert 0.0 <= profile.class_balance_score <= 1.0
    assert profile.estimated_complexity in ("trivial", "simple", "moderate", "complex")


def test_profile_jsonl(tmp_path):
    jsonl_path = _make_jsonl(tmp_path, num_samples=20, num_classes=3)
    profiler = DataProfiler()
    profile = profiler.profile_jsonl(jsonl_path)

    assert profile.num_images == 20
    assert profile.num_classes == 3
    assert profile.avg_annotations_per_image == 1.0


def test_complexity_trivial(tmp_path):
    coco_path = _make_coco(tmp_path, num_images=20, num_anns_per_image=1, num_classes=1)
    profiler = DataProfiler()
    profile = profiler.profile_coco(coco_path)
    assert profile.estimated_complexity == "trivial"


def test_complexity_complex(tmp_path):
    coco_path = _make_coco(tmp_path, num_images=1500, num_anns_per_image=5, num_classes=10)
    profiler = DataProfiler()
    profile = profiler.profile_coco(coco_path)
    assert profile.estimated_complexity == "complex"


def test_auto_selector_small_detection():
    profile = DataProfile(
        num_images=100,
        num_annotations=300,
        num_classes=1,
        avg_annotations_per_image=3.0,
        avg_box_area_ratio=0.05,
        class_balance_score=1.0,
        estimated_complexity="simple",
    )
    selector = AutoStrategySelector()
    strategy = selector.select(profile)
    assert strategy.backend == "yolo_ultralytics"
    assert "yolov8n" in strategy.model_name
    assert strategy.use_hpo is False


def test_auto_selector_medium_detection():
    profile = DataProfile(
        num_images=500,
        num_annotations=1500,
        num_classes=3,
        avg_annotations_per_image=3.0,
        avg_box_area_ratio=0.05,
        class_balance_score=0.9,
        estimated_complexity="moderate",
    )
    selector = AutoStrategySelector()
    strategy = selector.select(profile)
    assert strategy.backend == "yolo_ultralytics"
    assert "yolov8s" in strategy.model_name
    assert strategy.use_hpo is True
    assert strategy.suggested_n_trials == 10


def test_auto_selector_large_detection():
    profile = DataProfile(
        num_images=2000,
        num_annotations=10000,
        num_classes=8,
        avg_annotations_per_image=5.0,
        avg_box_area_ratio=0.03,
        class_balance_score=0.8,
        estimated_complexity="complex",
    )
    selector = AutoStrategySelector()
    strategy = selector.select(profile)
    assert "yolov8m" in strategy.model_name
    assert strategy.suggested_n_trials == 20


def test_auto_selector_grounding():
    profile = DataProfile(
        num_images=50,
        num_annotations=50,
        num_classes=1,
        avg_annotations_per_image=1.0,
        avg_box_area_ratio=0.1,
        class_balance_score=1.0,
        estimated_complexity="trivial",
    )
    selector = AutoStrategySelector()
    strategy = selector.select(profile)
    assert strategy.backend == "florence2"
    assert strategy.task_name == "grounding"


def test_to_train_run_config():
    profile = DataProfile(
        num_images=100,
        num_annotations=100,
        num_classes=1,
        avg_annotations_per_image=1.0,
        avg_box_area_ratio=0.05,
        class_balance_score=1.0,
        estimated_complexity="simple",
    )
    selector = AutoStrategySelector()
    strategy = selector.select(profile)

    data_cfg = DataConfig(train_jsonl="data/train.jsonl")
    config = selector.to_train_run_config(strategy, data_cfg)
    assert config.model.backend == strategy.backend
    assert config.trainer.epochs == strategy.suggested_epochs
