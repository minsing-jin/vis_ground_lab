from __future__ import annotations

import json

from PIL import Image

from vis_ground_lab.data_manager import JSONLVisualGroundingDataset


def _write_sample_dataset(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 200), color=(255, 255, 255)).save(image_path)

    jsonl_path = tmp_path / "data.jsonl"
    record = {
        "image_path": str(image_path.name),
        "prompt": "click the File button",
        "bbox": [10, 20, 50, 120],
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jsonl_path


def test_jsonl_dataset_none_normalization(tmp_path):
    jsonl_path = _write_sample_dataset(tmp_path)
    ds = JSONLVisualGroundingDataset(jsonl_path, image_root=tmp_path, normalize_mode="none")

    sample = ds[0]
    assert sample.text == "click the File button"
    assert sample.bbox.x_min == 10
    assert sample.bbox.y_min == 20
    assert sample.bbox.x_max == 50
    assert sample.bbox.y_max == 120


def test_jsonl_dataset_zero_one_normalization(tmp_path):
    jsonl_path = _write_sample_dataset(tmp_path)
    ds = JSONLVisualGroundingDataset(jsonl_path, image_root=tmp_path, normalize_mode="0-1")

    sample = ds[0]
    assert sample.bbox.x_min == 0.1
    assert sample.bbox.y_min == 0.1
    assert sample.bbox.x_max == 0.5
    assert sample.bbox.y_max == 0.6


def test_jsonl_dataset_zero_thousand_normalization(tmp_path):
    jsonl_path = _write_sample_dataset(tmp_path)
    ds = JSONLVisualGroundingDataset(jsonl_path, image_root=tmp_path, normalize_mode="0-1000")

    sample = ds[0]
    assert sample.bbox.x_min == 100.0
    assert sample.bbox.y_min == 100.0
    assert sample.bbox.x_max == 500.0
    assert sample.bbox.y_max == 600.0
