from __future__ import annotations

import json

from PIL import Image

from vis_ground_lab.data_manager import IGNORE_INDEX, JSONLVisualGroundingDataset, RouterClassificationDataset


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


def test_router_dataset_builds_vocab_and_handles_missing_aux(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_b)

    csv_path = tmp_path / "router.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sample_id,image_path,primitive_id,screen_type,situation_id,session_id",
                f"s1,{image_a.name},religion_primitive,popup,religion_choice_visible,session_01",
                f"s2,{image_b.name},unit_ops_primitive,main_map,,session_01",
            ]
        ),
        encoding="utf-8",
    )

    dataset = RouterClassificationDataset(
        source=csv_path,
        image_root=tmp_path,
        image_size=32,
        aux_label_columns=["screen_type", "situation_id"],
    )

    assert dataset.label_to_index == {"religion_primitive": 0, "unit_ops_primitive": 1}
    sample0 = dataset[0]
    sample1 = dataset[1]

    assert sample0["pixel_values"].shape == (3, 32, 32)
    assert sample0["aux_labels"]["screen_type"] == 1
    assert sample1["aux_labels"]["situation_id"] == IGNORE_INDEX
