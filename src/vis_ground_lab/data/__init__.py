"""Data utilities for extraction, deduplication, and COCO IO."""

from vis_ground_lab.data.coco import (
    add_annotation_entry,
    add_image_entry,
    coco_bbox_to_xyxy,
    empty_coco,
    load_coco,
    register_categories,
    save_coco,
)
from vis_ground_lab.data.dedup import deduplicate_images, hamming_distance, phash_from_image
from vis_ground_lab.data.extract import extract_frames

__all__ = [
    "extract_frames",
    "phash_from_image",
    "hamming_distance",
    "deduplicate_images",
    "empty_coco",
    "save_coco",
    "load_coco",
    "register_categories",
    "add_image_entry",
    "add_annotation_entry",
    "coco_bbox_to_xyxy",
]
