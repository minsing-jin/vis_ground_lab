"""Routing dataset exporters."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2

from data_harvest.core.types import HarvestSample, ReviewStatus
from data_harvest.profiles.base_profile import GameProfile

logger = logging.getLogger(__name__)

_FIELDS = ["sample_id", "image_path", "primitive_id", "situation_id", "screen_type", "session_id"]


def _is_duplicate_non_representative(sample: HarvestSample) -> bool:
    metadata = sample.metadata or {}
    filter_md = metadata.get("filter", {}) if isinstance(metadata, dict) else {}
    if not isinstance(filter_md, dict):
        return False
    flags = filter_md.get("flags", [])
    return isinstance(flags, list) and "duplicate_non_representative" in flags


def export_router_full(samples: list[HarvestSample], out_dir: str | Path) -> Path:
    """Export full screenshots for primitive routing classification."""
    out_dir = Path(out_dir)
    rows: list[dict[str, str]] = []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        row = _build_row(sample)
        if row is None:
            continue
        img = cv2.imread(str(sample.pre_frame_path))
        if img is None:
            continue
        out_path = images_dir / f"{sample.sample_id}.png"
        cv2.imwrite(str(out_path), img)
        row["image_path"] = str(out_path)
        rows.append(row)

    _write_split_csvs(out_dir, rows)
    logger.info("Exported %d full routing samples to %s", len(rows), out_dir)
    return out_dir


def export_router_roi(
    samples: list[HarvestSample],
    out_dir: str | Path,
    *,
    profile: GameProfile | None = None,
    fallback_rois: list[str] | None = None,
) -> Path:
    """Export ROI crops for primitive routing classification."""
    out_dir = Path(out_dir)
    rows: list[dict[str, str]] = []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = fallback_rois or ["bottom_right", "popup_center", "unit_panel", "main_map"]

    for sample in samples:
        row = _build_row(sample)
        if row is None:
            continue
        img = cv2.imread(str(sample.pre_frame_path))
        if img is None:
            continue
        crop = _crop_router_roi(img, sample, profile=profile, fallback_rois=fallbacks)
        out_path = images_dir / f"{sample.sample_id}.png"
        cv2.imwrite(str(out_path), crop)
        row["image_path"] = str(out_path)
        rows.append(row)

    _write_split_csvs(out_dir, rows)
    logger.info("Exported %d ROI routing samples to %s", len(rows), out_dir)
    return out_dir


def export_router(
    samples: list[HarvestSample],
    out_dir: str | Path,
    *,
    profile: GameProfile | None = None,
    fallback_rois: list[str] | None = None,
) -> Path:
    """Compatibility wrapper exporting both routing variants."""
    out_dir = Path(out_dir)
    export_router_full(samples, out_dir / "full")
    export_router_roi(samples, out_dir / "roi", profile=profile, fallback_rois=fallback_rois)
    return out_dir


def _build_row(sample: HarvestSample) -> dict[str, str] | None:
    if sample.review_status == ReviewStatus.rejected:
        return None
    if _is_duplicate_non_representative(sample):
        return None
    label = sample.effective_label()
    if label is None or not sample.pre_frame_path.exists():
        return None
    primitive_id = label.route_label.primitive_id if label.route_label else None
    if not primitive_id:
        return None
    return {
        "sample_id": sample.sample_id,
        "image_path": "",
        "primitive_id": str(primitive_id),
        "situation_id": str(label.page.situation_id if label.page and label.page.situation_id else ""),
        "screen_type": str(label.page.screen_type if label.page and label.page.screen_type else ""),
        "session_id": _session_id(sample),
    }


def _session_id(sample: HarvestSample) -> str:
    metadata = sample.metadata or {}
    session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
    if session_id:
        return str(session_id)
    return sample.sample_dir.parent.name


def _crop_router_roi(
    img,
    sample: HarvestSample,
    *,
    profile: GameProfile | None,
    fallback_rois: list[str],
):
    h, w = img.shape[:2]
    label = sample.effective_label() or sample.label
    situation_id = label.page.situation_id if label and label.page else None
    route = label.route_label if label else None

    if route and route.roi_bbox_norm and len(route.roi_bbox_norm) == 4:
        x1n, y1n, x2n, y2n = route.roi_bbox_norm
        x1 = max(0, min(w - 1, int(round(float(x1n) * w))))
        y1 = max(0, min(h - 1, int(round(float(y1n) * h))))
        x2 = max(x1 + 1, min(w, int(round(float(x2n) * w))))
        y2 = max(y1 + 1, min(h, int(round(float(y2n) * h))))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size else img

    roi_name = route.roi_name if route else None
    if profile is not None:
        if roi_name is None:
            roi_name = profile.situation_primary_roi(situation_id)
        if roi_name is None:
            for fallback in fallback_rois:
                if fallback in profile.roi_hints:
                    roi_name = fallback
                    break

    if roi_name is None or profile is None or roi_name not in profile.roi_hints:
        return img

    x1n, y1n, x2n, y2n = profile.roi_hints[roi_name]
    x1 = max(0, min(w - 1, int(round(x1n * w))))
    y1 = max(0, min(h - 1, int(round(y1n * h))))
    x2 = max(x1 + 1, min(w, int(round(x2n * w))))
    y2 = max(y1 + 1, min(h, int(round(y2n * h))))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else img


def _write_split_csvs(out_dir: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(out_dir / "labels.csv", rows)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    session_ids = sorted(grouped)
    if len(session_ids) <= 1:
        ordered_rows = sorted(rows, key=lambda row: _sample_sort_key(row.get("sample_id", "")))
        if len(ordered_rows) <= 1:
            train_rows = ordered_rows
            val_rows = []
        else:
            val_count = max(1, int(round(len(ordered_rows) * 0.1)))
            val_rows = ordered_rows[-val_count:]
            train_rows = ordered_rows[:-val_count]
    else:
        val_count = max(1, int(round(len(session_ids) * 0.1)))
        val_sessions = set(session_ids[-val_count:])
        train_rows = [row for row in rows if row["session_id"] not in val_sessions]
        val_rows = [row for row in rows if row["session_id"] in val_sessions]
    _write_csv(out_dir / "train.csv", train_rows)
    _write_csv(out_dir / "val.csv", val_rows)


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    suffix = sample_id.rsplit("_", 1)[-1]
    try:
        return int(suffix), sample_id
    except ValueError:
        return 0, sample_id


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
