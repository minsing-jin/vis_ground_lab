"""Unified JSONL exporter for routing-first harvest samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def _is_duplicate_non_representative(sample: HarvestSample) -> bool:
    metadata = sample.metadata or {}
    filter_md = metadata.get("filter", {}) if isinstance(metadata, dict) else {}
    if not isinstance(filter_md, dict):
        return False
    flags = filter_md.get("flags", [])
    return isinstance(flags, list) and "duplicate_non_representative" in flags


def export_unified(samples: list[HarvestSample], out_path: str | Path) -> Path:
    """Export one canonical routing record per sample."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            if sample.review_status == ReviewStatus.rejected:
                continue
            if _is_duplicate_non_representative(sample):
                continue
            label = sample.effective_label()
            if label is None:
                continue
            f.write(json.dumps(label.to_routing_record(sample, include_legacy=False), ensure_ascii=False) + "\n")
            exported += 1

    logger.info("Exported %d unified samples to %s", exported, out_path)
    return out_path
