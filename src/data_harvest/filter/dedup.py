"""Deduplication filter using pHash from vis_ground_lab.data.dedup."""

from __future__ import annotations

import logging

from PIL import Image

from data_harvest.core.types import HarvestSample

logger = logging.getLogger(__name__)


def deduplicate_samples(
    samples: list[HarvestSample],
    hash_threshold: int = 8,
) -> tuple[list[HarvestSample], list[HarvestSample]]:
    """Remove near-duplicate samples based on pre-frame pHash.

    Returns (kept, dropped).
    Reuses phash_from_image and hamming_distance from vis_ground_lab.data.dedup.
    """
    from vis_ground_lab.data.dedup import phash_from_image, hamming_distance

    kept: list[HarvestSample] = []
    dropped: list[HarvestSample] = []
    seen_hashes: list[int] = []

    for s in samples:
        if not s.pre_frame_path.exists():
            dropped.append(s)
            continue
        img = Image.open(s.pre_frame_path)
        h = phash_from_image(img)

        is_dup = False
        for prev_h in seen_hashes:
            if hamming_distance(h, prev_h) < hash_threshold:
                is_dup = True
                break

        if is_dup:
            dropped.append(s)
        else:
            seen_hashes.append(h)
            kept.append(s)

    logger.info("Dedup: kept=%d, dropped=%d", len(kept), len(dropped))
    return kept, dropped
