"""FilterPipeline: 4-stage sequential filter → FilterResult."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from data_harvest.core.config import FilterConfig
from data_harvest.core.types import HarvestSample
from data_harvest.filter.invalid_action import is_invalid_action
from data_harvest.filter.quality import has_quality_issue
from data_harvest.filter.dedup import deduplicate_samples

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Summary of filter pipeline execution."""

    total_input: int = 0
    kept: list[HarvestSample] = field(default_factory=list)
    dropped_invalid: int = 0
    dropped_quality: int = 0
    dropped_transition: int = 0
    dropped_dedup: int = 0

    @property
    def total_dropped(self) -> int:
        return self.dropped_invalid + self.dropped_quality + self.dropped_transition + self.dropped_dedup

    @property
    def total_kept(self) -> int:
        return len(self.kept)


class FilterPipeline:
    """Runs 4 stages sequentially: invalid_action → quality → transition → dedup."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def run(self, samples: list[HarvestSample]) -> FilterResult:
        result = FilterResult(total_input=len(samples))
        remaining = list(samples)

        # Stage 1: Invalid action (no visible change)
        stage1 = []
        for s in remaining:
            if is_invalid_action(s, min_diff_ratio=self.config.min_diff_ratio):
                result.dropped_invalid += 1
            else:
                stage1.append(s)
        remaining = stage1
        logger.info("Stage 1 (invalid_action): %d → %d", len(samples), len(remaining))

        # Stage 2: Quality (blur + dark)
        stage2 = []
        for s in remaining:
            if has_quality_issue(
                s,
                blur_threshold=self.config.blur_laplacian_threshold,
                dark_threshold=self.config.dark_overlay_threshold,
            ):
                result.dropped_quality += 1
            else:
                stage2.append(s)
        remaining = stage2
        logger.info("Stage 2 (quality): %d → %d", len(stage1), len(remaining))

        # Stage 3: Transition (large screen changes)
        stage3 = []
        for s in remaining:
            if s.label is not None and s.label.transition_detected:
                result.dropped_transition += 1
            else:
                stage3.append(s)
        remaining = stage3
        logger.info("Stage 3 (transition): %d → %d", len(stage2), len(remaining))

        # Stage 4: Dedup
        kept, dropped = deduplicate_samples(
            remaining, hash_threshold=self.config.dedup_hash_threshold
        )
        result.dropped_dedup = len(dropped)
        result.kept = kept
        logger.info("Stage 4 (dedup): %d → %d", len(remaining), len(kept))

        logger.info(
            "FilterPipeline: %d → %d (dropped: inv=%d, qual=%d, trans=%d, dup=%d)",
            result.total_input,
            result.total_kept,
            result.dropped_invalid,
            result.dropped_quality,
            result.dropped_transition,
            result.dropped_dedup,
        )
        return result
