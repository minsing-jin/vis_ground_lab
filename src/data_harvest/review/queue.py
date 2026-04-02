"""HarvestReviewQueue: confidence-sorted queue with auto-approve."""

from __future__ import annotations

import logging

from data_harvest.core.config import ReviewConfig
from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


def _is_duplicate_non_representative(sample: HarvestSample) -> bool:
    metadata = sample.metadata or {}
    filter_md = metadata.get("filter", {}) if isinstance(metadata, dict) else {}
    if not isinstance(filter_md, dict):
        return False
    flags = filter_md.get("flags", [])
    return isinstance(flags, list) and "duplicate_non_representative" in flags


class HarvestReviewQueue:
    """Manages a review queue sorted by confidence, with auto-approve for high confidence."""

    def __init__(self, config: ReviewConfig) -> None:
        self.config = config
        self._queue: list[HarvestSample] = []

    def load(self, samples: list[HarvestSample]) -> None:
        """Load labeled samples into the review queue.

        High-confidence samples are auto-approved.
        Remaining are sorted by confidence ascending (lowest first = most uncertain).
        """
        auto_approved = 0
        needs_review: list[HarvestSample] = []

        for s in samples:
            if s.label is None:
                continue
            if s.review_status != ReviewStatus.pending:
                continue
            if _is_duplicate_non_representative(s):
                continue

            label = s.effective_label() or s.label
            if self.config.enable_auto_approve and label.confidence >= self.config.auto_approve_confidence:
                s.review_status = ReviewStatus.approved
                s.save_review()
                auto_approved += 1
            else:
                needs_review.append(s)

        # Sort by confidence ascending (most uncertain first)
        needs_review.sort(key=self._priority_score)
        self._queue = needs_review

        logger.info(
            "ReviewQueue: auto-approved=%d, needs_review=%d",
            auto_approved,
            len(needs_review),
        )

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def next_sample(self) -> HarvestSample | None:
        """Pop the next sample to review (lowest confidence first)."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def approve(self, sample: HarvestSample) -> None:
        sample.review_status = ReviewStatus.approved
        sample.save_review()

    def reject(self, sample: HarvestSample) -> None:
        sample.review_status = ReviewStatus.rejected
        sample.save_review()

    def edit(self, sample: HarvestSample, corrections: dict) -> None:
        sample.review_status = ReviewStatus.edited
        sample.review_corrections = corrections
        sample.save_review()

    @staticmethod
    def _priority_score(sample: HarvestSample) -> float:
        """Lower score means higher review priority."""
        effective = sample.effective_label() or sample.label
        conf = effective.confidence if effective else 1.0
        md = sample.metadata or {}
        filter_md = md.get("filter", {}) if isinstance(md, dict) else {}
        flags = set(filter_md.get("flags", [])) if isinstance(filter_md, dict) else set()

        bonus = 0.0
        if "missing_primitive_id" in flags:
            bonus += 0.35
        if "missing_situation_id" in flags:
            bonus += 0.25
        if "candidate_conflict" in flags:
            bonus += 0.15
        if "taxonomy_mismatch" in flags:
            bonus += 0.15
        if filter_md.get("cluster_representative", False):
            bonus += 0.10
        if md.get("screen_novelty", False):
            bonus += 0.10
        if effective is not None and isinstance(effective.evidence, dict):
            triggered_hard_negatives = effective.evidence.get("triggered_hard_negatives", [])
            if isinstance(triggered_hard_negatives, list) and triggered_hard_negatives:
                bonus += min(0.20, 0.05 * len(triggered_hard_negatives))

            conflict_pair = effective.evidence.get("conflict_pair")
            if conflict_pair:
                bonus += 0.15

            open_screen_detected = effective.evidence.get("open_screen_detected")
            primitive_id = effective.route_label.primitive_id if effective.route_label else None
            if primitive_id == "popup_primitive" and open_screen_detected:
                bonus += 0.25

            route_candidates = effective.evidence.get("route_candidates", [])
            if isinstance(route_candidates, list) and len(route_candidates) >= 2:
                ranked = sorted(
                    [candidate for candidate in route_candidates if isinstance(candidate, dict)],
                    key=lambda candidate: float(candidate.get("confidence", 0.0)),
                    reverse=True,
                )
                if len(ranked) >= 2:
                    gap = float(ranked[0].get("confidence", 0.0)) - float(ranked[1].get("confidence", 0.0))
                    if gap < 0.10:
                        bonus += 0.20
                    elif gap < 0.20:
                        bonus += 0.10

        return max(0.0, conf - bonus)
