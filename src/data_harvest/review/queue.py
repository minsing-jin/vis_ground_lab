"""HarvestReviewQueue: confidence-sorted queue with auto-approve."""

from __future__ import annotations

import logging

from data_harvest.core.config import ReviewConfig
from data_harvest.core.types import HarvestSample, ReviewStatus

logger = logging.getLogger(__name__)


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

            if s.label.confidence >= self.config.auto_approve_confidence:
                s.review_status = ReviewStatus.approved
                s.save_review()
                auto_approved += 1
            else:
                needs_review.append(s)

        # Sort by confidence ascending (most uncertain first)
        needs_review.sort(key=lambda s: s.label.confidence if s.label else 0.0)
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
