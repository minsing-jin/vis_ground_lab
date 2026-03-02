"""Dataset statistics for harvest sessions."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from data_harvest.core.types import HarvestSample, ReviewStatus, ActionType

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics summary for a harvest dataset."""

    total_samples: int = 0
    labeled_samples: int = 0
    unlabeled_samples: int = 0
    approved: int = 0
    edited: int = 0
    rejected: int = 0
    pending_review: int = 0
    action_counts: dict[str, int] | None = None
    avg_confidence: float = 0.0
    transition_count: int = 0

    def to_report(self) -> str:
        lines = [
            "=== Harvest Dataset Stats ===",
            f"Total samples:     {self.total_samples}",
            f"  Labeled:         {self.labeled_samples}",
            f"  Unlabeled:       {self.unlabeled_samples}",
            "",
            "Review status:",
            f"  Approved:        {self.approved}",
            f"  Edited:          {self.edited}",
            f"  Rejected:        {self.rejected}",
            f"  Pending:         {self.pending_review}",
            "",
            f"Avg confidence:    {self.avg_confidence:.3f}",
            f"Transitions:       {self.transition_count}",
        ]
        if self.action_counts:
            lines.append("")
            lines.append("Action distribution:")
            for action, count in sorted(self.action_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {action:12s}  {count}")
        return "\n".join(lines)


def compute_stats(samples: list[HarvestSample]) -> DatasetStats:
    """Compute statistics over a list of HarvestSamples."""
    action_counter: Counter[str] = Counter()
    total_conf = 0.0
    conf_count = 0
    transition_count = 0
    labeled = 0
    unlabeled = 0
    approved = 0
    edited = 0
    rejected = 0
    pending = 0

    for s in samples:
        if s.event is not None:
            action_counter[s.event.action.value] += 1

        if s.label is not None:
            labeled += 1
            total_conf += s.label.confidence
            conf_count += 1
            if s.label.transition_detected:
                transition_count += 1
        else:
            unlabeled += 1

        if s.review_status == ReviewStatus.approved:
            approved += 1
        elif s.review_status == ReviewStatus.edited:
            edited += 1
        elif s.review_status == ReviewStatus.rejected:
            rejected += 1
        else:
            pending += 1

    return DatasetStats(
        total_samples=len(samples),
        labeled_samples=labeled,
        unlabeled_samples=unlabeled,
        approved=approved,
        edited=edited,
        rejected=rejected,
        pending_review=pending,
        action_counts=dict(action_counter),
        avg_confidence=total_conf / conf_count if conf_count > 0 else 0.0,
        transition_count=transition_count,
    )
