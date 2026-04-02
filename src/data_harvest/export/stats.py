"""Dataset statistics for routing-first harvest sessions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from data_harvest.core.types import HarvestSample, ReviewStatus


@dataclass
class DatasetStats:
    total_samples: int = 0
    labeled_samples: int = 0
    unlabeled_samples: int = 0
    approved: int = 0
    edited: int = 0
    rejected: int = 0
    pending_review: int = 0
    avg_confidence: float = 0.0
    missing_primitive_id: int = 0
    missing_situation_id: int = 0
    duplicate_non_representative: int = 0
    primitive_counts: dict[str, int] | None = None
    situation_counts: dict[str, int] | None = None

    def to_report(self) -> str:
        lines = [
            "=== Harvest Routing Stats ===",
            f"Total samples:           {self.total_samples}",
            f"  Labeled:               {self.labeled_samples}",
            f"  Unlabeled:             {self.unlabeled_samples}",
            "",
            "Review status:",
            f"  Approved:              {self.approved}",
            f"  Edited:                {self.edited}",
            f"  Rejected:              {self.rejected}",
            f"  Pending:               {self.pending_review}",
            "",
            f"Avg confidence:          {self.avg_confidence:.3f}",
            f"Missing primitive_id:    {self.missing_primitive_id}",
            f"Missing situation_id:    {self.missing_situation_id}",
            f"Duplicate non-repr:      {self.duplicate_non_representative}",
        ]
        if self.primitive_counts:
            lines.append("")
            lines.append("Primitive distribution:")
            for primitive_id, count in sorted(self.primitive_counts.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"  {primitive_id:24s} {count}")
        if self.situation_counts:
            lines.append("")
            lines.append("Situation distribution:")
            for situation_id, count in sorted(self.situation_counts.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"  {situation_id:24s} {count}")
        return "\n".join(lines)


def compute_stats(samples: list[HarvestSample]) -> DatasetStats:
    primitive_counts: Counter[str] = Counter()
    situation_counts: Counter[str] = Counter()
    total_conf = 0.0
    conf_count = 0
    labeled = 0
    unlabeled = 0
    approved = 0
    edited = 0
    rejected = 0
    pending = 0
    missing_primitive_id = 0
    missing_situation_id = 0
    duplicate_non_representative = 0

    for sample in samples:
        label = sample.effective_label() or sample.label
        if label is not None:
            labeled += 1
            total_conf += label.confidence
            conf_count += 1
            primitive_id = label.route_label.primitive_id if label.route_label else None
            situation_id = label.page.situation_id if label.page else None
            if primitive_id:
                primitive_counts[str(primitive_id)] += 1
            else:
                missing_primitive_id += 1
            if situation_id:
                situation_counts[str(situation_id)] += 1
            else:
                missing_situation_id += 1
        else:
            unlabeled += 1

        md = sample.metadata or {}
        filter_md = md.get("filter", {}) if isinstance(md, dict) else {}
        flags = set(filter_md.get("flags", [])) if isinstance(filter_md, dict) else set()
        if "duplicate_non_representative" in flags:
            duplicate_non_representative += 1

        if sample.review_status == ReviewStatus.approved:
            approved += 1
        elif sample.review_status == ReviewStatus.edited:
            edited += 1
        elif sample.review_status == ReviewStatus.rejected:
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
        avg_confidence=total_conf / conf_count if conf_count else 0.0,
        missing_primitive_id=missing_primitive_id,
        missing_situation_id=missing_situation_id,
        duplicate_non_representative=duplicate_non_representative,
        primitive_counts=dict(primitive_counts),
        situation_counts=dict(situation_counts),
    )
