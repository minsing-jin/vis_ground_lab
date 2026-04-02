"""Tests for data_harvest.cli filter decision persistence."""

from __future__ import annotations

from pathlib import Path

from data_harvest.cli import _persist_filter_decisions
from data_harvest.core.types import HarvestSample, ReviewStatus


def _sample(tmp_path: Path, sid: str, status: ReviewStatus = ReviewStatus.pending) -> HarvestSample:
    sample_dir = tmp_path / sid
    sample_dir.mkdir(parents=True)
    sample = HarvestSample(sample_id=sid, sample_dir=sample_dir, review_status=status)
    return sample


def test_persist_filter_decisions_rejects_dropped_pending_samples(tmp_path: Path):
    s1 = _sample(tmp_path, "sample_000001", ReviewStatus.pending)
    s2 = _sample(tmp_path, "sample_000002", ReviewStatus.pending)
    s3 = _sample(tmp_path, "sample_000003", ReviewStatus.pending)

    auto_rejected, skipped = _persist_filter_decisions(
        all_samples=[s1, s2, s3],
        kept_samples=[s1],
    )

    assert auto_rejected == 2
    assert skipped == 0
    assert s1.review_status == ReviewStatus.pending
    assert s2.review_status == ReviewStatus.rejected
    assert s3.review_status == ReviewStatus.rejected
    assert s2.review_corrections == {"auto_filter_rejected": True}


def test_persist_filter_decisions_does_not_override_non_pending_samples(tmp_path: Path):
    kept = _sample(tmp_path, "sample_000001", ReviewStatus.pending)
    approved = _sample(tmp_path, "sample_000002", ReviewStatus.approved)
    edited = _sample(tmp_path, "sample_000003", ReviewStatus.edited)

    auto_rejected, skipped = _persist_filter_decisions(
        all_samples=[kept, approved, edited],
        kept_samples=[kept],
    )

    assert auto_rejected == 0
    assert skipped == 2
    assert approved.review_status == ReviewStatus.approved
    assert edited.review_status == ReviewStatus.edited
