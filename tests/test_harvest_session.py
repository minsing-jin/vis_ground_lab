"""Tests for data_harvest.core.session."""

from __future__ import annotations

from pathlib import Path

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionEvent, ActionType, LabelResult


class TestHarvestSession:
    def test_setup_creates_dirs(self, tmp_path: Path):
        cfg = HarvestConfig(workdir=str(tmp_path / "sess"))
        session = HarvestSession(cfg)
        session.setup()
        assert session.samples_dir.exists()

    def test_next_sample_id(self, tmp_path: Path):
        cfg = HarvestConfig(workdir=str(tmp_path / "sess"))
        session = HarvestSession(cfg)
        session.setup()

        assert session.next_sample_id() == "sample_000001"
        s1 = session.create_sample()
        assert s1.sample_id == "sample_000001"
        assert session.next_sample_id() == "sample_000002"

    def test_iter_samples(self, tmp_path: Path):
        cfg = HarvestConfig(workdir=str(tmp_path / "sess"))
        session = HarvestSession(cfg)
        session.setup()

        # Create and save a sample with event
        s = session.create_sample()
        s.event = ActionEvent(timestamp_ms=100.0, action=ActionType.click, x=50.0, y=60.0)
        s.save_event()

        samples = session.iter_samples()
        assert len(samples) == 1
        assert samples[0].event is not None
        assert samples[0].event.action == ActionType.click

    def test_unlabeled_vs_labeled(self, tmp_path: Path):
        cfg = HarvestConfig(workdir=str(tmp_path / "sess"))
        session = HarvestSession(cfg)
        session.setup()

        # Unlabeled sample
        s1 = session.create_sample()
        s1.event = ActionEvent(timestamp_ms=100.0, action=ActionType.click, x=50.0, y=60.0)
        s1.save_event()

        # Labeled sample
        s2 = session.create_sample()
        s2.event = ActionEvent(timestamp_ms=200.0, action=ActionType.click, x=70.0, y=80.0)
        s2.save_event()
        s2.label = LabelResult(bbox_x_min=60, bbox_y_min=70, bbox_x_max=80, bbox_y_max=90, confidence=0.5)
        s2.save_label()

        assert len(session.unlabeled_samples()) == 1
        assert len(session.labeled_samples()) == 1
        assert session.sample_count == 2
