"""E2E test: synthetic data → label → filter → export full pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionEvent, ActionType
from data_harvest.labeler.fusion import AutoLabeler
from data_harvest.filter.pipeline import FilterPipeline
from data_harvest.export.router_exporter import export_router_full
from data_harvest.export.stats import compute_stats
from data_harvest.profiles.registry import discover_profiles, get_profile


def _create_synthetic_session(tmp_path: Path, n_samples: int = 5) -> HarvestConfig:
    """Create a synthetic session with pre/post frames and events."""
    cfg = HarvestConfig(workdir=str(tmp_path / "session"), game_profile="civ6")
    session = HarvestSession(cfg)
    session.setup()

    for i in range(n_samples):
        sample = session.create_sample()

        # Create pre frame (dark bg + white rectangle as UI element)
        pre = np.zeros((200, 300, 3), dtype=np.uint8)
        # Place a "button" at varying positions
        bx = 220 + i * 5
        by = 150
        pre[by : by + 30, bx : bx + 60] = 255

        # Create post frame (button changed color after click)
        post = pre.copy()
        post[by : by + 30, bx : bx + 60] = [0, 128, 255]  # Changed to orange

        cv2.imwrite(str(sample.pre_frame_path), pre)
        cv2.imwrite(str(sample.post_frame_path), post)

        # Click in the center of the button
        cx = float(bx + 30)
        cy = float(by + 15)
        sample.event = ActionEvent(
            timestamp_ms=1000.0 * (i + 1),
            action=ActionType.click,
            x=cx,
            y=cy,
            button="left",
        )
        sample.save_event()

    return cfg


class TestE2EPipeline:
    def test_label_filter_export(self, tmp_path: Path):
        # 1. Create synthetic session
        cfg = _create_synthetic_session(tmp_path, n_samples=5)
        session = HarvestSession(cfg)

        # Verify samples created
        unlabeled = session.unlabeled_samples()
        assert len(unlabeled) == 5

        # 2. Auto-label (disable VLM/OCR to avoid heavy dependencies)
        cfg.labeler.vlm.enabled = False
        cfg.labeler.use_ocr = False
        discover_profiles()
        labeler = AutoLabeler(cfg.labeler, profile=get_profile("civ6"))

        for sample in unlabeled:
            result = labeler.label_sample(sample)
            if result is not None:
                sample.label = result
                sample.save_label()

        labeled = session.labeled_samples()
        assert len(labeled) >= 3  # At least some should be labeled

        # 3. Filter pipeline
        pipeline = FilterPipeline(cfg.filter)
        # Patch dedup to avoid vis_ground_lab dependency
        with patch("data_harvest.filter.pipeline.deduplicate_samples", return_value=(labeled, [])):
            filter_result = pipeline.run(labeled)
        assert filter_result.total_kept > 0

        # 4. Export to routing dataset
        out_dir = tmp_path / "router_full"
        export_router_full(filter_result.kept, out_dir)
        labels_csv = out_dir / "labels.csv"
        assert labels_csv.exists()

        lines = labels_csv.read_text().strip().splitlines()
        assert len(lines) > 1
        header = lines[0].split(",")
        assert "primitive_id" in header
        assert "situation_id" in header

        # 5. Stats
        stats = compute_stats(session.iter_samples())
        assert stats.total_samples == 5
        assert stats.labeled_samples > 0

        report = stats.to_report()
        assert "Total samples" in report
