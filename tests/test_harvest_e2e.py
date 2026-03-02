"""E2E test: synthetic data → label → filter → export full pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionEvent, ActionType, ReviewStatus
from data_harvest.labeler.fusion import AutoLabeler
from data_harvest.filter.pipeline import FilterPipeline
from data_harvest.export.grounding_exporter import export_grounding
from data_harvest.export.stats import compute_stats


def _create_synthetic_session(tmp_path: Path, n_samples: int = 5) -> HarvestConfig:
    """Create a synthetic session with pre/post frames and events."""
    cfg = HarvestConfig(workdir=str(tmp_path / "session"))
    session = HarvestSession(cfg)
    session.setup()

    for i in range(n_samples):
        sample = session.create_sample()

        # Create pre frame (dark bg + white rectangle as UI element)
        pre = np.zeros((200, 300, 3), dtype=np.uint8)
        # Place a "button" at varying positions
        bx = 50 + i * 40
        by = 80
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

        # 2. Auto-label (skip OCR to avoid easyocr dependency)
        labeler = AutoLabeler(cfg.labeler)

        # Patch OCR to avoid dependency
        from unittest.mock import patch
        with patch("data_harvest.labeler.fusion.ocr_bboxes", return_value=[]):
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

        # 4. Export to grounding JSONL
        out_path = tmp_path / "grounding.jsonl"
        export_grounding(filter_result.kept, out_path, normalizing_range=1000)
        assert out_path.exists()

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            rec = json.loads(line)
            assert "image_path" in rec
            assert "bbox" in rec
            assert "action" in rec

        # 5. Stats
        stats = compute_stats(session.iter_samples())
        assert stats.total_samples == 5
        assert stats.labeled_samples > 0

        report = stats.to_report()
        assert "Total samples" in report
