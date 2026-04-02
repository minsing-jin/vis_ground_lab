"""Tests for label-auto provider selection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_harvest import cli
from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ActionEvent, ActionType, LabelResult
from data_harvest.llm.provider import RelabelCandidate, RelabelResult


def _make_unlabeled_session(tmp_path: Path) -> HarvestConfig:
    cfg = HarvestConfig(workdir=str(tmp_path / "session"), game_profile="civ6")
    session = HarvestSession(cfg)
    session.setup()

    sample = session.create_sample()
    frame = np.ones((80, 120, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(sample.pre_frame_path), frame)
    cv2.imwrite(str(sample.post_frame_path), frame)
    sample.event = ActionEvent(timestamp_ms=1.0, action=ActionType.click, x=40, y=30, button="left")
    sample.save_event()
    return cfg


def _make_unlabeled_samples(tmp_path: Path, count: int) -> HarvestConfig:
    cfg = HarvestConfig(workdir=str(tmp_path / "session"), game_profile="civ6")
    session = HarvestSession(cfg)
    session.setup()

    for idx in range(count):
        sample = session.create_sample()
        frame = np.ones((80, 120, 3), dtype=np.uint8) * (100 + idx)
        cv2.imwrite(str(sample.pre_frame_path), frame)
        cv2.imwrite(str(sample.post_frame_path), frame)
        sample.event = ActionEvent(
            timestamp_ms=float(idx + 1),
            action=ActionType.click,
            x=40 + idx,
            y=30 + idx,
            button="left",
        )
        sample.save_event()
    return cfg


class _FakeGeminiProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def relabel(self, sample_payload):
        self.calls.append(sample_payload["sample_id"])
        assert sample_payload["sample_id"].startswith("sample_")
        assert "ontology" in sample_payload
        return RelabelResult(
            chosen=RelabelCandidate(
                bbox_xyxy=[10, 12, 60, 52],
                semantic_text="Popup confirm button",
                semantic_id="btn_accept_popup",
                function_id="popup_primitive",
                primitive_id="popup_primitive",
                screen_type="popup",
                situation_id="generic_popup_or_entry_prompt_visible",
                roi_name="popup_center",
                action="click",
                confidence=0.88,
                source="gemini",
            ),
            candidates=[
                RelabelCandidate(
                    bbox_xyxy=[10, 12, 60, 52],
                    semantic_text="Popup confirm button",
                    semantic_id="btn_accept_popup",
                    function_id="popup_primitive",
                    primitive_id="popup_primitive",
                    roi_name="popup_center",
                    confidence=0.88,
                    source="gemini",
                ),
                RelabelCandidate(
                    bbox_xyxy=[65, 12, 110, 52],
                    semantic_text="Popup cancel button",
                    semantic_id="btn_cancel_popup",
                    function_id="policy_primitive",
                    primitive_id="policy_primitive",
                    roi_name="popup_center",
                    confidence=0.67,
                    source="gemini",
                ),
            ],
            evidence={
                "matched_must_have": ["generic popup visible"],
                "matched_strong_cues": ["bottom-right entry prompt"],
                "triggered_hard_negatives": [],
                "conflict_pair": ["popup_primitive", "policy_primitive"],
                "open_screen_detected": False,
                "reasoning": "Only the entry-style popup is visible.",
            },
        )


def test_label_auto_uses_gemini_provider(monkeypatch, tmp_path: Path):
    cfg = _make_unlabeled_session(tmp_path)
    cfg.labeler.provider = "gemini"
    cfg.labeler.provider_fallback_to_local = False

    monkeypatch.setattr(cli, "_load_config", lambda _path: cfg)
    monkeypatch.setattr(cli, "_build_relabel_provider", lambda _cfg: _FakeGeminiProvider())

    cli.label_auto(config="ignored")

    session = HarvestSession(cfg)
    labeled = session.labeled_samples()
    assert len(labeled) == 1
    sample = labeled[0]
    assert sample.label is not None
    assert sample.label.route_label is not None
    assert sample.label.route_label.primitive_id == "popup_primitive"
    assert sample.label.route_label.roi_name == "popup_center"
    assert sample.label.page is not None
    assert sample.label.page.situation_id == "generic_popup_or_entry_prompt_visible"
    assert sample.label.elements == []
    assert sample.metadata is not None
    assert sample.metadata["label_auto_provider"]["provider"] == "gemini"


def test_label_auto_falls_back_to_local(monkeypatch, tmp_path: Path):
    cfg = _make_unlabeled_session(tmp_path)
    cfg.labeler.provider = "gemini"
    cfg.labeler.provider_fallback_to_local = True

    monkeypatch.setattr(cli, "_load_config", lambda _path: cfg)
    monkeypatch.setattr(cli, "_build_relabel_provider", lambda _cfg: (_ for _ in ()).throw(RuntimeError("missing api key")))

    from data_harvest.labeler.fusion import AutoLabeler

    def _fake_label_sample(self, sample):  # noqa: ARG001
        return LabelResult(
            bbox_x_min=5,
            bbox_y_min=6,
            bbox_x_max=55,
            bbox_y_max=46,
            semantic_text="Choose Research button",
            semantic_id="btn_choose_research",
            function_id="popup_primitive",
            situation_id="research_prompt_visible",
            screen_type="main_map",
            confidence=0.5,
        )

    monkeypatch.setattr(AutoLabeler, "label_sample", _fake_label_sample)

    cli.label_auto(config="ignored")

    session = HarvestSession(cfg)
    labeled = session.labeled_samples()
    assert len(labeled) == 1
    sample = labeled[0]
    assert sample.metadata is not None
    assert sample.metadata["label_auto_provider"]["provider"] == "local_vlm"
    assert sample.metadata["label_auto_provider"]["fallback_from"] == "gemini"


def test_label_auto_deduplicates_before_provider(monkeypatch, tmp_path: Path):
    cfg = _make_unlabeled_samples(tmp_path, count=4)
    cfg.labeler.provider = "gemini"
    cfg.labeler.provider_fallback_to_local = False

    monkeypatch.setattr(cli, "_load_config", lambda _path: cfg)
    provider = _FakeGeminiProvider()
    monkeypatch.setattr(cli, "_build_relabel_provider", lambda _cfg: provider)

    session = HarvestSession(cfg)
    samples = session.unlabeled_samples()

    def _fake_group(_samples, *, hash_threshold):  # noqa: ARG001
        assert len(_samples) == 4
        ordered = ["cluster_1", "cluster_2"]
        grouped = {
            "cluster_1": [samples[0], samples[1], samples[2]],
            "cluster_2": [samples[3]],
        }
        cluster_ids = {
            samples[0].sample_id: "cluster_1",
            samples[1].sample_id: "cluster_1",
            samples[2].sample_id: "cluster_1",
            samples[3].sample_id: "cluster_2",
        }
        is_representative = {
            samples[0].sample_id: True,
            samples[1].sample_id: False,
            samples[2].sample_id: False,
            samples[3].sample_id: True,
        }
        return ordered, grouped, cluster_ids, is_representative

    monkeypatch.setattr(cli, "_group_samples_by_cluster", _fake_group)

    cli.label_auto(config="ignored")

    labeled = HarvestSession(cfg).labeled_samples()
    assert len(labeled) == 4
    assert provider.calls == [samples[0].sample_id, samples[3].sample_id]

    copied = {sample.sample_id: sample for sample in labeled if sample.sample_id in {samples[1].sample_id, samples[2].sample_id}}
    for sample in copied.values():
        assert sample.metadata is not None
        assert sample.metadata["label_auto_provider"]["status"] == "copied_from_representative"
        assert sample.metadata["label_auto_provider"]["source_sample_id"] == samples[0].sample_id
        assert sample.metadata["filter"]["cluster_id"] == "cluster_1"
        assert sample.metadata["filter"]["cluster_representative"] is False
        assert "duplicate_non_representative" in sample.metadata["filter"]["flags"]
        assert sample.label is not None
        assert sample.label.route_label is not None
        assert sample.label.route_label.primitive_id == "popup_primitive"


def test_label_auto_blocks_duplicate_cluster_when_representative_fails(monkeypatch, tmp_path: Path):
    cfg = _make_unlabeled_samples(tmp_path, count=2)
    cfg.labeler.provider = "gemini"
    cfg.labeler.provider_fallback_to_local = False

    monkeypatch.setattr(cli, "_load_config", lambda _path: cfg)

    class _FailingProvider:
        def relabel(self, sample_payload):  # noqa: ARG002
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_build_relabel_provider", lambda _cfg: _FailingProvider())

    session = HarvestSession(cfg)
    samples = session.unlabeled_samples()

    def _fake_group(_samples, *, hash_threshold):  # noqa: ARG001
        ordered = ["cluster_1"]
        grouped = {"cluster_1": [samples[0], samples[1]]}
        cluster_ids = {
            samples[0].sample_id: "cluster_1",
            samples[1].sample_id: "cluster_1",
        }
        is_representative = {
            samples[0].sample_id: True,
            samples[1].sample_id: False,
        }
        return ordered, grouped, cluster_ids, is_representative

    monkeypatch.setattr(cli, "_group_samples_by_cluster", _fake_group)

    cli.label_auto(config="ignored")

    reloaded = HarvestSession(cfg).iter_samples()
    assert len([sample for sample in reloaded if sample.label is not None]) == 0

    representative = next(sample for sample in reloaded if sample.sample_id == samples[0].sample_id)
    duplicate = next(sample for sample in reloaded if sample.sample_id == samples[1].sample_id)

    assert representative.metadata is not None
    assert representative.metadata["label_auto_provider"]["status"] == "failed"
    assert representative.metadata["filter"]["cluster_representative"] is True

    assert duplicate.metadata is not None
    assert duplicate.metadata["label_auto_provider"]["status"] == "blocked_by_representative_failure"
    assert duplicate.metadata["label_auto_provider"]["source_sample_id"] == representative.sample_id
    assert duplicate.metadata["filter"]["cluster_representative"] is False
    assert "duplicate_non_representative" in duplicate.metadata["filter"]["flags"]
