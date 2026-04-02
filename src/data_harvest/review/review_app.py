"""Gradio review UI for routing-first harvest labels with ROI correction."""

from __future__ import annotations

import base64
import html as html_lib
import logging
import mimetypes

import cv2
import gradio as gr
import numpy as np

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import HarvestSample, LabelResult
from data_harvest.profiles.base_profile import GameProfile
from data_harvest.review.queue import HarvestReviewQueue

logger = logging.getLogger(__name__)

_CATEGORY_LABELS = {
    "screen_type": "화면 종류",
    "situation_id": "상황",
    "primitive_id": "Primitive",
    "router_roi": "ROI",
}

_SCREEN_TYPE_LABELS = {
    "main_map": "메인 맵",
    "popup": "팝업",
    "tech_tree": "기술 트리",
    "civic_tree": "사회 제도 트리",
    "city_view": "도시 화면",
    "diplomacy": "외교 화면",
    "city_state_screen": "도시국가 화면",
    "government": "정부/정책 화면",
    "religion_screen": "종교 화면",
    "governor_screen": "총독 화면",
    "world_congress": "세계의회 화면",
    "era_screen": "시대 전략 화면",
    "deal_screen": "거래 화면",
    "war_screen": "전쟁 선포 화면",
    "loading": "로딩 화면",
    "main_menu": "메인 메뉴",
}

_PRIMITIVE_LABELS = {
    "religion_primitive": "종교 선택",
    "governor_primitive": "총독 관리",
    "voting_primitive": "세계의회 투표",
    "era_primitive": "시대 전략",
    "unit_ops_primitive": "유닛 일반 조작",
    "research_select_primitive": "기술 연구 선택",
    "city_production_primitive": "도시 생산 선택",
    "culture_decision_primitive": "사회 제도 선택",
    "diplomatic_primitive": "외교/사절",
    "combat_primitive": "전투 판단",
    "policy_primitive": "정책/정부",
    "popup_primitive": "일반 팝업/진입 버튼",
    "war_primitive": "전쟁 선포",
    "deal_primitive": "거래",
}

_ROI_LABELS = {
    "bottom_right": "우하단",
    "popup_center": "중앙 팝업",
    "unit_panel": "유닛 패널",
    "main_map": "전체 화면",
    "top_bar": "상단 바",
    "left_panel": "왼쪽 패널",
}

_ROI_INPUT_IDS = {
    "x1": "harvest-roi-x1",
    "y1": "harvest-roi-y1",
    "x2": "harvest-roi-x2",
    "y2": "harvest-roi-y2",
}
_ROI_SYNC_BUTTON_ID = "harvest-roi-sync"

_REVIEW_CSS = """
.harvest-roi-editor {
    border: 1px solid var(--block-border-color);
    border-radius: 12px;
    padding: 12px;
    background: var(--background-fill-secondary);
}
.harvest-roi-toolbar {
    font-weight: 600;
    margin-bottom: 8px;
}
.harvest-roi-subtitle {
    font-size: 0.88rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 10px;
}
.harvest-roi-stage {
    position: relative;
    display: inline-block;
    max-width: 100%;
    max-height: 48vh;
    line-height: 0;
    cursor: crosshair;
    user-select: none;
    overflow: hidden;
}
.harvest-roi-stage img {
    display: block;
    max-width: 100%;
    max-height: 48vh;
    width: auto;
    border-radius: 10px;
    margin: 0 auto;
}
.harvest-roi-box {
    position: absolute;
    border: 2px solid #3ddb87;
    background: rgba(61, 219, 135, 0.14);
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.18) inset;
    pointer-events: none;
}
.harvest-roi-help {
    margin-top: 8px;
    font-size: 0.86rem;
    color: var(--body-text-color-subdued);
}
.harvest-roi-empty {
    padding: 24px;
    border: 1px dashed var(--block-border-color);
    border-radius: 12px;
    color: var(--body-text-color-subdued);
}
.harvest-meta-row {
    align-items: start;
}
.harvest-evidence-accordion {
    margin-top: 6px;
}
.harvest-action-bar {
    position: sticky;
    bottom: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--background-fill-primary) 92%, transparent);
    backdrop-filter: blur(8px);
    border-top: 1px solid var(--block-border-color);
    padding-top: 12px;
    padding-bottom: 4px;
}
"""

_REVIEW_HEAD = f"""
<script>
(() => {{
  if (window.__harvestRoiEditorLoaded) return;
  window.__harvestRoiEditorLoaded = true;

  const INPUT_IDS = {{
    x1: "{_ROI_INPUT_IDS['x1']}",
    y1: "{_ROI_INPUT_IDS['y1']}",
    x2: "{_ROI_INPUT_IDS['x2']}",
    y2: "{_ROI_INPUT_IDS['y2']}",
  }};
  const SYNC_BUTTON_ID = "{_ROI_SYNC_BUTTON_ID}";

  function clamp(value) {{
    const num = Number(value);
    if (!Number.isFinite(num)) return 0;
    return Math.max(0, Math.min(1, num));
  }}

  function findNumberInput(elemId) {{
    const root = document.getElementById(elemId);
    if (!root) return null;
    return root.querySelector("input");
  }}

  function setNumber(elemId, value) {{
    const input = findNumberInput(elemId);
    if (!input) return;
    const text = clamp(value).toFixed(4);
    if (input.value === text) return;
    input.value = text;
    input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    input.dispatchEvent(new Event("change", {{ bubbles: true }}));
  }}

  function renderBox(box, bbox) {{
    if (!box || !bbox || bbox.length !== 4) return;
    box.style.left = `${{bbox[0] * 100}}%`;
    box.style.top = `${{bbox[1] * 100}}%`;
    box.style.width = `${{Math.max(0.01, bbox[2] - bbox[0]) * 100}}%`;
    box.style.height = `${{Math.max(0.01, bbox[3] - bbox[1]) * 100}}%`;
  }}

  function parseBBox(container) {{
    const raw = (container.dataset.bbox || "0,0,1,1").split(",");
    const bbox = raw.map((part) => clamp(part));
    while (bbox.length < 4) bbox.push(0);
    if (bbox[2] <= bbox[0]) bbox[2] = Math.min(1, bbox[0] + 0.01);
    if (bbox[3] <= bbox[1]) bbox[3] = Math.min(1, bbox[1] + 0.01);
    return bbox;
  }}

  function writeBBox(container, bbox) {{
    container.dataset.bbox = bbox.map((value) => clamp(value).toFixed(4)).join(",");
    setNumber(INPUT_IDS.x1, bbox[0]);
    setNumber(INPUT_IDS.y1, bbox[1]);
    setNumber(INPUT_IDS.x2, bbox[2]);
    setNumber(INPUT_IDS.y2, bbox[3]);
    window.clearTimeout(window.__harvestRoiSyncTimer);
    window.__harvestRoiSyncTimer = window.setTimeout(() => {{
      const root = document.getElementById(SYNC_BUTTON_ID);
      const button = root ? root.querySelector("button") : null;
      if (button) button.click();
    }}, 60);
  }}

  function pointToNorm(img, evt) {{
    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = clamp((evt.clientX - rect.left) / rect.width);
    const y = clamp((evt.clientY - rect.top) / rect.height);
    return [x, y];
  }}

  window.initHarvestRoiEditor = function(containerId) {{
    const container = document.getElementById(containerId);
    if (!container) return;
    const stage = container.querySelector(".harvest-roi-stage");
    const img = stage ? stage.querySelector("img") : null;
    const box = container.querySelector(".harvest-roi-box");
    const help = container.querySelector(".harvest-roi-help");
    if (!stage || !img || !box) return;

    const current = parseBBox(container);
    renderBox(box, current);

    if (container.dataset.bound === "1") return;
    container.dataset.bound = "1";

    let dragging = false;
    let start = null;

    const updateHelp = (text) => {{
      if (help) help.textContent = text;
    }};

    const finishDrag = (evt) => {{
      if (!dragging || !start) return;
      const point = pointToNorm(img, evt) || start;
      const bbox = [
        Math.min(start[0], point[0]),
        Math.min(start[1], point[1]),
        Math.max(start[0], point[0]),
        Math.max(start[1], point[1]),
      ];
      if (bbox[2] <= bbox[0]) bbox[2] = Math.min(1, bbox[0] + 0.01);
      if (bbox[3] <= bbox[1]) bbox[3] = Math.min(1, bbox[1] + 0.01);
      renderBox(box, bbox);
      writeBBox(container, bbox);
      updateHelp("ROI 업데이트됨. Save Edit 또는 Update Preview로 확인하세요.");
      dragging = false;
      start = null;
      try {{
        stage.releasePointerCapture(evt.pointerId);
      }} catch (_err) {{
        // no-op
      }}
      window.removeEventListener("pointerup", finishDrag);
    }};

    stage.addEventListener("pointerdown", (evt) => {{
      if (evt.button !== 0) return;
      const point = pointToNorm(img, evt);
      if (!point) return;
      evt.preventDefault();
      start = point;
      dragging = true;
      renderBox(box, [point[0], point[1], point[0], point[1]]);
      updateHelp("드래그를 끝내면 ROI가 저장됩니다.");
      try {{
        stage.setPointerCapture(evt.pointerId);
      }} catch (_err) {{
        // no-op
      }}
      window.addEventListener("pointerup", finishDrag);
    }});

    stage.addEventListener("pointermove", (evt) => {{
      if (!dragging || !start) return;
      const point = pointToNorm(img, evt);
      if (!point) return;
      renderBox(box, [
        Math.min(start[0], point[0]),
        Math.min(start[1], point[1]),
        Math.max(start[0], point[0]),
        Math.max(start[1], point[1]),
      ]);
    }});
  }};
}})();
</script>
"""


def _effective_label(sample: HarvestSample) -> LabelResult | None:
    return sample.effective_label()


def _prettify_slug(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _friendly_value(kind: str, value: str, profile: GameProfile | None) -> str:
    if not value:
        return ""
    if kind == "screen_types":
        return _SCREEN_TYPE_LABELS.get(value, _prettify_slug(value))
    if kind == "primitives":
        return _PRIMITIVE_LABELS.get(value, _prettify_slug(value.replace("_primitive", "")))
    if kind == "situations":
        if profile is not None:
            desc = profile.situation_dict.get(value, {}).get("description")
            if desc:
                return str(desc)
        return _prettify_slug(value.replace("_visible", "").replace("_open", ""))
    if kind == "rois":
        return _ROI_LABELS.get(value, _prettify_slug(value))
    return value


def _choice_pairs(kind: str, values: list[str], profile: GameProfile | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        display = _friendly_value(kind, value, profile)
        pairs.append((display, value))
    return pairs


def _ordered_primitive_choices(profile: GameProfile | None) -> list[str]:
    if profile is None:
        return []
    router_enabled = list(profile.router_primitive_dict.keys())
    hidden = [primitive_id for primitive_id in profile.primitive_dict.keys() if primitive_id not in profile.router_primitive_dict]
    return router_enabled + hidden


def _taxonomy_choices(profile: GameProfile | None) -> dict[str, list[str]]:
    if profile is None:
        return {"screen_types": [], "situations": [], "primitives": [], "semantics": [], "rois": []}
    return {
        "screen_types": sorted(profile.screen_types),
        "situations": sorted(profile.situation_dict.keys()),
        "primitives": _ordered_primitive_choices(profile),
        "semantics": sorted(profile.element_catalog.keys()),
        "rois": sorted(profile.roi_hints.keys()),
    }


def _candidate_key(kind: str) -> str:
    return {
        "situations": "situation_id",
        "primitives": "primitive_id",
        "rois": "roi_name",
    }.get(kind, kind)


def _ordered_situation_choices(profile: GameProfile | None, primitive_id: str | None = None) -> list[str]:
    if profile is None:
        return []
    situation_ids = list(profile.situation_dict.keys())
    if not primitive_id:
        return situation_ids

    preferred = [
        situation_id
        for situation_id, spec in profile.situation_dict.items()
        if primitive_id in [str(value) for value in spec.get("allowed_primitives", []) if value is not None]
    ]
    return preferred + [situation_id for situation_id in situation_ids if situation_id not in preferred]


def _ordered_roi_choices(profile: GameProfile | None, situation_id: str | None = None) -> list[str]:
    if profile is None:
        return []
    roi_ids = list(profile.roi_hints.keys())
    if not situation_id:
        return roi_ids
    preferred = [
        str(roi_id)
        for roi_id in profile.situation_dict.get(situation_id, {}).get("roi_priority", [])
        if roi_id is not None and str(roi_id) in profile.roi_hints
    ]
    return preferred + [roi_id for roi_id in roi_ids if roi_id not in preferred]


def _allowed_primitives(profile: GameProfile | None, situation_id: str | None) -> list[str]:
    if profile is None or not situation_id:
        return []
    return [
        str(value)
        for value in profile.situation_dict.get(situation_id.strip(), {}).get("allowed_primitives", [])
        if value is not None
    ]


def _primitive_suggestion(profile: GameProfile | None, situation_id: str | None) -> str:
    if profile is None or not situation_id:
        return ""
    situation = profile.situation_dict.get(situation_id.strip(), {})
    allowed = situation.get("allowed_primitives", [])
    if allowed:
        return str(allowed[0])
    return ""


def _with_current_value(choices: list[str], value: str | None) -> list[str]:
    merged = list(choices)
    if value:
        value = value.strip()
        if value and value not in merged:
            merged.insert(0, value)
    return merged


def _candidate_choices(
    sample: HarvestSample | None,
    label: LabelResult | None,
    *,
    kind: str,
    taxonomy_choices: list[str],
    current_value: str | None,
) -> list[str]:
    values: list[str] = []
    candidate_key = _candidate_key(kind)
    evidence = label.evidence if label is not None else {}
    route_candidates = evidence.get("route_candidates", []) if isinstance(evidence, dict) else []
    if isinstance(route_candidates, list):
        ranked = sorted(
            [candidate for candidate in route_candidates if isinstance(candidate, dict)],
            key=lambda candidate: float(candidate.get("confidence", 0.0)),
            reverse=True,
        )
        ranked = sorted(
            ranked,
            key=lambda candidate: (
                int(candidate.get("rank", 999)) if str(candidate.get("rank", "")).isdigit() else 999,
                -float(candidate.get("confidence", 0.0)),
            ),
        )
        for candidate in ranked[:3]:
            raw = candidate.get(candidate_key)
            if raw is not None:
                value = str(raw).strip()
                if value and value not in values:
                    values.append(value)

    if kind == "rois":
        route = label.route_label if label else None
        if route and route.roi_name:
            values = _with_current_value(values, route.roi_name)
    values = _with_current_value(values, current_value)
    for choice in taxonomy_choices:
        if choice not in values:
            values.append(choice)
    return values


def _clamp_norm_bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    bbox = [float(x1), float(y1), float(x2), float(y2)]
    bbox[0] = max(0.0, min(1.0, bbox[0]))
    bbox[1] = max(0.0, min(1.0, bbox[1]))
    bbox[2] = max(0.0, min(1.0, bbox[2]))
    bbox[3] = max(0.0, min(1.0, bbox[3]))
    if bbox[2] <= bbox[0]:
        bbox[2] = min(1.0, bbox[0] + 0.01)
    if bbox[3] <= bbox[1]:
        bbox[3] = min(1.0, bbox[1] + 0.01)
    return bbox


def _image_data_url(sample: HarvestSample | None) -> str:
    if sample is None or not sample.pre_frame_path.exists():
        return ""
    mime = mimetypes.guess_type(sample.pre_frame_path.name)[0] or "image/png"
    data = sample.pre_frame_path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _default_roi_bbox(
    profile: GameProfile | None,
    *,
    situation_id: str | None,
    roi_name: str | None,
    label: LabelResult | None,
) -> tuple[str, list[float]]:
    if label and label.route_label and label.route_label.roi_bbox_norm and len(label.route_label.roi_bbox_norm) == 4:
        current_roi = label.route_label.roi_name or roi_name or ""
        return current_roi, [float(value) for value in label.route_label.roi_bbox_norm]

    chosen_roi = (roi_name or "").strip()
    if not chosen_roi and profile is not None:
        chosen_roi = profile.situation_primary_roi(situation_id) or ""

    if profile is not None and chosen_roi and chosen_roi in profile.roi_hints:
        return chosen_roi, [float(value) for value in profile.roi_hints[chosen_roi]]
    return chosen_roi, [0.0, 0.0, 1.0, 1.0]


def _render_roi_editor_html(
    sample: HarvestSample | None,
    *,
    screen_type: str,
    situation_id: str,
    primitive_id: str,
    roi_name: str,
    roi_bbox: list[float] | None,
    profile: GameProfile | None,
) -> str:
    if sample is None:
        return '<div class="harvest-roi-empty">No sample loaded.</div>'

    data_url = _image_data_url(sample)
    if not data_url:
        return '<div class="harvest-roi-empty">Screenshot not found.</div>'

    bbox = _clamp_norm_bbox(*(roi_bbox or [0.0, 0.0, 1.0, 1.0]))
    bbox_text = ",".join(f"{value:.6f}" for value in bbox)
    screen_display = _friendly_value("screen_types", screen_type, profile) if screen_type else "화면 미지정"
    situation_display = _friendly_value("situations", situation_id, profile) if situation_id else "상황 미지정"
    primitive_display = _friendly_value("primitives", primitive_id, profile) if primitive_id else "Primitive 미지정"
    roi_display = _friendly_value("rois", roi_name, profile) if roi_name else "ROI 미지정"

    return f"""
<div class="harvest-roi-editor" id="harvest-roi-editor" data-bbox="{bbox_text}">
  <div class="harvest-roi-toolbar">ROI 드래그 편집</div>
  <div class="harvest-roi-subtitle">
    {html_lib.escape(screen_display)} / {html_lib.escape(situation_display)} / {html_lib.escape(primitive_display)} / {html_lib.escape(roi_display)}
  </div>
  <div class="harvest-roi-stage">
    <img src="{data_url}" alt="Review Screenshot" onload="window.initHarvestRoiEditor && window.initHarvestRoiEditor('harvest-roi-editor')" />
    <div class="harvest-roi-box"></div>
  </div>
  <div class="harvest-roi-help">이미지 위에서 사각형을 드래그하면 ROI 박스가 갱신되고, 아래 좌표 입력값도 자동으로 바뀝니다.</div>
</div>
"""


def _point_to_norm(sample: HarvestSample | None, point: object) -> tuple[float, float] | None:
    if sample is None or point is None:
        return None
    if not sample.pre_frame_path.exists():
        return None
    image = cv2.imread(str(sample.pre_frame_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return None
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        x, y = point[0], point[1]
        try:
            return (
                max(0.0, min(1.0, float(x) / float(w))),
                max(0.0, min(1.0, float(y) / float(h))),
            )
        except Exception:
            return None
    return None


def _format_evidence(label: LabelResult | None) -> str:
    if label is None or not isinstance(label.evidence, dict):
        return "No teacher evidence."

    evidence = label.evidence

    def _list_text(key: str) -> str:
        raw = evidence.get(key, [])
        if isinstance(raw, list):
            values = [str(value) for value in raw if value]
            return ", ".join(values) if values else "-"
        if raw:
            return str(raw)
        return "-"

    conflict_pair = evidence.get("conflict_pair")
    if isinstance(conflict_pair, list):
        conflict_text = " vs ".join(str(value) for value in conflict_pair if value) or "-"
    else:
        conflict_text = str(conflict_pair) if conflict_pair else "-"

    open_screen_detected = evidence.get("open_screen_detected")
    reasoning = evidence.get("reasoning") or "-"
    return (
        f"must-have: {_list_text('matched_must_have')}\n\n"
        f"strong cues: {_list_text('matched_strong_cues')}\n\n"
        f"hard negatives: {_list_text('triggered_hard_negatives')}\n\n"
        f"conflict: {conflict_text}\n\n"
        f"open screen: {open_screen_detected if open_screen_detected is not None else '-'}\n\n"
        f"reasoning: {reasoning}"
    )


def _evidence_sections(label: LabelResult | None) -> tuple[str, str, str, str, str]:
    if label is None or not isinstance(label.evidence, dict):
        return ("-", "-", "-", "-", "No teacher evidence.")

    evidence = label.evidence

    def _list_block(key: str) -> str:
        raw = evidence.get(key, [])
        if isinstance(raw, list):
            values = [str(value) for value in raw if value]
            if not values:
                return "-"
            return "\n".join(f"- {value}" for value in values)
        if raw:
            return f"- {raw}"
        return "-"

    conflict_pair = evidence.get("conflict_pair")
    if isinstance(conflict_pair, list):
        conflict_text = " vs ".join(str(value) for value in conflict_pair if value) or "-"
    else:
        conflict_text = str(conflict_pair) if conflict_pair else "-"

    open_screen_detected = evidence.get("open_screen_detected")
    reasoning = evidence.get("reasoning") or "-"
    summary = (
        f"- conflict: {conflict_text}\n"
        f"- open screen: {open_screen_detected if open_screen_detected is not None else '-'}\n"
        f"- reasoning: {reasoning}"
    )
    return (
        _list_block("matched_must_have"),
        _list_block("matched_strong_cues"),
        _list_block("triggered_hard_negatives"),
        f"- {conflict_text}" if conflict_text != "-" else "-",
        summary,
    )


def _current_labels_markdown(
    screen_type: str,
    situation_id: str,
    primitive_id: str,
    roi_name: str,
    profile: GameProfile | None,
) -> str:
    screen_type_display = _friendly_value("screen_types", screen_type, profile) if screen_type else "-"
    situation_display = _friendly_value("situations", situation_id, profile) if situation_id else "-"
    primitive_display = _friendly_value("primitives", primitive_id, profile) if primitive_id else "-"
    roi_display = _friendly_value("rois", roi_name, profile) if roi_name else "-"
    return (
        f"**Current Labels**\n\n"
        f"`screen_type`: {screen_type_display} (`{screen_type or '-'}`)\n\n"
        f"`situation_id`: {situation_display} (`{situation_id or '-'}`)\n\n"
        f"`primitive_id`: {primitive_display} (`{primitive_id or '-'}`)\n\n"
        f"`router_roi`: {roi_display} (`{roi_name or '-'}`)"
    )


def _render_preview(
    sample: HarvestSample,
    situation_id: str,
    primitive_id: str,
    *,
    roi_name: str = "",
    roi_bbox: list[float] | None = None,
    anchor_point: tuple[float, float] | None = None,
) -> np.ndarray | None:
    if not sample.pre_frame_path.exists():
        return None
    image = cv2.imread(str(sample.pre_frame_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if roi_bbox and len(roi_bbox) == 4:
        bx = _clamp_norm_bbox(*roi_bbox)
        px1 = max(0, min(w - 1, int(round(bx[0] * w))))
        py1 = max(0, min(h - 1, int(round(bx[1] * h))))
        px2 = max(px1 + 1, min(w, int(round(bx[2] * w))))
        py2 = max(py1 + 1, min(h, int(round(bx[3] * h))))
        cv2.rectangle(image, (px1, py1), (px2, py2), (60, 220, 120), 2)
        if roi_name:
            cv2.putText(
                image,
                roi_name[:48],
                (px1, max(20, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 220, 120),
                2,
                cv2.LINE_AA,
            )
    if anchor_point is not None:
        ax = max(0, min(w - 1, int(round(anchor_point[0] * w))))
        ay = max(0, min(h - 1, int(round(anchor_point[1] * h))))
        cv2.circle(image, (ax, ay), 6, (0, 255, 255), -1)
    summary = f"situation={situation_id or 'unknown'}  primitive={primitive_id or 'unknown'}"
    cv2.putText(
        image,
        summary[:100],
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _render_roi_preview(
    sample: HarvestSample,
    roi_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray | None:
    if not sample.pre_frame_path.exists():
        return None
    image = cv2.imread(str(sample.pre_frame_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    bx = _clamp_norm_bbox(x1, y1, x2, y2)
    px1 = max(0, min(w - 1, int(round(bx[0] * w))))
    py1 = max(0, min(h - 1, int(round(bx[1] * h))))
    px2 = max(px1 + 1, min(w, int(round(bx[2] * w))))
    py2 = max(py1 + 1, min(h, int(round(bx[3] * h))))
    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        crop = image
    label = roi_name or "manual_roi"
    cv2.putText(
        crop,
        label[:64],
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def launch_review_app(config: HarvestConfig) -> None:
    from data_harvest.profiles.registry import discover_profiles, get_profile

    session = HarvestSession(config)
    queue = HarvestReviewQueue(config.review)
    queue.load(session.labeled_samples())

    profile = None
    if config.game_profile:
        discover_profiles()
        try:
            profile = get_profile(config.game_profile)
        except Exception:
            logger.warning("Failed to load review profile '%s'.", config.game_profile, exc_info=True)

    taxonomy = _taxonomy_choices(profile)
    current_sample = {"ref": queue.next_sample()}

    def _screen_update(choices: list[str], value: str | None):
        merged = _with_current_value(choices, value)
        return gr.update(choices=_choice_pairs("screen_types", merged, profile), value=value or None)

    def _radio_update(
        kind: str,
        current_value: str | None,
        label: LabelResult | None,
        sample: HarvestSample | None,
        *,
        situation_id: str | None = None,
        primitive_id: str | None = None,
    ):
        if kind == "primitives":
            preferred = _allowed_primitives(profile, situation_id)
            taxonomy_choices = preferred + [
                primitive for primitive in _ordered_primitive_choices(profile) if primitive not in preferred
            ]
        elif kind == "situations":
            taxonomy_choices = _ordered_situation_choices(profile, primitive_id)
        elif kind == "rois":
            taxonomy_choices = _ordered_roi_choices(profile, situation_id)
        else:
            taxonomy_choices = taxonomy[kind]
        choices = _candidate_choices(sample, label, kind=kind, taxonomy_choices=taxonomy_choices, current_value=current_value)
        return gr.update(choices=_choice_pairs(kind, choices, profile), value=(current_value or choices[0] if choices else None))

    def _option_update(category: str, screen_type_value: str, situation_value: str, primitive_value: str, roi_value: str):
        sample = current_sample["ref"]
        label = _effective_label(sample) if sample is not None else None
        if category == "screen_type":
            choices = _with_current_value(taxonomy["screen_types"], screen_type_value)
            return gr.update(label="선택지", choices=_choice_pairs("screen_types", choices, profile), value=(screen_type_value or choices[0] if choices else None))
        if category == "situation_id":
            choices = _candidate_choices(
                sample,
                label,
                kind="situations",
                taxonomy_choices=_ordered_situation_choices(profile, primitive_value),
                current_value=situation_value,
            )
            return gr.update(label="선택지", choices=_choice_pairs("situations", choices, profile), value=(situation_value or choices[0] if choices else None))
        if category == "primitive_id":
            choices = _candidate_choices(
                sample,
                label,
                kind="primitives",
                taxonomy_choices=_allowed_primitives(profile, situation_value) + [
                    primitive for primitive in _ordered_primitive_choices(profile) if primitive not in _allowed_primitives(profile, situation_value)
                ],
                current_value=primitive_value,
            )
            return gr.update(label="선택지", choices=_choice_pairs("primitives", choices, profile), value=(primitive_value or choices[0] if choices else None))
        choices = _candidate_choices(
            sample,
            label,
            kind="rois",
            taxonomy_choices=_ordered_roi_choices(profile, situation_value),
            current_value=roi_value,
        )
        return gr.update(label="선택지", choices=_choice_pairs("rois", choices, profile), value=(roi_value or choices[0] if choices else None))

    def _load_current():
        sample = current_sample["ref"]
        if sample is None:
            return (
                _render_roi_editor_html(None, screen_type="", situation_id="", primitive_id="", roi_name="", roi_bbox=[0.0, 0.0, 1.0, 1.0], profile=profile),
                None,
                "No more samples to review.",
                "Remaining: 0",
                "-",
                "-",
                "-",
                "-",
                "No teacher evidence.",
                _current_labels_markdown("", "", "", "", profile),
                gr.update(label="선택지", choices=_choice_pairs("primitives", _ordered_primitive_choices(profile), profile), value=None),
                _screen_update(taxonomy["screen_types"], None),
                gr.update(choices=_choice_pairs("situations", _ordered_situation_choices(profile), profile), value=None),
                gr.update(choices=_choice_pairs("primitives", _ordered_primitive_choices(profile), profile), value=None),
                gr.update(choices=_choice_pairs("rois", _ordered_roi_choices(profile), profile), value=None),
                0.0,
                0.0,
                1.0,
                1.0,
                "ROI edit: no active sample.",
            )

        label = _effective_label(sample)
        screen_type = label.page.screen_type if label and label.page else ""
        situation_id = label.page.situation_id if label and label.page else ""
        primitive_id = label.route_label.primitive_id if label and label.route_label else ""
        roi_name, roi_bbox = _default_roi_bbox(profile, situation_id=situation_id, roi_name=(label.route_label.roi_name if label and label.route_label else None), label=label)
        preview = _render_roi_editor_html(
            sample,
            screen_type=screen_type or "",
            situation_id=situation_id or "",
            primitive_id=primitive_id or "",
            roi_name=roi_name,
            roi_bbox=roi_bbox,
            profile=profile,
        )
        roi_preview = _render_roi_preview(sample, roi_name, *roi_bbox)

        evidence = label.evidence if label is not None else {}
        route_candidates = evidence.get("route_candidates", []) if isinstance(evidence, dict) else []
        candidate_summary = ""
        if isinstance(route_candidates, list) and route_candidates:
            parts: list[str] = []
            ranked_candidates = sorted(
                [item for item in route_candidates if isinstance(item, dict)],
                key=lambda item: (
                    int(item.get("rank", 999)) if str(item.get("rank", "")).isdigit() else 999,
                    -float(item.get("confidence", 0.0)),
                ),
            )[:3]
            for candidate in ranked_candidates:
                primitive = candidate.get("primitive_id") or "?"
                situation = candidate.get("situation_id") or "?"
                confidence = float(candidate.get("confidence", 0.0))
                rank = candidate.get("rank") or "?"
                parts.append(f"#{rank} {primitive}/{situation} ({confidence:.2f})")
            candidate_summary = "  |  candidates=" + ", ".join(parts)

        info = f"**{sample.sample_id}**"
        if sample.event:
            info += f"  |  action={sample.event.action.value}"
            if sample.event.x is not None and sample.event.y is not None:
                info += f"  xy=({sample.event.x:.0f}, {sample.event.y:.0f})"
            if sample.event.key:
                info += f"  key={sample.event.key}"
        if label:
            info += f"  |  conf={label.confidence:.3f}"
        info += candidate_summary
        must_have_md, strong_cues_md, hard_negatives_md, conflict_md, summary_md = _evidence_sections(label)

        return (
            preview,
            roi_preview,
            info,
            f"Remaining: {queue.pending_count}",
            must_have_md,
            strong_cues_md,
            hard_negatives_md,
            conflict_md,
            summary_md,
            _current_labels_markdown(screen_type, situation_id, primitive_id, roi_name, profile),
            _option_update("primitive_id", screen_type, situation_id, primitive_id, roi_name),
            _screen_update(taxonomy["screen_types"], screen_type),
            _radio_update("situations", situation_id, label, sample, primitive_id=primitive_id),
            _radio_update("primitives", primitive_id, label, sample, situation_id=situation_id),
            _radio_update("rois", roi_name, label, sample, situation_id=situation_id),
            roi_bbox[0],
            roi_bbox[1],
            roi_bbox[2],
            roi_bbox[3],
            "ROI edit: drag a rectangle on the screenshot to replace the ROI.",
        )

    def _preview_from_state(screen_type: str, situation_id: str, primitive_id: str, roi_name: str, x1: float, y1: float, x2: float, y2: float):
        sample = current_sample["ref"]
        if sample is None:
            return None, None
        bbox = _clamp_norm_bbox(x1, y1, x2, y2)
        return (
            _render_roi_editor_html(
                sample,
                screen_type=screen_type,
                situation_id=situation_id,
                primitive_id=primitive_id,
                roi_name=roi_name,
                roi_bbox=bbox,
                profile=profile,
            ),
            _render_roi_preview(sample, roi_name, x1, y1, x2, y2),
        )

    def _apply_label_option(
        category: str,
        option_value: str,
        screen_type_value: str,
        situation_value: str,
        primitive_value: str,
        roi_value: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ):
        screen_type = screen_type_value or ""
        situation_id = situation_value or ""
        primitive_id = primitive_value or ""
        roi_name = roi_value or ""
        bbox = _clamp_norm_bbox(x1, y1, x2, y2)

        value = (option_value or "").strip()
        if category == "screen_type":
            screen_type = value
        elif category == "situation_id":
            situation_id = value
            allowed = _allowed_primitives(profile, situation_id)
            if allowed and primitive_id not in allowed:
                primitive_id = allowed[0]
        elif category == "primitive_id":
            primitive_id = value
            ordered = _ordered_situation_choices(profile, primitive_id)
            if ordered and situation_id not in ordered:
                situation_id = ordered[0]
        elif category == "router_roi":
            roi_name = value
            roi_name, bbox = _default_roi_bbox(profile, situation_id=situation_id, roi_name=roi_name, label=None)

        preview, roi_preview = _preview_from_state(screen_type, situation_id, primitive_id, roi_name, *bbox)
        return (
            preview,
            roi_preview,
            _current_labels_markdown(screen_type, situation_id, primitive_id, roi_name, profile),
            _option_update(category, screen_type, situation_id, primitive_id, roi_name),
            screen_type,
            gr.update(choices=_choice_pairs("situations", _ordered_situation_choices(profile, primitive_id), profile), value=situation_id or None),
            gr.update(choices=_choice_pairs("primitives", _ordered_primitive_choices(profile), profile), value=primitive_id or None),
            gr.update(choices=_choice_pairs("rois", _ordered_roi_choices(profile, situation_id), profile), value=roi_name or None),
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

    def _on_situation_change(situation_id: str, primitive_id: str):
        suggested = primitive_id.strip() or _primitive_suggestion(profile, situation_id)
        label = _effective_label(current_sample["ref"]) if current_sample["ref"] is not None else None
        sample = current_sample["ref"]
        chosen_roi, bbox = _default_roi_bbox(profile, situation_id=situation_id, roi_name=None, label=None)
        return (
            _radio_update("primitives", suggested, label, sample, situation_id=situation_id),
            _radio_update("rois", chosen_roi, label, sample, situation_id=situation_id),
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

    def _on_primitive_change(primitive_id: str, situation_id: str):
        label = _effective_label(current_sample["ref"]) if current_sample["ref"] is not None else None
        sample = current_sample["ref"]
        preferred_situation = situation_id.strip()
        ordered = _ordered_situation_choices(profile, primitive_id)
        if ordered and preferred_situation not in ordered:
            preferred_situation = ordered[0]
        chosen_roi, bbox = _default_roi_bbox(profile, situation_id=preferred_situation, roi_name=None, label=None)
        return (
            _radio_update("situations", preferred_situation, label, sample, primitive_id=primitive_id),
            _radio_update("rois", chosen_roi, label, sample, situation_id=preferred_situation),
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

    def _on_roi_change(roi_name: str):
        label = _effective_label(current_sample["ref"]) if current_sample["ref"] is not None else None
        chosen_roi, bbox = _default_roi_bbox(profile, situation_id=None, roi_name=roi_name, label=label)
        return gr.update(value=chosen_roi or None), bbox[0], bbox[1], bbox[2], bbox[3]

    def _save_edit(screen_type: str, situation_id: str, primitive_id: str, roi_name: str, x1: float, y1: float, x2: float, y2: float):
        sample = current_sample["ref"]
        if sample is not None:
            bbox = _clamp_norm_bbox(x1, y1, x2, y2)
            queue.edit(
                sample,
                {
                    "page": {
                        "screen_type": screen_type.strip() or None,
                        "situation_id": situation_id.strip() or None,
                        "state_flags": [],
                    },
                    "route_label": {
                        "primitive_id": primitive_id.strip() or None,
                        "roi_name": roi_name.strip() or None,
                        "roi_bbox_norm": bbox,
                        "trigger_modality": "keyboard" if (sample.event and sample.event.action.value in ("press", "type")) else "mouse",
                        "trigger_action_type": sample.event.action.value if sample.event else None,
                        "trigger_mouse_button": sample.event.button if sample.event else None,
                        "trigger_key": sample.event.key if sample.event else None,
                    },
                },
            )
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    def _approve():
        sample = current_sample["ref"]
        if sample is not None:
            queue.approve(sample)
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    def _reject():
        sample = current_sample["ref"]
        if sample is not None:
            queue.reject(sample)
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    with gr.Blocks(title="Data Harvest Review", css=_REVIEW_CSS, head=_REVIEW_HEAD) as app:
        gr.Markdown("# Data Harvest Review")
        gr.Markdown("Routing-first review. Gemini candidate order is shown first. Primitive, situation, and ROI are corrected here.")

        with gr.Row(elem_classes="harvest-meta-row"):
            pre_img = gr.HTML(label="Screenshot")
            roi_img = gr.Image(label="Router ROI Preview", type="numpy", height=320)

        info_md = gr.Markdown("")
        remaining_md = gr.Markdown("Loading...")
        roi_status_md = gr.Markdown("ROI edit: drag a rectangle on the screenshot to replace the ROI.")

        with gr.Row():
            with gr.Accordion("Must Have", open=False, elem_classes="harvest-evidence-accordion"):
                must_have_md = gr.Markdown("-")
            with gr.Accordion("Strong Cues", open=False, elem_classes="harvest-evidence-accordion"):
                strong_cues_md = gr.Markdown("-")
            with gr.Accordion("Hard Negatives", open=False, elem_classes="harvest-evidence-accordion"):
                hard_negatives_md = gr.Markdown("-")

        with gr.Row():
            with gr.Accordion("Conflict", open=False, elem_classes="harvest-evidence-accordion"):
                conflict_md = gr.Markdown("-")
            with gr.Accordion("Teacher Summary", open=False, elem_classes="harvest-evidence-accordion"):
                summary_md = gr.Markdown("-")

        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=320):
                    label_category = gr.Radio(
                        label="수정할 라벨",
                        choices=[
                            (_CATEGORY_LABELS["screen_type"], "screen_type"),
                            (_CATEGORY_LABELS["situation_id"], "situation_id"),
                            (_CATEGORY_LABELS["primitive_id"], "primitive_id"),
                            (_CATEGORY_LABELS["router_roi"], "router_roi"),
                        ],
                        value="primitive_id",
                    )
                with gr.Column(scale=1, min_width=320):
                    label_option = gr.Radio(
                        label="선택지",
                        choices=_choice_pairs("primitives", taxonomy["primitives"], profile),
                    )
            current_labels_md = gr.Markdown("")

        screen_type = gr.Dropdown(label="screen_type", choices=_choice_pairs("screen_types", taxonomy["screen_types"], profile), allow_custom_value=True, visible=False)
        situation_id = gr.Radio(label="situation_id", choices=_choice_pairs("situations", taxonomy["situations"], profile), visible=False)
        primitive_id = gr.Radio(label="primitive_id", choices=_choice_pairs("primitives", taxonomy["primitives"], profile), visible=False)
        roi_name = gr.Radio(label="router_roi", choices=_choice_pairs("rois", taxonomy["rois"], profile), visible=False)

        with gr.Row():
            roi_x1 = gr.Number(label="roi x1 (norm)", value=0.0, elem_id=_ROI_INPUT_IDS["x1"])
            roi_y1 = gr.Number(label="roi y1 (norm)", value=0.0, elem_id=_ROI_INPUT_IDS["y1"])
            roi_x2 = gr.Number(label="roi x2 (norm)", value=1.0, elem_id=_ROI_INPUT_IDS["x2"])
            roi_y2 = gr.Number(label="roi y2 (norm)", value=1.0, elem_id=_ROI_INPUT_IDS["y2"])

        with gr.Row(elem_classes="harvest-action-bar"):
            approve_btn = gr.Button("Approve (A)", variant="primary")
            save_btn = gr.Button("Save Edit + Next (E)")
            reject_btn = gr.Button("Reject (R)", variant="stop")
            preview_btn = gr.Button("Update Preview")
            roi_sync_btn = gr.Button("Sync ROI", visible=False, elem_id=_ROI_SYNC_BUTTON_ID)

        outputs = [
            pre_img,
            roi_img,
            info_md,
            remaining_md,
            must_have_md,
            strong_cues_md,
            hard_negatives_md,
            conflict_md,
            summary_md,
            current_labels_md,
            label_option,
            screen_type,
            situation_id,
            primitive_id,
            roi_name,
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2,
            roi_status_md,
        ]
        app.load(fn=_load_current, outputs=outputs)
        approve_btn.click(fn=_approve, outputs=outputs)
        reject_btn.click(fn=_reject, outputs=outputs)
        save_btn.click(fn=_save_edit, inputs=[screen_type, situation_id, primitive_id, roi_name, roi_x1, roi_y1, roi_x2, roi_y2], outputs=outputs)
        preview_btn.click(
            fn=_preview_from_state,
            inputs=[screen_type, situation_id, primitive_id, roi_name, roi_x1, roi_y1, roi_x2, roi_y2],
            outputs=[pre_img, roi_img],
        )
        roi_sync_btn.click(
            fn=_preview_from_state,
            inputs=[screen_type, situation_id, primitive_id, roi_name, roi_x1, roi_y1, roi_x2, roi_y2],
            outputs=[pre_img, roi_img],
        )
        label_category.change(
            fn=_option_update,
            inputs=[label_category, screen_type, situation_id, primitive_id, roi_name],
            outputs=[label_option],
        )
        label_option.input(
            fn=_apply_label_option,
            inputs=[label_category, label_option, screen_type, situation_id, primitive_id, roi_name, roi_x1, roi_y1, roi_x2, roi_y2],
            outputs=[pre_img, roi_img, current_labels_md, label_option, screen_type, situation_id, primitive_id, roi_name, roi_x1, roi_y1, roi_x2, roi_y2],
        )

    app.launch(server_port=config.review.server_port, share=False)
