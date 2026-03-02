"""Gradio UI for human review: pre/post side-by-side, approve/edit/reject."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data_harvest.core.config import HarvestConfig
from data_harvest.core.session import HarvestSession
from data_harvest.core.types import ReviewStatus
from data_harvest.review.queue import HarvestReviewQueue

logger = logging.getLogger(__name__)


def launch_review_app(config: HarvestConfig) -> None:
    """Launch the Gradio review UI."""
    import gradio as gr

    session = HarvestSession(config)
    queue = HarvestReviewQueue(config.review)
    samples = session.labeled_samples()
    queue.load(samples)

    current_sample = {"ref": queue.next_sample()}

    def _load_current():
        s = current_sample["ref"]
        if s is None:
            return None, None, "No more samples to review.", "", ""

        pre_img = str(s.pre_frame_path) if s.pre_frame_path.exists() else None
        post_img = str(s.post_frame_path) if s.post_frame_path.exists() else None
        info = f"**{s.sample_id}**"
        if s.event:
            info += f"  |  action={s.event.action.value}"
            if s.event.x is not None:
                info += f"  xy=({s.event.x:.0f}, {s.event.y:.0f})"
        label_json = ""
        if s.label:
            info += f"  |  conf={s.label.confidence:.3f}"
            label_json = json.dumps(s.label.to_dict(), indent=2, ensure_ascii=False)

        remaining = f"Remaining: {queue.pending_count}"
        return pre_img, post_img, info, label_json, remaining

    def on_approve():
        s = current_sample["ref"]
        if s:
            queue.approve(s)
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    def on_reject():
        s = current_sample["ref"]
        if s:
            queue.reject(s)
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    def on_edit(bbox_text: str):
        s = current_sample["ref"]
        if s and bbox_text.strip():
            try:
                corrections = json.loads(bbox_text)
                queue.edit(s, corrections)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON for edit corrections.")
        current_sample["ref"] = queue.next_sample()
        return _load_current()

    with gr.Blocks(title="Data Harvest Review") as app:
        gr.Markdown("# Data Harvest Review")
        remaining_md = gr.Markdown("Loading...")

        with gr.Row():
            pre_img = gr.Image(label="Pre-frame", type="filepath")
            post_img = gr.Image(label="Post-frame", type="filepath")

        info_md = gr.Markdown("")
        label_box = gr.Textbox(label="Label JSON (edit bbox_xyxy for corrections)", lines=8)

        with gr.Row():
            approve_btn = gr.Button("Approve (A)", variant="primary")
            edit_btn = gr.Button("Edit (E)")
            reject_btn = gr.Button("Reject (R)", variant="stop")

        outputs = [pre_img, post_img, info_md, label_box, remaining_md]

        approve_btn.click(fn=on_approve, outputs=outputs)
        reject_btn.click(fn=on_reject, outputs=outputs)
        edit_btn.click(fn=on_edit, inputs=[label_box], outputs=outputs)

        app.load(fn=_load_current, outputs=outputs)

    app.launch(server_port=config.review.server_port, share=False)
