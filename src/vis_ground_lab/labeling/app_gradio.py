"""Lightweight Gradio labeling assistant for candidate UI boxes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vis_ground_lab.data import coco_bbox_to_xyxy, empty_coco, load_coco, register_categories, save_coco


def _annotations_for_image(coco: dict[str, Any], image_id: int) -> list[dict[str, Any]]:
    return [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) == int(image_id)]


def _draw_preview(image_path: Path, boxes: list[dict[str, Any]], categories: dict[int, str]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in boxes:
        x1, y1, x2, y2 = coco_bbox_to_xyxy(ann["bbox"])
        name = categories.get(int(ann["category_id"]), str(ann["category_id"]))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), name, fill="yellow")
    return image


def launch_labeling_app(
    image_dir: str,
    candidate_coco_path: str,
    out_coco_path: str,
    class_names: list[str] | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("gradio is required. Install with: pip install gradio") from exc

    image_dir_path = Path(image_dir)
    coco = load_coco(candidate_coco_path) if Path(candidate_coco_path).exists() else empty_coco()

    if class_names:
        if not coco.get("categories"):
            register_categories(coco, class_names)

    categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    name_to_id = {v: k for k, v in categories.items()}

    images = coco.get("images", [])
    if not images:
        for i, path in enumerate(sorted(image_dir_path.glob("*.png")), start=1):
            with Image.open(path) as img:
                w, h = img.size
            images.append({"id": i, "file_name": path.name, "width": w, "height": h})
        coco["images"] = images

    image_lookup = {str(i["file_name"]): i for i in images}
    ordered_files = list(image_lookup.keys())
    if not ordered_files:
        raise RuntimeError("No images found for labeling.")

    def load_image_boxes(file_name: str) -> tuple[Image.Image, str]:
        img = image_lookup[file_name]
        anns = _annotations_for_image(coco, int(img["id"]))
        preview = _draw_preview(image_dir_path / file_name, anns, categories)

        editable = []
        for ann in anns:
            x1, y1, x2, y2 = coco_bbox_to_xyxy(ann["bbox"])
            editable.append(
                {
                    "id": int(ann["id"]),
                    "class_name": categories.get(int(ann["category_id"]), "candidate"),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )
        return preview, json.dumps(editable, ensure_ascii=False, indent=2)

    def save_boxes(file_name: str, edited_json: str) -> tuple[Image.Image, str]:
        img = image_lookup[file_name]
        image_id = int(img["id"])

        try:
            entries = json.loads(edited_json) if edited_json.strip() else []
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        coco["annotations"] = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) != image_id]
        next_id = 1 + max([int(ann["id"]) for ann in coco.get("annotations", [])], default=0)

        for entry in entries:
            class_name = str(entry.get("class_name", "candidate"))
            if class_name not in name_to_id:
                category_id = len(categories) + 1
                categories[category_id] = class_name
                name_to_id[class_name] = category_id
                coco.setdefault("categories", []).append({"id": category_id, "name": class_name})
            else:
                category_id = name_to_id[class_name]

            x1 = float(entry["x1"])
            y1 = float(entry["y1"])
            x2 = float(entry["x2"])
            y2 = float(entry["y2"])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            coco["annotations"].append(
                {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            next_id += 1

        save_coco(coco, out_coco_path)
        anns = _annotations_for_image(coco, image_id)
        preview = _draw_preview(image_dir_path / file_name, anns, categories)
        return preview, f"Saved {len(anns)} annotations to {out_coco_path}"

    init_preview, init_json = load_image_boxes(ordered_files[0])
    init_status = f"Output COCO: {out_coco_path}"

    with gr.Blocks(title="vis_ground_lab label assistant") as demo:
        gr.Markdown("## Label Assistant\nEdit JSON boxes, assign class names, and save COCO ground truth.")
        file_dropdown = gr.Dropdown(choices=ordered_files, value=ordered_files[0], label="Image")
        preview = gr.Image(label="Preview", type="pil", value=init_preview)
        box_json = gr.Code(label="Editable boxes JSON", language="json", value=init_json)
        load_btn = gr.Button("Load")
        save_btn = gr.Button("Save")
        status = gr.Textbox(label="Status", value=init_status)

        load_btn.click(fn=load_image_boxes, inputs=[file_dropdown], outputs=[preview, box_json])
        file_dropdown.change(fn=load_image_boxes, inputs=[file_dropdown], outputs=[preview, box_json])
        save_btn.click(fn=save_boxes, inputs=[file_dropdown, box_json], outputs=[preview, status])

    demo.launch(server_name=server_name, server_port=server_port)
