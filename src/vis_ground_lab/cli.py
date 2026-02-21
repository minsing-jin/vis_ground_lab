"""Command line interface for vis_ground_lab."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import typer
from PIL import Image

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.config.schema import ModelConfig
from vis_ground_lab.config.loader import load_train_config
from vis_ground_lab.data import (
    add_annotation_entry,
    add_image_entry,
    deduplicate_images,
    empty_coco,
    extract_frames,
    register_categories,
    save_coco,
)
from vis_ground_lab.data_manager import JSONLVisualGroundingDataset
from vis_ground_lab.evaluation import Evaluator
from vis_ground_lab.export import compute_dataset_hash, create_model_package
from vis_ground_lab.labeling import launch_labeling_app
from vis_ground_lab.models.factory import create_model_wrapper
from vis_ground_lab.models.florence2 import Florence2Wrapper
from vis_ground_lab.optimization import run_optimization
from vis_ground_lab.prelabel import create_prelabeler
from vis_ground_lab.training.trainer_engine import TrainerEngine

app = typer.Typer(help="Visual grounding and tool-specific detector toolkit")


def _to_pixel_bbox(bbox: BoundingBox, normalize_mode: str, width: int, height: int) -> BoundingBox:
    if normalize_mode == "none":
        return bbox
    if normalize_mode == "0-1":
        return BoundingBox(
            x_min=bbox.x_min * width,
            y_min=bbox.y_min * height,
            x_max=bbox.x_max * width,
            y_max=bbox.y_max * height,
        )
    if normalize_mode == "0-1000":
        return BoundingBox(
            x_min=(bbox.x_min / 1000.0) * width,
            y_min=(bbox.y_min / 1000.0) * height,
            x_max=(bbox.x_max / 1000.0) * width,
            y_max=(bbox.y_max / 1000.0) * height,
        )
    raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")


def _iter_images(image_dir: Path) -> list[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(image_dir.glob(pattern)))
    return files


def _sample_paths(paths: list[Path], max_samples: int | None) -> list[Path]:
    if max_samples is None or max_samples <= 0 or len(paths) <= max_samples:
        return paths
    random.seed(7)
    return sorted(random.sample(paths, max_samples))


@app.command()
def train(config: str = typer.Option(..., "--config", "-c", help="Path to training YAML config")) -> None:
    """Run training from a YAML config file."""
    cfg = load_train_config(config)
    task_name = cfg.task.name.lower()

    if task_name == "grounding":
        model = create_model_wrapper(cfg.model)
        model.load_model(adapter_path_or_repo=cfg.model.adapter_path_or_repo)

        train_dataset = JSONLVisualGroundingDataset(
            source=cfg.data.train_jsonl,
            image_root=cfg.data.image_root,
            normalize_mode=cfg.data.normalize_mode,
        )

        eval_dataset = None
        if cfg.data.eval_jsonl:
            eval_dataset = JSONLVisualGroundingDataset(
                source=cfg.data.eval_jsonl,
                image_root=cfg.data.image_root,
                normalize_mode=cfg.data.normalize_mode,
            )

        engine = TrainerEngine(model_wrapper=model, config=cfg.trainer)
        engine.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        return

    if task_name == "tool_button_detection":
        if cfg.model.backend.lower() != "yolo_ultralytics":
            raise typer.BadParameter("tool_button_detection currently requires model.backend=yolo_ultralytics")
        dataset_yaml = cfg.data.dataset_yaml or cfg.data.train_jsonl
        if not dataset_yaml:
            raise typer.BadParameter("For tool_button_detection, set data.dataset_yaml in config")

        wrapper = create_model_wrapper(cfg.model)
        metrics = wrapper.train(
            dataset=dataset_yaml,
            cfg={
                "epochs": cfg.trainer.epochs,
                "batch_size": cfg.trainer.batch_size,
                "learning_rate": cfg.trainer.learning_rate,
                "run_name": "tool_button_train",
            },
            workdir=cfg.trainer.checkpoint_dir,
        )
        typer.echo(metrics)
        return

    raise typer.BadParameter(
        f"Unsupported task.name={cfg.task.name}. Use 'grounding' or 'tool_button_detection'."
    )


@app.command()
def extract(
    workdir: str = typer.Option("runs/default", help="Run working directory"),
    video: str | None = typer.Option(None, help="Input video path (mp4)"),
    images: str | None = typer.Option(None, help="Input screenshot folder"),
    fps: float | None = typer.Option(None, help="Extraction fps for video input"),
    every_nth: int = typer.Option(1, help="Keep every N-th frame"),
    max_frames: int | None = typer.Option(None, help="Maximum number of frames before dedup"),
    dedup_threshold: int = typer.Option(8, help="pHash hamming threshold"),
    sample_count: int | None = typer.Option(None, help="Final random sample count"),
) -> None:
    """Extract and curate images from video/screenshots for detector training."""
    if not video and not images:
        raise typer.BadParameter("Provide either --video or --images")

    workdir_path = Path(workdir)
    extract_dir = workdir_path / "extract"
    raw_dir = extract_dir / "raw"
    dedup_dir = extract_dir / "dedup"
    sample_dir = extract_dir / "sample"

    raw_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if video:
        raw_paths = extract_frames(
            video_path=video,
            out_dir=raw_dir,
            fps=fps,
            every_nth=every_nth,
            max_frames=max_frames,
        )
    else:
        src_dir = Path(images or "")
        src_paths = _iter_images(src_dir)
        src_paths = _sample_paths(src_paths, max_frames)
        raw_paths = []
        for i, src in enumerate(src_paths):
            dst = raw_dir / f"frame_{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            raw_paths.append(dst)

    kept, dropped = deduplicate_images(raw_paths, distance_threshold=dedup_threshold)
    for src in kept:
        shutil.copy2(src, dedup_dir / src.name)

    sampled = _sample_paths(kept, sample_count)
    for src in sampled:
        shutil.copy2(src, sample_dir / src.name)

    manifest = {
        "raw_count": len(raw_paths),
        "dedup_count": len(kept),
        "dropped_count": len(dropped),
        "sample_count": len(sampled),
        "raw_dir": str(raw_dir),
        "dedup_dir": str(dedup_dir),
        "sample_dir": str(sample_dir),
    }
    (extract_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(manifest)


@app.command()
def prelabel(
    image_dir: str = typer.Option(..., help="Directory with images for prelabeling"),
    out_coco: str = typer.Option("runs/default/prelabel/candidates.coco.json", help="Output candidate COCO path"),
    backend: str = typer.Option("florence2_teacher", help="Prelabel backend"),
    model_name: str = typer.Option("microsoft/Florence-2-base", help="Teacher model"),
    adapter_path_or_repo: str | None = typer.Option(None, help="Optional adapter for teacher model"),
    class_name: str = typer.Option("candidate_button", help="Proposed class name"),
) -> None:
    """Generate candidate COCO annotations from prelabel plugin."""
    image_paths = _iter_images(Path(image_dir))
    if not image_paths:
        raise typer.BadParameter(f"No images found in {image_dir}")

    coco = empty_coco()
    category_map = register_categories(coco, [class_name])
    prelabeler = create_prelabeler(
        backend=backend,
        model_name=model_name,
        adapter_path_or_repo=adapter_path_or_repo,
    )

    ann_id = 1
    for image_id, path in enumerate(image_paths, start=1):
        add_image_entry(coco, path, image_id=image_id)
        with Image.open(path) as image:
            boxes = prelabeler.predict_boxes(image)

        for box in boxes:
            add_annotation_entry(
                coco,
                annotation_id=ann_id,
                image_id=image_id,
                category_id=category_map[class_name],
                bbox_xyxy=[box.x_min, box.y_min, box.x_max, box.y_max],
                score=0.5,
            )
            ann_id += 1

    save_coco(coco, out_coco)
    typer.echo({"images": len(coco["images"]), "annotations": len(coco["annotations"]), "out": out_coco})


@app.command()
def label(
    image_dir: str = typer.Option(..., help="Directory containing images"),
    candidate_coco: str = typer.Option(..., help="Input candidate COCO annotations"),
    out_coco: str = typer.Option("runs/default/labels/gt.coco.json", help="Output GT COCO path"),
    class_names: str = typer.Option("button", help="Comma-separated class names"),
    server_port: int = typer.Option(7860, help="Gradio server port"),
) -> None:
    """Launch lightweight labeling assist UI."""
    class_list = [x.strip() for x in class_names.split(",") if x.strip()]
    launch_labeling_app(
        image_dir=image_dir,
        candidate_coco_path=candidate_coco,
        out_coco_path=out_coco,
        class_names=class_list,
        server_port=server_port,
    )


@app.command()
def optimize(
    data_yaml: str = typer.Option(..., help="Ultralytics dataset YAML path"),
    val_coco: str = typer.Option(..., help="Validation COCO annotations path"),
    image_dir: str = typer.Option(..., help="Validation image directory"),
    workdir: str = typer.Option("runs/default", help="Output run/work directory"),
    model_name: str = typer.Option("yolov8n.pt", help="YOLO base model"),
    class_names: str = typer.Option("button", help="Comma-separated class names"),
    n_trials: int = typer.Option(10, help="Number of optimization trials"),
    timeout_sec: int | None = typer.Option(None, help="Optional optimization timeout"),
    tool_id: str = typer.Option("tool", help="Tool identifier"),
    tool_version: str = typer.Option("v1", help="Tool version"),
) -> None:
    """Optimize detector hyperparameters with weighted objective and export best run."""
    class_list = [x.strip() for x in class_names.split(",") if x.strip()]
    result = run_optimization(
        model_name=model_name,
        data_yaml=data_yaml,
        val_coco=val_coco,
        image_dir=image_dir,
        workdir=workdir,
        class_names=class_list,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        tool_id=tool_id,
        tool_version=tool_version,
    )
    typer.echo(result.get("table", ""))
    typer.echo({"leaderboard": result.get("leaderboard_path"), "best_package": result.get("package_dir")})


@app.command()
def export(
    weights: str = typer.Option(..., help="Path to trained detector weights (.pt)"),
    outdir: str = typer.Option("runs/default/package", help="Output package directory"),
    class_names: str = typer.Option("button", help="Comma-separated class names"),
    tool_id: str = typer.Option("tool", help="Tool identifier"),
    tool_version: str = typer.Option("v1", help="Tool version"),
    dataset_dir: str | None = typer.Option(None, help="Optional dataset directory for hashing"),
) -> None:
    """Export a self-contained package for runtime inference."""
    wrapper = create_model_wrapper(ModelConfig(backend="yolo_ultralytics", name=weights))
    wrapper.load_model(weights=weights)
    export_dir = Path(outdir) / "artifacts"
    artifacts = wrapper.export(export_dir, formats=["pt", "onnx"])

    names = [x.strip() for x in class_names.split(",") if x.strip()]
    label_map = {name: idx for idx, name in enumerate(names)}

    metrics = {"source": "manual_export"}
    latency = {"latency_ms_mean": 0.0, "latency_ms_p95": 0.0}
    if dataset_dir:
        sample_images = _iter_images(Path(dataset_dir))[:8]
        if sample_images:
            latency = wrapper.benchmark_latency([str(p) for p in sample_images], repeats=1)

    package_dir = Path(outdir)
    create_model_package(
        outdir=package_dir,
        artifacts=artifacts,
        label_map=label_map,
        preprocessing={"imgsz": 640},
        postprocessing={"conf_threshold": 0.25, "iou_threshold": 0.5},
        metrics=metrics,
        latency=latency,
        tool_metadata={
            "tool_id": tool_id,
            "tool_version": tool_version,
            "dataset_hash": compute_dataset_hash(dataset_dir) if dataset_dir else "",
        },
    )
    typer.echo({"package_dir": str(package_dir), "artifacts": artifacts})


@app.command()
def run(
    package_dir: str = typer.Option(..., help="Exported package directory"),
    image_path: str = typer.Option(..., help="Input image path"),
) -> None:
    """Run detector inference from exported package."""
    package = Path(package_dir)
    model_path = package / "model.pt"
    if not model_path.exists():
        raise typer.BadParameter(f"model.pt not found under {package_dir}")

    wrapper = create_model_wrapper(ModelConfig(backend="yolo_ultralytics", name=str(model_path)))
    wrapper.load_model(weights=str(model_path))
    preds = wrapper.predict(image_path)

    output = [
        {
            "class_name": pred.class_name,
            "score": pred.score,
            "bbox": [pred.bbox.x_min, pred.bbox.y_min, pred.bbox.x_max, pred.bbox.y_max],
        }
        for pred in preds
    ]
    typer.echo(output)


@app.command()
def push(
    base_model: str = typer.Option(..., help="Base model id, e.g. microsoft/Florence-2-base"),
    adapter_path: str = typer.Option(..., help="Local adapter path from training output"),
    repo_name: str = typer.Option(..., help="HF repo name, e.g. user/my-vg-adapter"),
    token: str = typer.Option(..., help="HF token"),
) -> None:
    """Upload a trained LoRA adapter to HF Hub."""
    model = Florence2Wrapper(model_name=base_model, use_lora=False)
    model.load_model(adapter_path_or_repo=adapter_path, is_trainable_adapter=False)
    model.push_to_hub(token=token, repo_name=repo_name)


@app.command()
def infer(
    base_model: str = typer.Option(..., help="Base model id, e.g. microsoft/Florence-2-base"),
    image_path: str = typer.Option(..., help="Path to input image"),
    prompt: str = typer.Option(..., help="Grounding prompt, e.g. click the File button"),
    adapter_repo: str | None = typer.Option(
        None,
        help="Optional HF adapter repo or local adapter path",
    ),
) -> None:
    """Load model (and optional LoRA adapter) and run one prediction."""
    if adapter_repo:
        model = Florence2Wrapper.from_pretrained_adapter(
            base_model_name=base_model,
            adapter_path_or_repo=adapter_repo,
        )
    else:
        model = Florence2Wrapper(model_name=base_model, use_lora=False)
        model.load_model()

    image = Image.open(image_path).convert("RGB")
    pred = model.predict(image=image, text=prompt)
    typer.echo(
        {
            "x1": pred.x_min,
            "y1": pred.y_min,
            "x2": pred.x_max,
            "y2": pred.y_max,
        }
    )


@app.command()
def evaluate(
    base_model: str = typer.Option(..., help="Base model id, e.g. microsoft/Florence-2-base"),
    eval_jsonl: str = typer.Option(..., help="Evaluation JSONL path"),
    image_root: str | None = typer.Option(None, help="Optional image root for relative image_path"),
    normalize_mode: str = typer.Option("none", help="bbox normalize mode: none, 0-1, 0-1000"),
    adapter_repo: str | None = typer.Option(
        None,
        help="Optional HF adapter repo or local adapter path",
    ),
) -> None:
    """Run model on eval set and print mean IoU and center pixel distance."""
    if adapter_repo:
        model = Florence2Wrapper.from_pretrained_adapter(
            base_model_name=base_model,
            adapter_path_or_repo=adapter_repo,
        )
    else:
        model = Florence2Wrapper(model_name=base_model, use_lora=False)
        model.load_model()

    dataset = JSONLVisualGroundingDataset(
        source=eval_jsonl,
        image_root=image_root,
        normalize_mode=normalize_mode,
    )
    evaluator = Evaluator()

    pred_boxes: list[BoundingBox] = []
    gt_boxes: list[BoundingBox] = []
    for sample in dataset:
        pred = model.predict(image=sample.image, text=sample.text)
        width = int(sample.metadata["width"]) if sample.metadata and "width" in sample.metadata else 1
        height = int(sample.metadata["height"]) if sample.metadata and "height" in sample.metadata else 1

        pred_boxes.append(_to_pixel_bbox(pred, normalize_mode=normalize_mode, width=width, height=height))
        gt_boxes.append(_to_pixel_bbox(sample.bbox, normalize_mode=normalize_mode, width=width, height=height))

    result = evaluator.evaluate(predictions=pred_boxes, targets=gt_boxes)
    typer.echo(result)


if __name__ == "__main__":
    app()
