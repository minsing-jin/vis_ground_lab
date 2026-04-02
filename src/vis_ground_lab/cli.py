"""Command line interface for vis_ground_lab."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import typer
from PIL import Image
from torch.utils.data import DataLoader

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.config.schema import ModelConfig
from vis_ground_lab.config.loader import load_factory_config, load_train_config
from vis_ground_lab.data import (
    add_annotation_entry,
    add_image_entry,
    deduplicate_images,
    empty_coco,
    extract_frames,
    register_categories,
    save_coco,
)
from vis_ground_lab.data_manager import JSONLVisualGroundingDataset, RouterClassificationDataset
from vis_ground_lab.evaluation import Evaluator
from vis_ground_lab.export import compute_dataset_hash, create_model_package
from vis_ground_lab.labeling import launch_labeling_app
from vis_ground_lab.models.factory import create_model_wrapper
from vis_ground_lab.models.florence2 import Florence2Wrapper
from vis_ground_lab.optimization.optuna_runner import evaluate_detector_checkpoint
from vis_ground_lab.optimization import run_optimization
from vis_ground_lab.prelabel import create_prelabeler
from vis_ground_lab.training import RouterTrainer, TrainerEngine

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


def _build_evaluation_result(
    *,
    task: str,
    primary_metric: str,
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    return {
        "task": task,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "artifacts": artifacts or {},
        "checkpoint_path": checkpoint_path,
    }


def _evaluate_grounding_task(
    *,
    base_model: str,
    eval_jsonl: str,
    image_root: str | None,
    normalize_mode: str,
    adapter_repo: str | None,
) -> dict[str, Any]:
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

    metrics = evaluator.evaluate(predictions=pred_boxes, targets=gt_boxes)
    return _build_evaluation_result(
        task="grounding",
        primary_metric="mean_iou",
        metrics=metrics,
        artifacts={"eval_jsonl": eval_jsonl},
        checkpoint_path=adapter_repo,
    )


def _resolve_detector_weights(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    candidates = [
        checkpoint_dir / "best_package" / "model.pt",
        checkpoint_dir / "best_export" / "model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    best_weights = sorted(checkpoint_dir.glob("yolo_runs/**/weights/best.pt"))
    if best_weights:
        return best_weights[-1]
    return None


def _extract_reported_map50(checkpoint_dir: str | Path) -> float:
    checkpoint_dir = Path(checkpoint_dir)
    metrics_path = checkpoint_dir / "best_package" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        try:
            return float(metrics.get("mAP50", 0.0))
        except (TypeError, ValueError):
            return 0.0

    leaderboard_path = checkpoint_dir / "leaderboard.json"
    if leaderboard_path.exists():
        rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        if rows:
            try:
                return float(rows[0].get("mAP50", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _evaluate_detector_task(cfg, checkpoint_path: str | None = None) -> dict[str, Any]:
    weights_path = Path(checkpoint_path) if checkpoint_path else _resolve_detector_weights(cfg.trainer.checkpoint_dir)
    if weights_path is None or not weights_path.exists():
        raise typer.BadParameter("Detector checkpoint not found. Run train/optimize first or pass --checkpoint-path.")

    if not cfg.data.val_coco or not cfg.data.image_root:
        raise typer.BadParameter("Detector evaluation requires data.val_coco and data.image_root")

    metrics = evaluate_detector_checkpoint(
        model_name=cfg.model.name,
        weights=str(weights_path),
        val_coco=cfg.data.val_coco,
        image_dir=cfg.data.image_root,
        reported_map50=_extract_reported_map50(cfg.trainer.checkpoint_dir),
    )
    return _build_evaluation_result(
        task="tool_button_detection",
        primary_metric="score",
        metrics=metrics,
        artifacts={"val_coco": cfg.data.val_coco, "image_dir": cfg.data.image_root},
        checkpoint_path=str(weights_path),
    )


def _evaluate_router_task(cfg, checkpoint_path: str | None = None) -> dict[str, Any]:
    train_dataset = RouterClassificationDataset(
        source=cfg.data.train_csv,
        image_root=cfg.data.image_root,
        image_size=cfg.model.router_image_size,
        label_column=cfg.data.label_column,
        aux_label_columns=cfg.data.aux_label_columns,
    )
    val_dataset = RouterClassificationDataset(
        source=cfg.data.val_csv,
        image_root=cfg.data.image_root,
        image_size=cfg.model.router_image_size,
        label_column=cfg.data.label_column,
        aux_label_columns=cfg.data.aux_label_columns,
        label_to_index=train_dataset.label_to_index,
        aux_label_to_index=train_dataset.aux_label_to_index,
    )

    wrapper = create_model_wrapper(cfg.model)
    resolved_checkpoint = checkpoint_path or str(Path(cfg.trainer.checkpoint_dir) / "best_router.pt")
    wrapper.load_model(
        checkpoint_path=resolved_checkpoint,
        label_to_index=train_dataset.label_to_index,
        aux_label_to_index=train_dataset.aux_label_to_index,
    )

    trainer = RouterTrainer(wrapper, cfg.trainer, aux_loss_weight=cfg.model.router_aux_loss_weight)
    loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        collate_fn=RouterTrainer._collate_fn,
    )
    metrics = trainer.evaluate(loader)
    return _build_evaluation_result(
        task="router_classification",
        primary_metric="primitive_macro_f1",
        metrics=metrics,
        artifacts={"val_csv": cfg.data.val_csv, "label_column": cfg.data.label_column},
        checkpoint_path=resolved_checkpoint,
    )


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

    if task_name == "router_classification":
        if cfg.model.backend.lower() != "timm_router":
            raise typer.BadParameter("router_classification currently requires model.backend=timm_router")

        train_dataset = RouterClassificationDataset(
            source=cfg.data.train_csv,
            image_root=cfg.data.image_root,
            image_size=cfg.model.router_image_size,
            label_column=cfg.data.label_column,
            aux_label_columns=cfg.data.aux_label_columns,
        )
        val_dataset = RouterClassificationDataset(
            source=cfg.data.val_csv,
            image_root=cfg.data.image_root,
            image_size=cfg.model.router_image_size,
            label_column=cfg.data.label_column,
            aux_label_columns=cfg.data.aux_label_columns,
            label_to_index=train_dataset.label_to_index,
            aux_label_to_index=train_dataset.aux_label_to_index,
        )

        wrapper = create_model_wrapper(cfg.model)
        trainer = RouterTrainer(wrapper, cfg.trainer, aux_loss_weight=cfg.model.router_aux_loss_weight)
        metrics = trainer.train(train_dataset=train_dataset, val_dataset=val_dataset)
        typer.echo(json.dumps(metrics, indent=2))
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
        typer.echo(json.dumps(metrics, indent=2))
        return

    raise typer.BadParameter(
        f"Unsupported task.name={cfg.task.name}. Use 'grounding', 'router_classification', or 'tool_button_detection'."
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
    config: str | None = typer.Option(None, "--config", "-c", help="Optional training YAML config"),
    checkpoint_path: str | None = typer.Option(None, help="Optional checkpoint/model override for router/detector/grounding"),
    base_model: str | None = typer.Option(None, help="Base model id, e.g. microsoft/Florence-2-base"),
    eval_jsonl: str | None = typer.Option(None, help="Evaluation JSONL path"),
    image_root: str | None = typer.Option(None, help="Optional image root for relative image_path"),
    normalize_mode: str = typer.Option("none", help="bbox normalize mode: none, 0-1, 0-1000"),
    adapter_repo: str | None = typer.Option(
        None,
        help="Optional HF adapter repo or local adapter path",
    ),
) -> None:
    """Run model on eval set and print mean IoU and center pixel distance."""
    if config:
        cfg = load_train_config(config)
        task_name = cfg.task.name.lower()
        if task_name == "router_classification":
            result = _evaluate_router_task(cfg, checkpoint_path=checkpoint_path)
        elif task_name == "tool_button_detection":
            result = _evaluate_detector_task(cfg, checkpoint_path=checkpoint_path)
        elif task_name == "grounding":
            eval_path = cfg.data.eval_jsonl or cfg.data.train_jsonl
            result = _evaluate_grounding_task(
                base_model=cfg.model.name,
                eval_jsonl=eval_path,
                image_root=cfg.data.image_root,
                normalize_mode=cfg.data.normalize_mode,
                adapter_repo=checkpoint_path or cfg.model.adapter_path_or_repo or cfg.trainer.checkpoint_dir,
            )
        else:
            raise typer.BadParameter(f"Unsupported task.name={cfg.task.name}")
        typer.echo(json.dumps(result, indent=2))
        return

    if not base_model or not eval_jsonl:
        raise typer.BadParameter("Provide either --config or both --base-model and --eval-jsonl")

    result = _evaluate_grounding_task(
        base_model=base_model,
        eval_jsonl=eval_jsonl,
        image_root=image_root,
        normalize_mode=normalize_mode,
        adapter_repo=adapter_repo or checkpoint_path,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def capture(
    config: str = typer.Option(..., "--config", "-c", help="Path to factory YAML config"),
    workdir: str = typer.Option("runs/capture", help="Working directory for capture output"),
) -> None:
    """Parse input logs, correlate with frames, auto-label, and route to HITL."""
    from vis_ground_lab.capture import ActionFrameMatcher, InputLogParser
    from vis_ground_lab.hitl import ConfidenceScorer, ReviewQueue
    from vis_ground_lab.hitl.review_queue import ReviewItem

    cfg = load_factory_config(config)
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Parse input log
    if not cfg.capture.input_log_path:
        raise typer.BadParameter("capture.input_log_path is required in config")

    if cfg.capture.input_log_format == "csv":
        events = InputLogParser.from_csv(cfg.capture.input_log_path)
    else:
        events = InputLogParser.from_jsonl(cfg.capture.input_log_path)
    typer.echo(f"Parsed {len(events)} input events")

    # Determine frame dir
    frame_dir = cfg.capture.frame_dir
    if not frame_dir:
        if cfg.capture.video_path:
            frame_dir = str(workdir_path / "frames")
            extract_frames(
                video_path=cfg.capture.video_path,
                out_dir=frame_dir,
                fps=cfg.capture.fps,
            )
        else:
            raise typer.BadParameter("capture.frame_dir or capture.video_path is required")

    # Match actions to frames
    matcher = ActionFrameMatcher(
        frame_dir=frame_dir,
        fps=cfg.capture.fps,
        time_tolerance_ms=cfg.capture.time_tolerance_ms,
    )
    pairs = matcher.match(events)
    typer.echo(f"Matched {len(pairs)} action-frame pairs")

    # Export as COCO
    coco_path = workdir_path / "auto_labels.coco.json"
    matcher.to_coco(pairs, class_names=["button"], out_path=coco_path, crop_radius_px=cfg.capture.crop_radius_px)
    typer.echo(f"COCO saved to {coco_path}")

    # Route low-confidence pairs to HITL queue
    scorer = ConfidenceScorer(
        low_confidence_threshold=cfg.hitl.low_confidence_threshold,
        ambiguity_iou_threshold=cfg.hitl.ambiguity_iou_threshold,
    )
    queue = ReviewQueue(cfg.hitl.queue_dir)
    routed = 0
    for pair in pairs:
        if pair.auto_label is None:
            from datetime import datetime, timezone

            item = ReviewItem(
                image_path=str(pair.frame_path),
                frame_id=f"capture_{pair.frame_index}",
                elements=[],
                uncertainty_score=1.0,
                source="auto_capture",
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
            )
            queue.enqueue(item)
            routed += 1

    typer.echo(f"Routed {routed} items to HITL queue at {cfg.hitl.queue_dir}")


@app.command()
def review(
    queue_dir: str = typer.Option("runs/hitl_queue", help="HITL queue directory"),
    out_coco: str = typer.Option("runs/hitl_queue/corrections.coco.json", help="Output corrections COCO"),
) -> None:
    """Show HITL review queue stats and export corrections."""
    from vis_ground_lab.hitl import ReviewQueue

    queue = ReviewQueue(queue_dir)
    stats = queue.stats()
    typer.echo(f"Queue stats: {stats}")

    pending = queue.peek(n=5)
    if pending:
        typer.echo(f"Top {len(pending)} pending items:")
        for item in pending:
            typer.echo(f"  {item.frame_id}: score={item.uncertainty_score:.3f} source={item.source}")

    if stats["with_corrections"] > 0:
        queue.export_corrections_as_coco(out_coco)
        typer.echo(f"Corrections exported to {out_coco}")


@app.command()
def factory(
    config: str = typer.Option(..., "--config", "-c", help="Path to factory YAML config"),
    workdir: str = typer.Option("runs/factory", help="Factory working directory"),
) -> None:
    """Full lifecycle: profile data → select strategy → train → export → create profile."""
    from vis_ground_lab.profile import ToolProfile
    from vis_ground_lab.strategy import AutoStrategySelector, DataProfiler

    cfg = load_factory_config(config)
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    # Profile data
    profiler = DataProfiler()
    if cfg.data.dataset_yaml:
        profile = profiler.profile_coco(cfg.data.dataset_yaml, cfg.data.image_root)
    elif cfg.data.train_jsonl:
        profile = profiler.profile_jsonl(cfg.data.train_jsonl, cfg.data.image_root)
    else:
        raise typer.BadParameter("data.train_jsonl or data.dataset_yaml is required")
    typer.echo(f"Data profile: {profile}")

    # Auto-select strategy
    selector = AutoStrategySelector()
    strategy = selector.select(profile)
    typer.echo(f"Selected strategy: {strategy.rationale}")

    # Build config and train
    train_cfg = selector.to_train_run_config(strategy, cfg.data)
    model = create_model_wrapper(train_cfg.model)
    model.load_model()

    if train_cfg.task.name == "grounding":
        train_dataset = JSONLVisualGroundingDataset(
            source=train_cfg.data.train_jsonl,
            image_root=train_cfg.data.image_root,
            normalize_mode=train_cfg.data.normalize_mode,
        )
        engine = TrainerEngine(model_wrapper=model, config=train_cfg.trainer)
        engine.train(train_dataset=train_dataset)
    else:
        dataset_yaml = train_cfg.data.dataset_yaml or train_cfg.data.train_jsonl
        model.train(
            dataset=dataset_yaml,
            cfg={
                "epochs": train_cfg.trainer.epochs,
                "batch_size": train_cfg.trainer.batch_size,
                "learning_rate": train_cfg.trainer.learning_rate,
            },
            workdir=str(workdir_path),
        )

    # Create tool profile
    profile_data = ToolProfile(
        tool_id=cfg.tool_id,
        tool_version=cfg.tool_version,
        package_dir=str(workdir_path),
        model_cfg=train_cfg.model,
        data_config=train_cfg.data,
        runtime_config=cfg.runtime.model_dump(),
    )
    profile_path = workdir_path / "tool_profile.json"
    profile_data.save(profile_path)
    typer.echo(f"Tool profile saved to {profile_path}")


@app.command()
def monitor(
    profile_path: str = typer.Option(..., help="Path to tool_profile.json"),
    image_path: str = typer.Option(..., help="Input image to analyze"),
) -> None:
    """Analyze a frame and report runtime monitoring signals."""
    from vis_ground_lab.profile import ToolProfile
    from vis_ground_lab.runtime import FrameAnalyzer, RuntimeMonitor

    tp = ToolProfile.load(profile_path)
    model = tp.get_model_wrapper()

    analyzer = FrameAnalyzer(model=model)
    analysis = analyzer.analyze(image_path)

    ref_frames = None
    if tp.reference_frames_dir:
        ref_dir = Path(tp.reference_frames_dir)
        ref_frames = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg"))

    rt_monitor = RuntimeMonitor(
        reference_frames=ref_frames,
        drift_hash_threshold=tp.runtime_config.get("drift_hash_threshold", 12),
        low_confidence_threshold=tp.runtime_config.get("low_confidence_threshold", 0.3),
    )

    img = Image.open(image_path)
    signals = rt_monitor.observe(analysis, img)

    typer.echo(json.dumps({"analysis": analysis.to_dict(), "signals": signals}, indent=2, default=str))


@app.command()
def retrain(
    profile_path: str = typer.Option(..., help="Path to tool_profile.json"),
    force: bool = typer.Option(False, help="Force retrain regardless of conditions"),
) -> None:
    """Evaluate retrain conditions and optionally trigger incremental update."""
    from vis_ground_lab.hitl import ReviewQueue
    from vis_ground_lab.profile import ToolProfile
    from vis_ground_lab.runtime import FailureStore, RetrainTrigger

    tp = ToolProfile.load(profile_path)

    failure_store = FailureStore(tp.failure_store_dir or "runs/failures")
    review_queue = ReviewQueue(tp.review_queue_dir or "runs/hitl_queue")

    trigger = RetrainTrigger(
        failure_threshold=tp.runtime_config.get("failure_threshold", 50),
        correction_threshold=tp.runtime_config.get("correction_threshold", 20),
    )

    decision = trigger.evaluate(failure_store, review_queue)
    typer.echo(f"Retrain decision: should_retrain={decision.should_retrain}, reason={decision.reason}")
    typer.echo(f"  failures={decision.failure_count}, corrections={decision.correction_count}")
    typer.echo(f"  recommended_strategy={decision.recommended_strategy}")

    if not decision.should_retrain and not force:
        typer.echo("No retrain needed.")
        return

    if force:
        typer.echo("Forced retrain triggered.")

    typer.echo("Retrain would be executed here with the tool profile's model and data config.")
    trigger.record_retrain()


if __name__ == "__main__":
    app()
