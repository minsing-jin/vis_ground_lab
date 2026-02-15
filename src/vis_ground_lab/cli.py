"""Command line interface for vis_ground_lab."""

from __future__ import annotations

import typer
from PIL import Image

from vis_ground_lab.base import BoundingBox
from vis_ground_lab.config.loader import load_train_config
from vis_ground_lab.data_manager import JSONLVisualGroundingDataset
from vis_ground_lab.evaluation import Evaluator
from vis_ground_lab.models.florence2 import Florence2Wrapper
from vis_ground_lab.training.trainer_engine import TrainerEngine

app = typer.Typer(help="Visual grounding training toolkit")


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


@app.command()
def train(config: str = typer.Option(..., "--config", "-c", help="Path to training YAML config")) -> None:
    """Run training with Florence-2 + LoRA from a YAML config file."""
    cfg = load_train_config(config)

    model = Florence2Wrapper(
        model_name=cfg.model.name,
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
    )
    model.load_model()

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
