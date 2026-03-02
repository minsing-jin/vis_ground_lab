"""Typer CLI for data-harvest commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(name="data-harvest", help="Game-Focused Real-Time Data Harvesting Engine")

logger = logging.getLogger("data_harvest")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_config(config_path: str) -> "HarvestConfig":
    from data_harvest.core.config import HarvestConfig

    return HarvestConfig.from_yaml(config_path)


@app.command()
def record(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Start real-time recording (Ctrl+C to stop)."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.recorder.recorder import HarvestRecorder

    recorder = HarvestRecorder(cfg)
    recorder.run()


@app.command(name="label-auto")
def label_auto(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Auto-label unlabeled samples."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.labeler.fusion import AutoLabeler

    session = HarvestSession(cfg)
    unlabeled = session.unlabeled_samples()
    if not unlabeled:
        typer.echo("No unlabeled samples found.")
        raise typer.Exit()

    labeler = AutoLabeler(cfg.labeler)
    labeled_count = 0
    for sample in unlabeled:
        result = labeler.label_sample(sample)
        if result is not None:
            sample.label = result
            sample.save_label()
            labeled_count += 1

    typer.echo(f"Labeled {labeled_count}/{len(unlabeled)} samples.")


@app.command()
def filter(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Run noise filter pipeline on labeled samples."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.filter.pipeline import FilterPipeline

    session = HarvestSession(cfg)
    samples = session.labeled_samples()
    if not samples:
        typer.echo("No labeled samples found.")
        raise typer.Exit()

    pipeline = FilterPipeline(cfg.filter)
    result = pipeline.run(samples)

    typer.echo(
        f"Filter: {result.total_input} → {result.total_kept} "
        f"(dropped: inv={result.dropped_invalid}, qual={result.dropped_quality}, "
        f"trans={result.dropped_transition}, dup={result.dropped_dedup})"
    )


@app.command()
def review(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Launch Gradio review UI."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.review.review_app import launch_review_app

    launch_review_app(cfg)


@app.command()
def export(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
    format: str = typer.Option("all", "--format", "-f", help="Export format: coco, yolo, grounding, all"),
) -> None:
    """Export dataset to COCO/YOLO/Grounding JSONL format."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.core.types import ReviewStatus

    session = HarvestSession(cfg)
    samples = [
        s for s in session.labeled_samples()
        if s.review_status != ReviewStatus.rejected
    ]
    if not samples:
        typer.echo("No exportable samples found.")
        raise typer.Exit()

    export_dir = Path(cfg.workdir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    formats = ["coco", "yolo", "grounding"] if format == "all" else [format]

    for fmt in formats:
        if fmt == "coco":
            from data_harvest.export.coco_exporter import export_coco

            out = export_coco(samples, export_dir / "annotations.json")
            typer.echo(f"COCO exported: {out}")
        elif fmt == "yolo":
            from data_harvest.export.yolo_exporter import export_yolo

            out = export_yolo(samples, export_dir / "yolo")
            typer.echo(f"YOLO exported: {out}")
        elif fmt == "grounding":
            from data_harvest.export.grounding_exporter import export_grounding

            out = export_grounding(
                samples,
                export_dir / "grounding.jsonl",
                normalizing_range=cfg.export.normalizing_range,
            )
            typer.echo(f"Grounding exported: {out}")
        else:
            typer.echo(f"Unknown format: {fmt}")


@app.command()
def stats(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Print dataset statistics."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.export.stats import compute_stats

    session = HarvestSession(cfg)
    samples = session.iter_samples()
    if not samples:
        typer.echo("No samples found.")
        raise typer.Exit()

    s = compute_stats(samples)
    typer.echo(s.to_report())


@app.command()
def profiles() -> None:
    """List registered game profiles."""
    _setup_logging()

    from data_harvest.profiles.registry import discover_profiles, list_profiles, get_profile

    discover_profiles()
    names = list_profiles()
    if not names:
        typer.echo("No profiles registered.")
        raise typer.Exit()

    typer.echo("Registered game profiles:")
    for name in names:
        p = get_profile(name)
        typer.echo(f"  {name:12s}  {p.display_name}")
        typer.echo(f"               screens: {', '.join(p.screen_types)}")
        typer.echo(f"               semantics: {len(p.semantic_dict)} entries")


if __name__ == "__main__":
    app()
