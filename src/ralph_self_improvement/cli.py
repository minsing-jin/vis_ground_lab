"""CLI for ralph_self_improvement: RLAIF self-improvement loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

from ralph_self_improvement.core.config import RalphConfig

app = typer.Typer(name="ralph", help="RLAIF self-improvement loop for data_harvest labels.")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.command()
def run(
    config: str = typer.Option("configs/ralph.yaml", "-c", "--config", help="Path to ralph config YAML"),
) -> None:
    """Run the full RLAIF self-improvement loop."""
    _setup_logging()
    cfg = RalphConfig.from_yaml(config)

    from ralph_self_improvement.core.loop import ImprovementLoop

    loop = ImprovementLoop(cfg)
    results = loop.run()

    typer.echo(f"\nCompleted {len(results)} iterations.")
    if results:
        best = max(results, key=lambda r: r.mean_ensemble_score)
        typer.echo(f"Best iteration: #{best.iteration} (score={best.mean_ensemble_score:.4f})")


@app.command()
def judge(
    config: str = typer.Option("configs/ralph.yaml", "-c", "--config", help="Path to ralph config YAML"),
) -> None:
    """Run label quality judgment only (output to JSONL)."""
    _setup_logging()
    cfg = RalphConfig.from_yaml(config)

    from data_harvest.core.config import HarvestConfig
    from data_harvest.core.session import HarvestSession
    from ralph_self_improvement.judge.ensemble import EnsembleJudge

    harvest_cfg = HarvestConfig.from_yaml(cfg.harvest_config_path)
    session = HarvestSession(harvest_cfg)
    samples = session.labeled_samples()

    if not samples:
        typer.echo("No labeled samples found.")
        raise typer.Exit(1)

    ensemble = EnsembleJudge(cfg.judge)
    judgments = ensemble.judge_batch(samples)

    out_path = Path(cfg.judgments_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for j in judgments:
            f.write(j.to_json() + "\n")

    typer.echo(f"Judged {len(judgments)} samples → {out_path}")
    mean_score = sum(j.ensemble_score for j in judgments) / max(len(judgments), 1)
    typer.echo(f"Mean ensemble score: {mean_score:.4f}")


@app.command(name="tune-weights")
def tune_weights(
    config: str = typer.Option("configs/ralph.yaml", "-c", "--config", help="Path to ralph config YAML"),
) -> None:
    """Optimize fusion weights via Bayesian search."""
    _setup_logging()
    cfg = RalphConfig.from_yaml(config)

    from data_harvest.core.config import HarvestConfig
    from data_harvest.core.session import HarvestSession
    from ralph_self_improvement.optimizer.weight_tuner import WeightTuner

    harvest_cfg = HarvestConfig.from_yaml(cfg.harvest_config_path)
    session = HarvestSession(harvest_cfg)
    samples = session.labeled_samples()

    if len(samples) < 5:
        typer.echo(f"Need at least 5 labeled samples (found {len(samples)}).")
        raise typer.Exit(1)

    tuner = WeightTuner(
        config=cfg.weight_tuner,
        judge_config=cfg.judge,
        harvest_config=harvest_cfg,
    )
    snapshot = tuner.tune(samples)

    typer.echo(f"Optimal weights (score={snapshot.objective_value:.4f}):")
    for k, v in sorted(snapshot.weights.items()):
        typer.echo(f"  {k}: {v:.4f}")


@app.command()
def report(
    config: str = typer.Option("configs/ralph.yaml", "-c", "--config", help="Path to ralph config YAML"),
) -> None:
    """Print iteration results summary."""
    _setup_logging()
    cfg = RalphConfig.from_yaml(config)

    from ralph_self_improvement.tracker.metrics import MetricsTracker
    from ralph_self_improvement.tracker.report import generate_report

    tracker = MetricsTracker(cfg.metrics_path)
    typer.echo(generate_report(tracker))


@app.command(name="apply-weights")
def apply_weights(
    config: str = typer.Option("configs/ralph.yaml", "-c", "--config", help="Path to ralph config YAML"),
) -> None:
    """Apply best fusion weights to harvest.yaml."""
    _setup_logging()
    cfg = RalphConfig.from_yaml(config)

    from ralph_self_improvement.tracker.metrics import MetricsTracker

    tracker = MetricsTracker(cfg.metrics_path)
    best = tracker.best_iteration()

    if best is None or best.weight_snapshot is None:
        typer.echo("No weight snapshots found. Run 'ralph run' or 'ralph tune-weights' first.")
        raise typer.Exit(1)

    harvest_path = Path(cfg.harvest_config_path)
    with open(harvest_path, encoding="utf-8") as f:
        harvest_data = yaml.safe_load(f) or {}

    if "labeler" not in harvest_data:
        harvest_data["labeler"] = {}
    harvest_data["labeler"]["fusion_weights"] = best.weight_snapshot.weights

    with open(harvest_path, "w", encoding="utf-8") as f:
        yaml.dump(harvest_data, f, default_flow_style=False, allow_unicode=True)

    typer.echo(f"Applied weights from iteration #{best.iteration} to {harvest_path}")
    for k, v in sorted(best.weight_snapshot.weights.items()):
        typer.echo(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    app()
