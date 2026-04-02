"""Text-based summary report with trend indicators."""

from __future__ import annotations

from ralph_self_improvement.tracker.metrics import MetricsTracker


def _trend(current: float, previous: float) -> str:
    """Return a trend indicator arrow."""
    diff = current - previous
    if abs(diff) < 0.001:
        return "="
    return "^" if diff > 0 else "v"


def generate_report(tracker: MetricsTracker) -> str:
    """Generate a text summary of all iterations."""
    results = tracker.results
    if not results:
        return "No iterations recorded yet."

    lines = [
        "=" * 60,
        "  RALPH Self-Improvement Report",
        "=" * 60,
        "",
        f"  Total iterations: {len(results)}",
    ]

    best = tracker.best_iteration()
    if best is not None:
        lines.append(f"  Best iteration:   #{best.iteration} (score={best.mean_ensemble_score:.4f})")
    lines.append("")

    # Per-iteration table
    header = f"  {'Iter':>4}  {'Score':>7}  {'IoU':>7}  {'Dist':>8}  {'Samples':>7}  {'Pairs':>5}  {'Trend':>5}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    prev_score = 0.0
    for r in results:
        trend = _trend(r.mean_ensemble_score, prev_score) if r.iteration > 1 else " "
        lines.append(
            f"  {r.iteration:>4}  {r.mean_ensemble_score:>7.4f}  {r.mean_iou:>7.4f}  "
            f"{r.mean_distance_px:>8.1f}  {r.n_samples:>7}  {r.n_preference_pairs:>5}  "
            f"{'  ' + trend:>5}"
        )
        prev_score = r.mean_ensemble_score

    lines.append("")

    # Weight snapshot from best iteration
    if best is not None and best.weight_snapshot is not None:
        lines.append("  Best fusion weights:")
        for k, v in sorted(best.weight_snapshot.weights.items()):
            lines.append(f"    {k}: {v:.4f}")
        lines.append("")

    if best is not None and best.checkpoint_path:
        lines.append(f"  Best checkpoint: {best.checkpoint_path}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
