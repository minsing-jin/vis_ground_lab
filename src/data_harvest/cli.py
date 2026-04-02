"""Typer CLI for data-harvest commands."""

from __future__ import annotations

import base64
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(name="data-harvest", help="Game-Focused Real-Time Data Harvesting Engine")

logger = logging.getLogger("data_harvest")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _get_rich_console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def _render_label_auto_start(
    console,
    *,
    workdir: str,
    provider_name: str,
    unlabeled_count: int,
    cluster_count: int,
    dedup_threshold: int,
) -> None:
    if console is None:
        typer.echo(
            f"label-auto: workdir={workdir} provider={provider_name} "
            f"unlabeled={unlabeled_count} clusters={cluster_count} dedup_threshold={dedup_threshold}"
        )
        return

    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_row("workdir", workdir)
    table.add_row("provider", provider_name)
    table.add_row("unlabeled samples", str(unlabeled_count))
    table.add_row("dedup clusters", str(cluster_count))
    table.add_row("dedup threshold", str(dedup_threshold))
    console.print(Panel(table, title="label-auto", border_style="cyan"))


def _render_label_auto_step(
    console,
    *,
    cluster_index: int,
    cluster_total: int,
    sample_id: str,
    primitive_id: str | None,
    situation_id: str | None,
    roi_name: str | None,
    confidence: float | None,
    copied_duplicates: int,
    status: str,
) -> None:
    primitive_text = primitive_id or "-"
    situation_text = situation_id or "-"
    roi_text = roi_name or "-"
    confidence_text = f"{confidence:.2f}" if confidence is not None else "-"
    line = (
        f"[{cluster_index}/{cluster_total}] {sample_id}  "
        f"status={status}  primitive={primitive_text}  situation={situation_text}  "
        f"roi={roi_text}  conf={confidence_text}  copied={copied_duplicates}"
    )
    if console is None:
        typer.echo(line)
        return
    console.log(line)


def _render_label_auto_summary(
    console,
    *,
    unlabeled_count: int,
    cluster_count: int,
    labeled_count: int,
    gemini_calls: int,
    fallback_count: int,
    copied_count: int,
    blocked_count: int,
    primitive_counter: Counter[str],
) -> None:
    saved_calls = copied_count
    if console is None:
        typer.echo(
            f"Labeled {labeled_count}/{unlabeled_count} samples. "
            f"clusters={cluster_count}, gemini_calls={gemini_calls}, local_fallback={fallback_count}, "
            f"copied={copied_count}, blocked={blocked_count}, gemini_calls_saved={saved_calls}"
        )
        if primitive_counter:
            typer.echo("Primitive counts: " + ", ".join(f"{k}={v}" for k, v in primitive_counter.items()))
        return

    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    summary = Table(show_header=False, box=box.SIMPLE_HEAVY)
    summary.add_row("unlabeled", str(unlabeled_count))
    summary.add_row("clusters", str(cluster_count))
    summary.add_row("labeled", str(labeled_count))
    summary.add_row("gemini calls", str(gemini_calls))
    summary.add_row("local fallback", str(fallback_count))
    summary.add_row("copied duplicates", str(copied_count))
    summary.add_row("blocked duplicates", str(blocked_count))
    summary.add_row("gemini calls saved", str(saved_calls))
    console.print(Panel(summary, title="label-auto summary", border_style="green"))

    if primitive_counter:
        primitive_table = Table(title="primitive distribution", box=box.SIMPLE_HEAVY)
        primitive_table.add_column("primitive_id")
        primitive_table.add_column("count", justify="right")
        for primitive_id, count in primitive_counter.most_common():
            primitive_table.add_row(primitive_id, str(count))
        console.print(primitive_table)


def _load_config(config_path: str) -> "HarvestConfig":
    from data_harvest.core.config import HarvestConfig

    return HarvestConfig.from_yaml(config_path)


def _persist_filter_decisions(
    all_samples: list["HarvestSample"],
    kept_samples: list["HarvestSample"],
) -> tuple[int, int]:
    """Persist filter drops as rejected review status for pending samples."""
    from data_harvest.core.types import ReviewStatus

    kept_ids = {s.sample_id for s in kept_samples}
    auto_rejected = 0
    skipped_non_pending = 0

    for sample in all_samples:
        if sample.sample_id in kept_ids:
            continue
        if sample.review_status != ReviewStatus.pending:
            skipped_non_pending += 1
            continue

        sample.review_status = ReviewStatus.rejected
        review_corrections = dict(sample.review_corrections or {})
        review_corrections.setdefault("auto_filter_rejected", True)
        sample.review_corrections = review_corrections
        sample.save_review()
        auto_rejected += 1

    return auto_rejected, skipped_non_pending


def _update_label_metadata(sample: "HarvestSample", label: "LabelResult") -> None:
    md = dict(sample.metadata or {})
    md["label_auto"] = {
        "page": label.page.to_dict() if label.page else {},
        "route_label": label.route_label.to_dict() if label.route_label else {},
        "confidence": label.confidence,
        "evidence": label.evidence,
    }
    if label.elements or label.candidates:
        import cv2

        pre = cv2.imread(str(sample.pre_frame_path))
        h = int(pre.shape[0]) if pre is not None else None
        w = int(pre.shape[1]) if pre is not None else None
        md["label_auto"]["legacy"] = {
            "elements": [_normalize_element(element.to_dict(), w=w, h=h) for element in label.elements],
            "candidates": [c.to_dict() for c in label.candidates],
        }
    sample.metadata = md
    sample.save_metadata()


def _normalize_element(element: dict[str, object], w: int | None, h: int | None) -> dict[str, object]:
    out = dict(element)
    bbox = [
        float(out.get("bbox_x_min", 0.0)),
        float(out.get("bbox_y_min", 0.0)),
        float(out.get("bbox_x_max", 1.0)),
        float(out.get("bbox_y_max", 1.0)),
    ]
    out["bbox_xyxy"] = bbox
    if w and h:
        out["bbox_norm"] = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
    else:
        out["bbox_norm"] = None
    return out


def _merge_filter_metadata(
    sample: "HarvestSample",
    *,
    cluster_id: str,
    cluster_representative: bool,
    extra_flags: list[str] | None = None,
) -> None:
    md = dict(sample.metadata or {})
    filter_md = dict(md.get("filter") or {})
    existing_flags = filter_md.get("flags") or []
    flags = [str(flag) for flag in existing_flags if str(flag) != "duplicate_non_representative"]
    for flag in extra_flags or []:
        if flag not in flags:
            flags.append(flag)
    if not cluster_representative and "duplicate_non_representative" not in flags:
        flags.append("duplicate_non_representative")
    filter_md["flags"] = flags
    filter_md["cluster_id"] = cluster_id
    filter_md["cluster_representative"] = cluster_representative
    md["filter"] = filter_md
    sample.metadata = md


def _cluster_key_for_sample(sample: "HarvestSample", cluster_ids: dict[str, str]) -> str:
    cluster_id = cluster_ids.get(sample.sample_id)
    if not cluster_id or cluster_id == "missing":
        return f"sample::{sample.sample_id}"
    return cluster_id


def _group_samples_by_cluster(
    samples: list["HarvestSample"],
    *,
    hash_threshold: int,
) -> tuple[list[str], dict[str, list["HarvestSample"]], dict[str, str], dict[str, bool]]:
    cluster_ids, is_representative = _assign_cluster_ids(samples, hash_threshold=hash_threshold)
    ordered_keys: list[str] = []
    grouped: dict[str, list["HarvestSample"]] = {}
    for sample in samples:
        key = _cluster_key_for_sample(sample, cluster_ids)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(sample)
    return ordered_keys, grouped, cluster_ids, is_representative


def _copy_label_result(label: "LabelResult") -> "LabelResult":
    from data_harvest.core.types import LabelResult

    return LabelResult.from_dict(label.to_dict())


def _label_single_sample(
    sample: "HarvestSample",
    *,
    provider,
    provider_name: str,
    provider_init_error: str | None,
    cfg: "HarvestConfig",
    profile,
    labeler,
) -> tuple["LabelResult | None", dict[str, object] | None, bool, bool]:
    from data_harvest.labeler.relabel import apply_relabel_result, build_sample_payload

    result = None
    provider_status: dict[str, object] | None = None
    used_provider = False
    used_fallback = False

    if provider is not None:
        used_provider = True
        try:
            payload = build_sample_payload(
                sample,
                profile=profile,
                include_image_b64=_image_to_base64(sample.pre_frame_path),
            )
            relabel_result = provider.relabel(payload)
            apply_relabel_result(sample, relabel_result)
            result = sample.label
            used_provider = True
            provider_status = {
                "provider": provider_name,
                "model_name": cfg.relabel.model_name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                "mode": "routing_only",
            }
        except Exception as exc:
            provider_status = {
                "provider": provider_name,
                "model_name": cfg.relabel.model_name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": str(exc),
            }
            if not cfg.labeler.provider_fallback_to_local:
                return None, provider_status, used_provider, used_fallback

    if result is None:
        result = labeler.label_sample(sample)
        if result is not None:
            used_fallback = True
            provider_status = {
                "provider": "local_vlm",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                "fallback_from": provider_name if provider_name != "local_vlm" else None,
                "provider_init_error": provider_init_error,
            }

    return result, provider_status, used_provider, used_fallback


def _assign_cluster_ids(
    samples: list["HarvestSample"],
    hash_threshold: int,
) -> tuple[dict[str, str], dict[str, bool]]:
    from PIL import Image
    from vis_ground_lab.data.dedup import hamming_distance, phash_from_image

    reps: list[tuple[str, int]] = []
    cluster_ids: dict[str, str] = {}
    is_representative: dict[str, bool] = {}
    cluster_seq = 1

    for s in samples:
        if not s.pre_frame_path.exists():
            cluster_ids[s.sample_id] = "missing"
            is_representative[s.sample_id] = True
            continue

        h = phash_from_image(Image.open(s.pre_frame_path))
        found_cluster: str | None = None
        for cid, rep_hash in reps:
            if hamming_distance(h, rep_hash) < hash_threshold:
                found_cluster = cid
                break

        if found_cluster is None:
            cid = f"cluster_{cluster_seq:06d}"
            cluster_seq += 1
            reps.append((cid, h))
            cluster_ids[s.sample_id] = cid
            is_representative[s.sample_id] = True
        else:
            cluster_ids[s.sample_id] = found_cluster
            is_representative[s.sample_id] = False

    return cluster_ids, is_representative


def _tag_filter_metadata(
    samples: list["HarvestSample"],
    config: "FilterConfig",
) -> dict[str, int]:
    from data_harvest.filter.invalid_action import is_invalid_action
    from data_harvest.filter.quality import has_quality_issue

    cluster_ids, is_representative = _assign_cluster_ids(
        samples,
        hash_threshold=config.dedup_hash_threshold,
    )
    summary = {
        "quality_issue": 0,
        "duplicate_non_representative": 0,
        "missing_primitive_id": 0,
        "missing_situation_id": 0,
    }

    for s in samples:
        flags: list[str] = []
        if has_quality_issue(
            s,
            blur_threshold=config.blur_laplacian_threshold,
            dark_threshold=config.dark_overlay_threshold,
        ):
            flags.append("quality_issue")
        if s.label is not None:
            label = s.effective_label() or s.label
            if label.route_label is None or not label.route_label.primitive_id:
                flags.append("missing_primitive_id")
            if label.page is None or not label.page.situation_id:
                flags.append("missing_situation_id")
        if is_invalid_action(s, min_diff_ratio=config.min_diff_ratio):
            flags.append("legacy_invalid_action")
        if s.label is not None and s.label.transition_detected:
            flags.append("legacy_transition")
        if not is_representative.get(s.sample_id, True):
            flags.append("duplicate_non_representative")

        for f in flags:
            summary[f] += 1

        score = 1.0
        penalties = {
            "quality_issue": 0.25,
            "duplicate_non_representative": 0.20,
            "missing_primitive_id": 0.35,
            "missing_situation_id": 0.25,
            "legacy_invalid_action": 0.10,
            "legacy_transition": 0.05,
        }
        for f in flags:
            score -= penalties.get(f, 0.0)
        score = max(0.0, min(1.0, score))

        md = dict(s.metadata or {})
        md["filter"] = {
            "flags": flags,
            "score": score,
            "cluster_id": cluster_ids.get(s.sample_id),
            "cluster_representative": is_representative.get(s.sample_id, True),
        }
        s.metadata = md
        s.save_metadata()

    return summary


def _build_relabel_provider(cfg: "HarvestConfig"):
    provider_name = cfg.labeler.provider.strip().lower()
    if provider_name == "gemini":
        from data_harvest.llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            model_name=cfg.relabel.model_name,
            temperature=cfg.relabel.temperature,
            timeout_sec=cfg.relabel.timeout_sec,
            max_retries=cfg.relabel.max_retries,
        )
    raise ValueError(f"Configured teacher provider does not support relabel reruns: {cfg.labeler.provider}")


def _image_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


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
    from data_harvest.profiles.registry import discover_profiles, get_profile

    session = HarvestSession(cfg)
    unlabeled = session.unlabeled_samples()
    if not unlabeled:
        typer.echo("No unlabeled samples found.")
        raise typer.Exit()

    profile = None
    if cfg.game_profile:
        discover_profiles()
        try:
            profile = get_profile(cfg.game_profile)
        except Exception as exc:
            typer.echo(f"Profile load failed ({cfg.game_profile}): {exc}")

    provider = None
    provider_name = cfg.labeler.provider.strip().lower()
    provider_init_error: str | None = None
    console = _get_rich_console()
    if provider_name == "gemini":
        try:
            provider = _build_relabel_provider(cfg)
        except Exception as exc:
            provider_init_error = str(exc)
            if not cfg.labeler.provider_fallback_to_local:
                typer.echo(f"Label provider init failed ({provider_name}): {exc}")
                raise typer.Exit(code=1)
            typer.echo(f"Label provider init failed ({provider_name}); falling back to local_vlm: {exc}")
    elif provider_name != "local_vlm":
        typer.echo(f"Unknown labeler.provider: {cfg.labeler.provider}")
        raise typer.Exit(code=1)

    labeler = AutoLabeler(cfg.labeler, profile=profile)
    labeled_count = 0
    gemini_count = 0
    fallback_count = 0
    copied_count = 0
    blocked_count = 0
    primitive_counter: Counter[str] = Counter()
    ordered_clusters, grouped_samples, cluster_ids, _ = _group_samples_by_cluster(
        unlabeled,
        hash_threshold=cfg.filter.dedup_hash_threshold,
    )
    _render_label_auto_start(
        console,
        workdir=cfg.workdir,
        provider_name=provider_name,
        unlabeled_count=len(unlabeled),
        cluster_count=len(ordered_clusters),
        dedup_threshold=cfg.filter.dedup_hash_threshold,
    )

    progress = None
    task_id = None
    if console is not None:
        try:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            )
            progress.start()
            task_id = progress.add_task("label-auto representatives", total=len(ordered_clusters))
        except Exception:
            progress = None
            task_id = None

    try:
        for cluster_index, cluster_key in enumerate(ordered_clusters, start=1):
            cluster_samples = grouped_samples[cluster_key]
            representative = cluster_samples[0]
            rep_cluster_id = cluster_ids.get(representative.sample_id) or cluster_key

            result, provider_status, used_provider, used_fallback = _label_single_sample(
                representative,
                provider=provider,
                provider_name=provider_name,
                provider_init_error=provider_init_error,
                cfg=cfg,
                profile=profile,
                labeler=labeler,
            )
            if used_provider:
                gemini_count += 1
            if used_fallback:
                fallback_count += 1

            if result is not None:
                representative.label = result
                md = dict(representative.metadata or {})
                if provider_status is not None:
                    md["label_auto_provider"] = provider_status
                representative.metadata = md
                _merge_filter_metadata(
                    representative,
                    cluster_id=rep_cluster_id,
                    cluster_representative=True,
                )
                representative.save_label()
                _update_label_metadata(representative, result)
                labeled_count += 1

                primitive_id = result.route_label.primitive_id if result.route_label else None
                if primitive_id:
                    primitive_counter[primitive_id] += 1

                copied_for_cluster = 0
                for duplicate in cluster_samples[1:]:
                    duplicate.label = _copy_label_result(result)
                    duplicate_provider_status = {
                        "provider": (provider_status or {}).get("provider", provider_name),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "copied_from_representative",
                        "source_sample_id": representative.sample_id,
                        "source_provider_status": (provider_status or {}).get("status"),
                    }
                    md = dict(duplicate.metadata or {})
                    md["label_auto_provider"] = duplicate_provider_status
                    duplicate.metadata = md
                    _merge_filter_metadata(
                        duplicate,
                        cluster_id=rep_cluster_id,
                        cluster_representative=False,
                    )
                    duplicate.save_label()
                    _update_label_metadata(duplicate, duplicate.label)
                    labeled_count += 1
                    copied_count += 1
                    copied_for_cluster += 1
                    if primitive_id:
                        primitive_counter[primitive_id] += 1

                _render_label_auto_step(
                    console,
                    cluster_index=cluster_index,
                    cluster_total=len(ordered_clusters),
                    sample_id=representative.sample_id,
                    primitive_id=primitive_id,
                    situation_id=(result.page.situation_id if result.page else None),
                    roi_name=(result.route_label.roi_name if result.route_label else None),
                    confidence=result.confidence,
                    copied_duplicates=copied_for_cluster,
                    status=(provider_status or {}).get("provider", "unknown"),
                )
            else:
                if provider_status is not None:
                    md = dict(representative.metadata or {})
                    md["label_auto_provider"] = provider_status
                    representative.metadata = md
                _merge_filter_metadata(
                    representative,
                    cluster_id=rep_cluster_id,
                    cluster_representative=True,
                )
                representative.save_metadata()

                blocked_for_cluster = 0
                for duplicate in cluster_samples[1:]:
                    duplicate_provider_status = {
                        "provider": (provider_status or {}).get("provider", provider_name),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "blocked_by_representative_failure",
                        "source_sample_id": representative.sample_id,
                        "source_provider_status": (provider_status or {}).get("status"),
                    }
                    md = dict(duplicate.metadata or {})
                    md["label_auto_provider"] = duplicate_provider_status
                    duplicate.metadata = md
                    _merge_filter_metadata(
                        duplicate,
                        cluster_id=rep_cluster_id,
                        cluster_representative=False,
                    )
                    duplicate.save_metadata()
                    blocked_count += 1
                    blocked_for_cluster += 1

                _render_label_auto_step(
                    console,
                    cluster_index=cluster_index,
                    cluster_total=len(ordered_clusters),
                    sample_id=representative.sample_id,
                    primitive_id=None,
                    situation_id=None,
                    roi_name=None,
                    confidence=None,
                    copied_duplicates=blocked_for_cluster,
                    status=(provider_status or {}).get("status", "failed"),
                )

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    _render_label_auto_summary(
        console,
        unlabeled_count=len(unlabeled),
        cluster_count=len(ordered_clusters),
        labeled_count=labeled_count,
        gemini_calls=gemini_count,
        fallback_count=fallback_count,
        copied_count=copied_count,
        blocked_count=blocked_count,
        primitive_counter=primitive_counter,
    )


@app.command()
def relabel(
    config: str = typer.Option("configs/harvest.yaml", "-c", "--config", help="Config YAML path"),
) -> None:
    """Rerun the configured teacher on labeled samples to refresh labels."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.labeler.relabel import apply_relabel_result, build_sample_payload
    from data_harvest.profiles.registry import discover_profiles, get_profile

    session = HarvestSession(cfg)
    samples = session.labeled_samples()
    if not samples:
        typer.echo("No labeled samples found.")
        raise typer.Exit()

    discover_profiles()
    profile = None
    if cfg.game_profile:
        try:
            profile = get_profile(cfg.game_profile)
        except Exception as exc:
            typer.echo(f"Profile load failed ({cfg.game_profile}): {exc}")

    try:
        provider = _build_relabel_provider(cfg)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    processed = 0
    success = 0
    failed = 0
    max_samples = cfg.relabel.max_samples or len(samples)
    for sample in samples[:max_samples]:
        if sample.label is None:
            continue
        processed += 1
        try:
            pre_b64 = None
            if sample.pre_frame_path.exists():
                from data_harvest.llm.gemini_provider import GeminiProvider

                pre_b64 = GeminiProvider.image_to_base64(str(sample.pre_frame_path))

            payload = build_sample_payload(sample, profile=profile, include_image_b64=pre_b64)
            result = provider.relabel(payload)
            apply_relabel_result(sample, result)
            sample.save_label()

            md = dict(sample.metadata or {})
            md["relabel"] = {
                "provider": cfg.labeler.provider,
                "model_name": cfg.relabel.model_name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                "mode": "routing_only",
            }
            sample.metadata = md
            sample.save_metadata()
            success += 1
        except Exception as exc:
            md = dict(sample.metadata or {})
            md["relabel"] = {
                "provider": cfg.labeler.provider,
                "model_name": cfg.relabel.model_name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": str(exc),
            }
            sample.metadata = md
            sample.save_metadata()
            failed += 1

    typer.echo(
        f"Relabel rerun done: processed={processed}, success={success}, failed={failed}, provider={cfg.labeler.provider}"
    )


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
    summary = _tag_filter_metadata(samples, cfg.filter)

    typer.echo(
        f"Filter: {result.total_input} → {result.total_kept} "
        f"(dropped: inv={result.dropped_invalid}, qual={result.dropped_quality}, "
        f"trans={result.dropped_transition}, dup={result.dropped_dedup})"
    )
    typer.echo(
        "Routing tags: "
        f"missing_primitive={summary['missing_primitive_id']}, "
        f"missing_situation={summary['missing_situation_id']}, "
        f"quality={summary['quality_issue']}, "
        f"dup={summary['duplicate_non_representative']}"
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
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Export format: router_full, router_roi, router, unified, coco, yolo, grounding, roi_state, legacy_all, all",
    ),
) -> None:
    """Export routing datasets and optional legacy formats."""
    _setup_logging()
    cfg = _load_config(config)

    from data_harvest.core.session import HarvestSession
    from data_harvest.core.types import ReviewStatus
    from data_harvest.profiles.registry import discover_profiles, get_profile

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

    profile = None
    if cfg.game_profile:
        discover_profiles()
        try:
            profile = get_profile(cfg.game_profile)
        except Exception:
            logger.warning("Failed to load export profile '%s'.", cfg.game_profile, exc_info=True)

    if format == "all":
        formats = ["router_full", "router_roi"]
    elif format == "legacy_all":
        formats = ["unified", "coco", "yolo", "grounding", "roi_state"]
    elif format == "router":
        formats = ["router_full", "router_roi"]
    else:
        formats = [format]

    for fmt in formats:
        if fmt == "router_full":
            from data_harvest.export.router_exporter import export_router_full

            out = export_router_full(samples, export_dir / "router_full")
            typer.echo(f"Router full exported: {out}")
        elif fmt == "router_roi":
            from data_harvest.export.router_exporter import export_router_roi

            out = export_router_roi(
                samples,
                export_dir / "router_roi",
                profile=profile,
                fallback_rois=cfg.export.router_roi_fallbacks,
            )
            typer.echo(f"Router ROI exported: {out}")
        elif fmt == "coco":
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
        elif fmt == "roi_state":
            from data_harvest.export.roi_state_exporter import export_roi_state

            out = export_roi_state(samples, export_dir / "roi_state")
            typer.echo(f"ROI state exported: {out}")
        elif fmt == "unified":
            from data_harvest.export.unified_exporter import export_unified

            out = export_unified(samples, export_dir / "unified.jsonl")
            typer.echo(f"Unified exported: {out}")
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
