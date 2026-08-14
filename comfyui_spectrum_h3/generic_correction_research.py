from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generic_correction_evaluator import (
    CalibrationError,
    analyze_group,
    compatibility_signature,
    independence_identity,
    live_configuration_for_candidate,
    validate_blocks,
)

LOG = logging.getLogger(__name__)

STORE_SCHEMA_VERSION = 1
STORE_DIRECTORY_NAME = "spectrum_h3/generic_correction/v1"
MAX_RUNS_PER_GROUP = 12
MAX_STORED_RUNS = 96
MAX_REPORT_GROUPS = 24
MAX_QUARANTINED_FILES = 16
TOP_CONSOLE_CANDIDATES = 3


@dataclass(frozen=True, slots=True)
class ResearchResult:
    group_id: str
    run_count: int
    duplicate: bool
    run_path: Path | None
    json_report_path: Path
    markdown_report_path: Path
    report: dict[str, Any]
    console_summary: str
    elapsed_seconds: float


def default_store_root() -> Path:
    """Resolve ComfyUI's internal user cache without importing it at module load."""
    try:
        import folder_paths

        system_directory = getattr(folder_paths, "get_system_user_directory", None)
        if callable(system_directory):
            base = Path(system_directory("cache"))
        else:
            user_directory = getattr(folder_paths, "get_user_directory", None)
            if callable(user_directory):
                base = Path(user_directory()) / "__cache"
            else:
                base = Path(folder_paths.user_directory) / "__cache"
    except (ImportError, AttributeError, TypeError, ValueError):
        # This fallback supports source tests and older ComfyUI revisions. In a
        # live install folder_paths is available and the branch above is used.
        base = Path.cwd() / "user" / "__cache"
    return base / STORE_DIRECTORY_NAME


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _file_order(path: Path) -> tuple[int, str]:
    try:
        modified = path.stat().st_mtime_ns
    except OSError:
        modified = 0
    return modified, path.name


def _trim_files(paths: list[Path], keep: int) -> None:
    ordered = sorted(paths, key=_file_order)
    for path in ordered[: max(0, len(ordered) - keep)]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _quarantine(path: Path, root: Path, reason: str) -> None:
    quarantine = root / "corrupt"
    quarantine.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(reason.encode("utf-8")).hexdigest()[:8]
    target = quarantine / f"{path.stem}.{suffix}.json"
    counter = 1
    while target.exists():
        target = quarantine / f"{path.stem}.{suffix}.{counter}.json"
        counter += 1
    try:
        os.replace(path, target)
    except OSError:
        LOG.warning(
            "Spectrum H3 could not quarantine corrupt generic-correction state %s",
            path,
        )
    _trim_files(list(quarantine.glob("*.json")), MAX_QUARANTINED_FILES)


def _load_stored_blocks(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs_directory = root / "runs"
    if not runs_directory.exists():
        return []
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(runs_directory.glob("*.json"), key=lambda item: item.name):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise CalibrationError("stored calibration run is not a JSON object")
            validate_blocks([value])
        except (CalibrationError, json.JSONDecodeError, OSError, TypeError) as exc:
            LOG.warning(
                "Spectrum H3 ignored corrupt generic-correction state %s: %s",
                path,
                exc,
            )
            _quarantine(path, root, str(exc))
            continue
        loaded.append((path, value))
    return loaded


def _group_id(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]


def _validation_label(run_count: int) -> tuple[str, str]:
    if run_count <= 1:
        return "development", "development only / non-confirmatory"
    if run_count == 2:
        return "preliminary_loro", "preliminary whole-run leave-one-out"
    return "loro_generalization", "whole-run leave-one-out generalization"


def _recommendation(group: dict[str, Any]) -> dict[str, Any]:
    run_count = int(group["run_count"])
    equivalence_groups = group.get("candidate_equivalence_groups") or []
    if not equivalence_groups:
        return {
            "available": False,
            "reason": "no scoreable candidate equivalence group",
        }
    best_group = equivalence_groups[0]
    mappings = [
        live_configuration_for_candidate(candidate)
        for candidate in best_group["members"]
    ]
    reproducible = [item for item in mappings if item["live_reproducible"]]
    if not reproducible:
        return {
            "available": False,
            "reason": (
                "strongest offline candidate family is not exactly live-reproducible; "
                "no live A/B configuration is emitted"
            ),
            "offline_only_candidates": mappings,
        }
    selected = reproducible[0]
    candidate = selected["candidate"]
    metrics = group["candidates"][candidate]["aggregate"]
    _, label = _validation_label(run_count)
    return {
        "available": True,
        **selected,
        "metrics": metrics,
        "equivalent_candidates": list(best_group["members"]),
        "numerical_tie": bool(best_group["numerically_equivalent"]),
        "tie_breaker": best_group.get("tie_breaker"),
        "rls_lambda_reason": (
            "numerically equivalent applied gains/errors; canonical runtime lambda retained"
            if best_group["numerically_equivalent"]
            else "canonical runtime lambda is the selected live-reproducible candidate"
        ),
        "evidence_strength": label,
        "ready_for_perceptual_ab": run_count >= 2,
        "interpretation": (
            "hidden-space candidate for a manual perceptual A/B; production default remains legacy"
            if run_count >= 2
            else "development candidate only; collect another independent run before perceptual A/B"
        ),
    }


def _research_report(
    blocks: list[dict[str, Any]],
    signature: str,
    group_id: str,
) -> dict[str, Any]:
    group = analyze_group(validate_blocks(blocks))
    validation_key, validation_label = _validation_label(group["run_count"])
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "spectrum_h3_generic_correction_research_report",
        "compatibility_group_id": group_id,
        "compatibility_signature": json.loads(signature),
        "compatible_independent_runs": group["run_count"],
        "validation_level": validation_key,
        "validation_label": validation_label,
        "hidden_space_recommendation": _recommendation(group),
        "production_default": {
            "generic_correction_mode": "legacy",
            "generic_correction_attenuation": "mode_default",
            "generic_correction_limiter": "rational",
            "generic_correction_limit": 0.25,
            "promotion_status": "unchanged; perceptual validation required",
        },
        "analysis": group,
    }


def _format_metric(metrics: dict[str, Any]) -> str:
    return (
        f"error={metrics['mean_normalized_hidden_error']:.6f} "
        f"vs_legacy={metrics['relative_improvement_over_legacy']:+.2%} "
        f"wins/losses={metrics['wins_vs_legacy']}/{metrics['losses_vs_legacy']} "
        f"worst_regression={metrics['worst_regression_vs_legacy']:+.6f} "
        f"oracle_headroom_captured={metrics['oracle_headroom_captured']:+.2%}"
    )


def _stream_candidates(
    group: dict[str, Any],
    stream: str,
) -> list[tuple[str, dict[str, Any]]]:
    metric_name = "audio" if stream == "audio" else "video_global"
    entries = [
        (candidate, report[metric_name])
        for candidate, report in group["candidates"].items()
        if report[metric_name]["targets"] > 0
    ]
    if stream == "video" and group["regional_candidate_ranking"]:
        entries.extend(
            (f"regional::{candidate}", group["candidates"][candidate]["video_regional"])
            for candidate in group["regional_candidate_ranking"]
            if group["candidates"][candidate]["video_regional"]["targets"] > 0
        )
    return sorted(
        entries,
        key=lambda item: (item[1]["mean_normalized_hidden_error"], item[0]),
    )


def render_console_summary(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> str:
    group = report["analysis"]
    lines = [
        "Generic correction research",
        f"active compatibility group: {report['compatibility_group_id']}",
        f"compatible independent runs: {report['compatible_independent_runs']}",
        f"validation level: {report['validation_label']}",
    ]
    for stream, label in (("video", "VIDEO"), ("audio", "AUDIO")):
        lines.extend(("", label))
        baseline = group["baselines"]["legacy"][stream]
        if baseline["targets"]:
            lines.append(f"legacy baseline: {_format_metric(baseline)}")
        candidates = _stream_candidates(group, stream)
        if not candidates:
            lines.append("no scoreable targets")
        for candidate, metrics in candidates[:TOP_CONSOLE_CANDIDATES]:
            live = group["candidates"][candidate]["live_configuration"][
                "live_reproducible"
            ]
            lines.append(
                f"- {candidate} ({'exact live' if live else 'offline-only'}): "
                f"{_format_metric(metrics)}"
            )
    best = group["candidate_ranking"][0]
    for band, label in (
        ("audio_start", "AUDIO start boundary"),
        ("audio_middle", "AUDIO middle"),
        ("audio_end", "AUDIO end boundary"),
    ):
        metrics = group["candidates"][best][band]
        lines.extend(("", label))
        lines.append(
            _format_metric(metrics) if metrics["targets"] else "unavailable in this calibration schema"
        )
    recommendation = report["hidden_space_recommendation"]
    lines.extend(("", "Recommended live perceptual A/B candidate:"))
    if recommendation.get("available"):
        lines.extend(
            (
                f"- generic_correction_mode={recommendation['generic_correction_mode']}",
                f"- generic_correction_attenuation={recommendation['generic_correction_attenuation']}",
                f"- generic_correction_limiter={recommendation['generic_correction_limiter']}",
                f"- generic_correction_limit={recommendation['generic_correction_limit']:.2f}",
                f"- RLS lambda={recommendation['rls_lambda']:.2f}; {recommendation['rls_lambda_reason']}",
                f"- numerical equivalence group: {', '.join(recommendation['equivalent_candidates'])}",
                f"- evidence: {recommendation['evidence_strength']}; {recommendation['interpretation']}",
            )
        )
    else:
        lines.append(f"- unavailable: {recommendation['reason']}")
    lines.extend(
        (
            "- production/default promotion: unchanged (legacy)",
            f"detailed Markdown report: {markdown_path}",
            f"machine-readable JSON report: {json_path}",
        )
    )
    return "\n".join(lines)


def _markdown_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics['mean_normalized_hidden_error']:.8f} | "
        f"{metrics['relative_improvement_over_legacy']:+.4%} | "
        f"{metrics['wins_vs_legacy']} / {metrics['losses_vs_legacy']} | "
        f"{metrics['worst_regression_vs_legacy']:+.8f} | "
        f"{metrics['oracle_headroom_captured']:+.4%}"
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    group = report["analysis"]
    recommendation = report["hidden_space_recommendation"]
    lines = [
        "# Generic correction research",
        "",
        f"- Compatibility group: `{report['compatibility_group_id']}`",
        f"- Compatible independent runs: {report['compatible_independent_runs']}",
        f"- Validation level: **{report['validation_label']}**",
        "- Hidden-space results identify candidates for perceptual A/B only.",
        "- Production/default status: `legacy` remains unchanged.",
        "",
        "## Recommended live perceptual A/B candidate",
        "",
    ]
    if recommendation.get("available"):
        lines.extend(
            (
                f"- `generic_correction_mode={recommendation['generic_correction_mode']}`",
                f"- `generic_correction_attenuation={recommendation['generic_correction_attenuation']}`",
                f"- `generic_correction_limiter={recommendation['generic_correction_limiter']}`",
                f"- `generic_correction_limit={recommendation['generic_correction_limit']:.2f}`",
                f"- RLS lambda: `{recommendation['rls_lambda']:.2f}` — {recommendation['rls_lambda_reason']}",
                f"- Numerically equivalent candidates: `{', '.join(recommendation['equivalent_candidates'])}`",
                f"- Evidence: {recommendation['evidence_strength']}",
                f"- Interpretation: {recommendation['interpretation']}",
            )
        )
    else:
        lines.append(f"- Unavailable: {recommendation['reason']}")
    lines.extend(
        (
            "",
            "## Baselines",
            "",
            "| Baseline | Scope | Mean normalized error | Improvement vs legacy | Wins / losses | Worst regression | Oracle headroom captured |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for baseline, scopes in group["baselines"].items():
        for scope in (
            "aggregate",
            "video",
            "audio",
            "audio_start",
            "audio_middle",
            "audio_end",
        ):
            metrics = scopes[scope]
            if metrics["targets"]:
                lines.append(
                    f"| `{baseline}` | {scope} | {_markdown_metrics(metrics)} |"
                )
    lines.extend(
        (
            "",
            "## Complete global candidate ranking",
            "",
            "| Rank | Candidate | Live status | Tie | Aggregate error | vs legacy | wins / losses | worst regression | headroom captured | VIDEO error | AUDIO error | AUDIO start | AUDIO middle | AUDIO end |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for rank, candidate in enumerate(group["candidate_ranking"], start=1):
        metrics = group["candidates"][candidate]
        aggregate = metrics["aggregate"]
        equivalence = next(
            item
            for item in group["candidate_equivalence_groups"]
            if candidate in item["members"]
        )
        tie = (
            f"equivalent ({len(equivalence['members'])})"
            if equivalence["numerically_equivalent"]
            else "unique"
        )
        live = (
            "exact live mapping"
            if metrics["live_configuration"]["live_reproducible"]
            else "offline-only (noncanonical lambda)"
        )
        lines.append(
            f"| {rank} | `{candidate}` | {live} | {tie} | "
            f"{aggregate['mean_normalized_hidden_error']:.8f} | "
            f"{aggregate['relative_improvement_over_legacy']:+.4%} | "
            f"{aggregate['wins_vs_legacy']} / {aggregate['losses_vs_legacy']} | "
            f"{aggregate['worst_regression_vs_legacy']:+.8f} | "
            f"{aggregate['oracle_headroom_captured']:+.4%} | "
            f"{metrics['video_global']['mean_normalized_hidden_error']:.8f} | "
            f"{metrics['audio']['mean_normalized_hidden_error']:.8f} | "
            f"{metrics['audio_start']['mean_normalized_hidden_error']:.8f} | "
            f"{metrics['audio_middle']['mean_normalized_hidden_error']:.8f} | "
            f"{metrics['audio_end']['mean_normalized_hidden_error']:.8f} |"
        )
    if group["regional_candidate_ranking"]:
        lines.extend(
            (
                "",
                "## Complete regional VIDEO ranking",
                "",
                "| Rank | Candidate | Combined error | vs legacy | wins / losses | worst regression | headroom captured | Regional VIDEO error | AUDIO error |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for rank, candidate in enumerate(
            group["regional_candidate_ranking"], start=1
        ):
            metrics = group["candidates"][candidate]
            aggregate = metrics["regional_combined"]
            lines.append(
                f"| {rank} | `{candidate}` | "
                f"{aggregate['mean_normalized_hidden_error']:.8f} | "
                f"{aggregate['relative_improvement_over_legacy']:+.4%} | "
                f"{aggregate['wins_vs_legacy']} / {aggregate['losses_vs_legacy']} | "
                f"{aggregate['worst_regression_vs_legacy']:+.8f} | "
                f"{aggregate['oracle_headroom_captured']:+.4%} | "
                f"{metrics['video_regional']['mean_normalized_hidden_error']:.8f} | "
                f"{metrics['audio']['mean_normalized_hidden_error']:.8f} |"
            )
    lines.extend(
        (
            "",
            "## Whole-run validation",
            "",
            "```json",
            json.dumps(group["cross_validation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Mathematical interpretation",
            "",
            group["oracle_interpretation"],
            "",
            "## Compatibility signature",
            "",
            "```json",
            json.dumps(report["compatibility_signature"], indent=2, sort_keys=True),
            "```",
            "",
        )
    )
    return "\n".join(lines)


def _trim_reports(root: Path) -> None:
    reports = root / "reports"
    json_paths = list(reports.glob("*.json")) if reports.exists() else []
    ordered = sorted(json_paths, key=_file_order)
    for json_path in ordered[: max(0, len(ordered) - MAX_REPORT_GROUPS)]:
        markdown_path = json_path.with_suffix(".md")
        try:
            json_path.unlink()
        except FileNotFoundError:
            pass
        try:
            markdown_path.unlink()
        except FileNotFoundError:
            pass


def persist_and_analyze(
    block: dict[str, Any],
    *,
    root: Path | None = None,
) -> ResearchResult:
    """Atomically retain one run, evaluate its group, and refresh both reports."""
    started = time.perf_counter()
    validate_blocks([block])
    store_root = Path(root) if root is not None else default_store_root()
    signature = compatibility_signature(block)
    group_id = _group_id(signature)
    trace = str(block["provenance"]["trace_fingerprint"])
    identity = independence_identity(block)
    loaded = _load_stored_blocks(store_root)
    duplicate_path: Path | None = None
    for path, existing in loaded:
        if str(existing["provenance"]["trace_fingerprint"]) == trace:
            duplicate_path = path
            break
        if (
            compatibility_signature(existing) == signature
            and independence_identity(existing) == identity
        ):
            duplicate_path = path
            break

    run_path: Path | None = duplicate_path
    duplicate = duplicate_path is not None
    if not duplicate:
        run_path = store_root / "runs" / f"{trace}.json"
        _atomic_write_text(
            run_path,
            json.dumps(block, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        loaded.append((run_path, block))

    stored_groups: dict[str, list[Path]] = {}
    for path, stored in loaded:
        stored_groups.setdefault(compatibility_signature(stored), []).append(path)
    for paths in stored_groups.values():
        _trim_files([path for path in paths if path.exists()], MAX_RUNS_PER_GROUP)
    _trim_files([path for path, _ in loaded if path.exists()], MAX_STORED_RUNS)
    loaded = _load_stored_blocks(store_root)
    group_blocks: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    for _, stored in loaded:
        if compatibility_signature(stored) != signature:
            continue
        stored_identity = independence_identity(stored)
        if stored_identity in seen_identities:
            LOG.warning(
                "Spectrum H3 ignored duplicate generic-correction identity %s in group %s",
                stored_identity,
                group_id,
            )
            continue
        seen_identities.add(stored_identity)
        group_blocks.append(stored)
    group_blocks.sort(
        key=lambda item: str(item["provenance"]["trace_fingerprint"])
    )
    if not group_blocks:
        raise CalibrationError("active compatibility group has no recoverable runs")

    report = _research_report(group_blocks, signature, group_id)
    reports_directory = store_root / "reports"
    json_path = reports_directory / f"{group_id}.json"
    markdown_path = reports_directory / f"{group_id}.md"
    _atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(markdown_path, render_markdown_report(report))
    _trim_reports(store_root)
    summary = render_console_summary(report, json_path, markdown_path)
    elapsed = time.perf_counter() - started
    return ResearchResult(
        group_id=group_id,
        run_count=len(group_blocks),
        duplicate=duplicate,
        run_path=run_path,
        json_report_path=json_path,
        markdown_report_path=markdown_path,
        report=report,
        console_summary=summary,
        elapsed_seconds=elapsed,
    )


__all__ = [
    "MAX_QUARANTINED_FILES",
    "MAX_REPORT_GROUPS",
    "MAX_RUNS_PER_GROUP",
    "MAX_STORED_RUNS",
    "ResearchResult",
    "default_store_root",
    "persist_and_analyze",
    "render_console_summary",
    "render_markdown_report",
]
