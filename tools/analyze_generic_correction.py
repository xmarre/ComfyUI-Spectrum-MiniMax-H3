from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_PREFIX = "SPECTRUM_GENERIC_CORRECTION_CALIBRATION_JSON="
SCHEMA_VERSION = 1
RLS_FORGETTING = (0.75, 0.90, 0.97, 1.0)
LIMITERS = ("rational", "hard_clip", "tanh")
LIMITS = (0.15, 0.25, 0.40)
SCALINGS = (
    "no_attenuation",
    "general_confidence",
    "correction_reliability",
    "combined_conservative",
)
EPSILON = 1.0e-12


class CalibrationError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, float(value)))


def _limit_gain(value: float, limiter: str, limit: float) -> float:
    if limiter == "rational":
        return value / (1.0 + abs(value) / limit)
    if limiter == "hard_clip":
        return max(-limit, min(limit, value))
    if limiter == "tanh":
        return limit * math.tanh(value / limit)
    raise CalibrationError(f"unknown limiter {limiter!r}")


def _mse(row: dict[str, Any], gain: float, *, legacy: bool = False) -> float:
    prefix = "legacy_" if legacy else ""
    a = float(row[f"{prefix}A"])
    b = float(row[f"{prefix}B"])
    c = float(row[f"{prefix}C"])
    value = a - 2.0 * float(gain) * b + float(gain) ** 2 * c
    tolerance = 1.0e-10 * max(1.0, abs(a), abs(b), abs(c))
    if value < 0.0 and abs(value) <= tolerance:
        value = 0.0
    if value < 0.0 or not math.isfinite(value):
        raise CalibrationError("quadratic moments produced an invalid MSE")
    return value


def _ratio(row: dict[str, Any], gain: float, *, legacy: bool = False) -> float:
    denominator = float(row["ratio_denominator_rms"])
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CalibrationError("ratio denominator must be positive and finite")
    return math.sqrt(_mse(row, gain, legacy=legacy)) / denominator


def _oracle(row: dict[str, Any], *, legacy: bool = False) -> float:
    prefix = "legacy_" if legacy else ""
    b = float(row[f"{prefix}B"])
    c = float(row[f"{prefix}C"])
    epsilon = float(row["ratio_epsilon"])
    return b / c if c > max(EPSILON, epsilon * epsilon) else 0.0


def _extract_log_blocks(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    blocks: list[dict[str, Any]] = []
    cursor = 0
    while True:
        marker = text.find(LOG_PREFIX, cursor)
        if marker < 0:
            break
        start = marker + len(LOG_PREFIX)
        try:
            value, consumed = decoder.raw_decode(text[start:].lstrip())
        except json.JSONDecodeError as exc:
            raise CalibrationError(
                f"malformed calibration JSON after marker at byte {marker}"
            ) from exc
        if isinstance(value, dict):
            blocks.append(value)
        cursor = start + consumed
    return blocks


def parse_calibration_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        return _extract_log_blocks(text)
    if isinstance(direct, dict):
        if direct.get("kind") == "spectrum_h3_generic_correction_calibration":
            return [direct]
        blocks = direct.get("calibration_blocks")
        if isinstance(blocks, list):
            return [item for item in blocks if isinstance(item, dict)]
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    return []


def _validate_block(block: dict[str, Any]) -> None:
    if block.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError(
            f"unsupported generic calibration schema {block.get('schema_version')!r}"
        )
    if block.get("kind") != "spectrum_h3_generic_correction_calibration":
        raise CalibrationError("input is not a Spectrum generic correction block")
    if not block.get("compatible"):
        raise CalibrationError("runtime marked generic calibration block incompatible")
    rows = block.get("target_rows")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("generic calibration block has no target rows")
    provenance = block.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("trace_fingerprint"):
        raise CalibrationError("generic calibration block lacks a trace fingerprint")
    required = {
        "target_step_id",
        "stream",
        "sample_count",
        "A",
        "B",
        "C",
        "legacy_A",
        "legacy_B",
        "legacy_C",
        "ratio_denominator_rms",
        "ratio_epsilon",
        "bounded_legacy_gain",
        "general_forecast_confidence",
    }
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise CalibrationError("generic calibration row is malformed")
        numeric = (
            "A",
            "B",
            "C",
            "legacy_A",
            "legacy_B",
            "legacy_C",
            "ratio_denominator_rms",
            "ratio_epsilon",
            "bounded_legacy_gain",
            "general_forecast_confidence",
        )
        if not all(math.isfinite(float(row[name])) for name in numeric):
            raise CalibrationError("generic calibration row contains nonfinite data")
        if int(row["sample_count"]) <= 0:
            raise CalibrationError("generic calibration row has no samples")
        if row["stream"] not in {"audio", "video"}:
            raise CalibrationError("generic calibration row has an unknown stream")
        _ratio(row, 0.0)
        _ratio(row, float(row["bounded_legacy_gain"]), legacy=True)


def _compatibility_key(block: dict[str, Any]) -> str:
    config = dict(block.get("config") or {})
    for name in (
        "debug",
        "generic_correction_mode",
        "generic_correction_limiter",
        "generic_correction_limit",
    ):
        config.pop(name, None)
    provenance = block["provenance"]
    metadata = block.get("metadata") or {}
    return _canonical_json(
        {
            "schema": provenance.get("source_schema_revision"),
            "package": provenance.get("package_version"),
            "source_revision": provenance.get("source_revision"),
            "sampler": metadata.get("sampler"),
            "steps": metadata.get("steps"),
            "base_config": config,
        }
    )


def load_blocks(paths: Iterable[Path]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for path in paths:
        parsed = parse_calibration_text(path.read_text(encoding="utf-8"))
        if not parsed:
            raise CalibrationError(f"no generic calibration block found in {path}")
        for block in parsed:
            _validate_block(block)
            fingerprint = str(block["provenance"]["trace_fingerprint"])
            if fingerprint in fingerprints:
                raise CalibrationError(
                    "duplicate calibration run identity; refusing to count one trace twice"
                )
            fingerprints.add(fingerprint)
            blocks.append(block)
    return blocks


@dataclass(slots=True)
class _OnlineState:
    forgetting: float
    b_acc: float = 0.0
    c_acc: float = 0.0
    observations: int = 0
    effective_age: float = 0.0
    alignment: float = 0.0
    sign_stability: float = 0.0
    advantage: float = 0.0
    previous_oracle: float | None = None
    nondegenerate: int = 0

    @property
    def gain(self) -> float:
        return self.b_acc / self.c_acc if self.c_acc > EPSILON else 0.0

    @property
    def support(self) -> float:
        return _clamp(self.effective_age / 3.0)

    @property
    def reliability(self) -> float:
        if not self.observations:
            return 0.0
        support = _clamp(self.nondegenerate / 3.0)
        success = _clamp(0.5 + 2.0 * self.advantage)
        return _clamp(
            0.30 * self.alignment
            + 0.25 * self.sign_stability
            + 0.25 * success
            + 0.20 * support
        )

    def update(self, row: dict[str, Any], predicted_gain: float) -> None:
        alpha = 0.5 if self.observations < 2 else 0.3
        a, b, c = (float(row[name]) for name in ("A", "B", "C"))
        denominator = math.sqrt(max(0.0, a) * max(0.0, c))
        alignment = abs(b / denominator) if denominator > EPSILON else 0.0
        oracle = _oracle(row)
        if (
            self.previous_oracle is None
            or min(abs(oracle), abs(self.previous_oracle)) <= 1.0e-9
        ):
            sign = 0.5
        else:
            sign = 1.0 if oracle * self.previous_oracle > 0.0 else 0.0
        advantage = (_mse(row, 0.0) - _mse(row, predicted_gain)) / max(
            _mse(row, 0.0),
            float(row["ratio_epsilon"]) ** 2,
        )
        self.alignment = (1.0 - alpha) * self.alignment + alpha * alignment
        self.sign_stability = (1.0 - alpha) * self.sign_stability + alpha * sign
        self.advantage = (1.0 - alpha) * self.advantage + alpha * _clamp(
            advantage, -1.0, 1.0
        )
        self.previous_oracle = oracle
        if c > max(EPSILON, float(row["ratio_epsilon"]) ** 2):
            self.nondegenerate += 1
        self.b_acc = self.forgetting * self.b_acc + b
        self.c_acc = self.forgetting * self.c_acc + c
        self.effective_age = self.forgetting * self.effective_age + 1.0
        self.observations += 1


def _candidate_names() -> list[str]:
    return [
        f"rls{forgetting:.2f}__{scaling}__{limiter}__L{limit:.2f}"
        for forgetting in RLS_FORGETTING
        for scaling in SCALINGS
        for limiter in LIMITERS
        for limit in LIMITS
    ]


def _candidate_spec(name: str) -> tuple[float, str, str, float]:
    rls, scaling, limiter, limit = name.split("__")
    return float(rls[3:]), scaling, limiter, float(limit[1:])


def _scaled_gain(
    state: _OnlineState,
    row: dict[str, Any],
    scaling: str,
    limiter: str,
    limit: float,
    raw_override: float | None = None,
) -> float:
    confidence = _clamp(float(row["general_forecast_confidence"]))
    reliability = state.reliability
    scale = {
        "no_attenuation": 1.0,
        "general_confidence": confidence,
        "correction_reliability": reliability,
        "combined_conservative": confidence * reliability,
    }[scaling]
    raw = state.gain if raw_override is None else float(raw_override)
    return _limit_gain(raw * scale, limiter, limit)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["stream"]), str(row.get("region_id") or "global")


def _ordered_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        block["target_rows"],
        key=lambda row: (
            int(row["target_step_id"]),
            str(row["stream"]),
            str(row.get("region_id") or ""),
        ),
    )


def _evaluate_regional_candidate(
    block: dict[str, Any],
    candidate: str,
) -> list[dict[str, Any]]:
    forgetting, scaling, limiter, limit = _candidate_spec(candidate)
    global_rows = {
        int(row["target_step_id"]): row
        for row in block["target_rows"]
        if row["stream"] == "video" and row.get("region_id") is None
    }
    region_groups: dict[int, list[dict[str, Any]]] = {}
    for row in block["target_rows"]:
        if row["stream"] == "video" and row.get("region_id") is not None:
            region_groups.setdefault(int(row["target_step_id"]), []).append(row)
    global_state = _OnlineState(forgetting=forgetting)
    region_states: dict[str, _OnlineState] = {}
    results: list[dict[str, Any]] = []
    for step_id in sorted(region_groups):
        global_row = global_rows.get(step_id)
        if global_row is None:
            raise CalibrationError("regional VIDEO target lacks a global scoring row")
        regions = sorted(region_groups[step_id], key=lambda row: str(row["region_id"]))
        ids = tuple(str(row["region_id"]) for row in regions)
        if region_states and set(region_states) != set(ids):
            raise CalibrationError("regional topology changed within one run")
        for region_id in ids:
            region_states.setdefault(
                region_id,
                _OnlineState(forgetting=forgetting),
            )
        global_raw = global_state.gain
        shrunk = [
            global_raw
            + region_states[region_id].support
            * (region_states[region_id].gain - global_raw)
            for region_id in ids
        ]
        regularized: list[float] = []
        for index, value in enumerate(shrunk):
            neighbors = [value]
            if index > 0:
                neighbors.append(shrunk[index - 1])
            if index + 1 < len(shrunk):
                neighbors.append(shrunk[index + 1])
            regularized.append(0.75 * value + 0.25 * statistics.fmean(neighbors))

        parts: list[tuple[int, float]] = []
        predicted: list[float] = []
        for row, region_id, raw in zip(regions, ids, regularized, strict=True):
            state = region_states[region_id]
            gain = _scaled_gain(
                state,
                row,
                scaling,
                limiter,
                limit,
                raw_override=raw,
            )
            predicted.append(gain)
            parts.append((int(row["sample_count"]), _mse(row, gain)))
        total = sum(count for count, _ in parts)
        if total <= 0 or total != int(global_row["sample_count"]):
            raise CalibrationError("regional rows do not exactly cover global VIDEO")
        aggregate_mse = sum(count * value for count, value in parts) / total
        results.append(
            {
                "stream": "video",
                "target_step_id": step_id,
                "ratio": math.sqrt(max(0.0, aggregate_mse))
                / float(global_row["ratio_denominator_rms"]),
                "uncorrected": _ratio(global_row, 0.0),
                "legacy": _ratio(
                    global_row,
                    float(global_row["bounded_legacy_gain"]),
                    legacy=True,
                ),
                "oracle": _ratio(global_row, _oracle(global_row)),
            }
        )
        for row, region_id, gain in zip(regions, ids, predicted, strict=True):
            region_states[region_id].update(row, gain)
        global_gain = _scaled_gain(
            global_state,
            global_row,
            scaling,
            limiter,
            limit,
        )
        global_state.update(global_row, global_gain)
    return results


def _evaluate_run(block: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_names()
    specs = {name: _candidate_spec(name) for name in candidates}
    states: dict[tuple[str, tuple[str, str]], _OnlineState] = {}
    scored: dict[str, list[dict[str, Any]]] = {name: [] for name in candidates}
    baselines: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in ("uncorrected", "legacy", "oracle_legacy", "oracle_coordinate")
    }

    for row in _ordered_rows(block):
        is_global = row.get("region_id") is None
        if is_global:
            base_entry = {
                "stream": row["stream"],
                "target_step_id": int(row["target_step_id"]),
                "uncorrected": _ratio(row, 0.0),
                "legacy": _ratio(
                    row,
                    float(row["bounded_legacy_gain"]),
                    legacy=True,
                ),
                "oracle": _ratio(row, _oracle(row)),
            }
            baselines["uncorrected"].append(
                {**base_entry, "ratio": base_entry["uncorrected"]}
            )
            baselines["legacy"].append({**base_entry, "ratio": base_entry["legacy"]})
            baselines["oracle_legacy"].append(
                {
                    **base_entry,
                    "ratio": _ratio(row, _oracle(row, legacy=True), legacy=True),
                }
            )
            baselines["oracle_coordinate"].append(
                {**base_entry, "ratio": base_entry["oracle"]}
            )

        for candidate in candidates:
            if not is_global:
                continue
            forgetting, scaling, limiter, limit = specs[candidate]
            key = (candidate, _row_key(row))
            state = states.setdefault(key, _OnlineState(forgetting=forgetting))
            gain = _scaled_gain(state, row, scaling, limiter, limit)
            ratio = _ratio(row, gain)
            if is_global:
                scored[candidate].append(
                    {
                        "stream": row["stream"],
                        "target_step_id": int(row["target_step_id"]),
                        "ratio": ratio,
                        "uncorrected": _ratio(row, 0.0),
                        "legacy": _ratio(
                            row,
                            float(row["bounded_legacy_gain"]),
                            legacy=True,
                        ),
                        "oracle": _ratio(row, _oracle(row)),
                    }
                )
            state.update(row, gain)

    regional_scores = {
        candidate: _evaluate_regional_candidate(block, candidate)
        for candidate in candidates
    }

    return {
        "trace_fingerprint": block["provenance"]["trace_fingerprint"],
        "baselines": baselines,
        "candidates": scored,
        "regional_video_candidates": regional_scores,
    }


def _metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "targets": 0,
            "mean_normalized_hidden_error": 0.0,
            "relative_improvement_over_uncorrected": 0.0,
            "relative_improvement_over_legacy": 0.0,
            "oracle_headroom": 0.0,
            "oracle_headroom_captured": 0.0,
            "wins_vs_legacy": 0,
            "losses_vs_legacy": 0,
            "worst_regression_vs_legacy": 0.0,
        }
    ratio = statistics.fmean(float(item["ratio"]) for item in entries)
    uncorrected = statistics.fmean(float(item["uncorrected"]) for item in entries)
    legacy = statistics.fmean(float(item["legacy"]) for item in entries)
    oracle = statistics.fmean(float(item["oracle"]) for item in entries)
    headroom = max(0.0, uncorrected - oracle)
    return {
        "targets": len(entries),
        "mean_normalized_hidden_error": ratio,
        "relative_improvement_over_uncorrected": (
            (uncorrected - ratio) / uncorrected if uncorrected > 0.0 else 0.0
        ),
        "relative_improvement_over_legacy": (
            (legacy - ratio) / legacy if legacy > 0.0 else 0.0
        ),
        "oracle_headroom": headroom,
        "oracle_headroom_captured": (
            (uncorrected - ratio) / headroom if headroom > 0.0 else 0.0
        ),
        "wins_vs_legacy": sum(
            float(item["ratio"]) < float(item["legacy"]) for item in entries
        ),
        "losses_vs_legacy": sum(
            float(item["ratio"]) > float(item["legacy"]) for item in entries
        ),
        "worst_regression_vs_legacy": max(
            float(item["ratio"]) - float(item["legacy"]) for item in entries
        ),
    }


def _candidate_entries(
    run: dict[str, Any],
    candidate: str,
    *,
    regional: bool,
) -> list[dict[str, Any]]:
    entries = run["candidates"][candidate]
    if not regional:
        return entries
    return [item for item in entries if item["stream"] == "audio"] + list(
        run["regional_video_candidates"][candidate]
    )


def _candidate_run_score(
    run: dict[str, Any],
    candidate: str,
    *,
    regional: bool,
) -> float:
    entries = _candidate_entries(run, candidate, regional=regional)
    return statistics.fmean(float(item["ratio"]) for item in entries)


def _cross_validation(
    runs: list[dict[str, Any]],
    *,
    regional: bool,
) -> dict[str, Any]:
    candidates = _candidate_names()
    if len(runs) == 1:
        best = min(
            candidates,
            key=lambda name: _candidate_run_score(
                runs[0],
                name,
                regional=regional,
            ),
        )
        return {
            "status": "development_only_non_confirmatory",
            "selected_candidate": best,
            "development_metrics": _metrics(
                _candidate_entries(runs[0], best, regional=regional)
            ),
        }
    folds = []
    for held_out in runs:
        training = [run for run in runs if run is not held_out]
        selected = min(
            candidates,
            key=lambda name: statistics.fmean(
                _candidate_run_score(run, name, regional=regional) for run in training
            ),
        )
        folds.append(
            {
                "held_out_run": held_out["trace_fingerprint"],
                "selected_candidate": selected,
                "metrics": _metrics(
                    _candidate_entries(
                        held_out,
                        selected,
                        regional=regional,
                    )
                ),
            }
        )
    held_out_entries = []
    for fold, run in zip(folds, runs, strict=True):
        held_out_entries.extend(
            _candidate_entries(
                run,
                fold["selected_candidate"],
                regional=regional,
            )
        )
    return {
        "status": (
            "whole_run_leave_one_out_preliminary"
            if len(runs) == 2
            else "whole_run_leave_one_out_generalization"
        ),
        "folds": folds,
        "aggregate_held_out": _metrics(held_out_entries),
    }


def analyze_group(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [_evaluate_run(block) for block in blocks]
    aggregate_baselines: dict[str, Any] = {}
    for baseline in runs[0]["baselines"]:
        entries = [item for run in runs for item in run["baselines"][baseline]]
        if baseline == "uncorrected":
            for item in entries:
                item["ratio"] = item["uncorrected"]
        elif baseline == "legacy":
            for item in entries:
                item["ratio"] = item["legacy"]
        aggregate_baselines[baseline] = {
            "aggregate": _metrics(entries),
            "audio": _metrics([item for item in entries if item["stream"] == "audio"]),
            "video": _metrics([item for item in entries if item["stream"] == "video"]),
        }

    candidate_reports: dict[str, Any] = {}
    for candidate in _candidate_names():
        entries = [item for run in runs for item in run["candidates"][candidate]]
        regional = [
            item for run in runs for item in run["regional_video_candidates"][candidate]
        ]
        regional_combined = [
            item for item in entries if item["stream"] == "audio"
        ] + regional
        candidate_reports[candidate] = {
            "aggregate": _metrics(entries),
            "audio": _metrics([item for item in entries if item["stream"] == "audio"]),
            "video_global": _metrics(
                [item for item in entries if item["stream"] == "video"]
            ),
            "video_regional": _metrics(regional),
            "regional_combined": _metrics(regional_combined),
        }
    ranked = sorted(
        _candidate_names(),
        key=lambda name: candidate_reports[name]["aggregate"][
            "mean_normalized_hidden_error"
        ],
    )
    regional_available = any(
        run["regional_video_candidates"][_candidate_names()[0]] for run in runs
    )
    regional_ranked = (
        sorted(
            _candidate_names(),
            key=lambda name: candidate_reports[name]["regional_combined"][
                "mean_normalized_hidden_error"
            ],
        )
        if regional_available
        else []
    )
    return {
        "run_count": len(runs),
        "cross_validation": _cross_validation(runs, regional=False),
        "regional_cross_validation": (
            _cross_validation(runs, regional=True)
            if regional_available
            else {"status": "unavailable_without_regional_topology"}
        ),
        "baselines": aggregate_baselines,
        "candidate_ranking": ranked,
        "regional_candidate_ranking": regional_ranked,
        "candidates": candidate_reports,
    }


def analyze_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        groups.setdefault(_compatibility_key(block), []).append(block)
    reports = [
        {
            "compatibility_signature": key,
            "report": analyze_group(group),
        }
        for key, group in sorted(groups.items())
    ]
    return {
        "schema_version": 1,
        "kind": "spectrum_h3_generic_correction_analysis",
        "input_run_count": len(blocks),
        "compatibility_group_count": len(reports),
        "groups": reports,
    }


def _human_summary(report: dict[str, Any]) -> str:
    lines = [
        f"runs={report['input_run_count']} compatibility_groups={report['compatibility_group_count']}"
    ]
    for index, item in enumerate(report["groups"], start=1):
        group = item["report"]
        best = group["candidate_ranking"][0]
        metrics = group["candidates"][best]["aggregate"]
        lines.append(
            f"group={index} runs={group['run_count']} cv={group['cross_validation']['status']} "
            f"best_fixed={best} ratio={metrics['mean_normalized_hidden_error']:.6f} "
            f"delta_vs_legacy={metrics['relative_improvement_over_legacy']:+.2%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze exact Spectrum MiniMax-H3 generic-correction calibration "
            "without ComfyUI, model loading, GPU use, or row-randomized CV."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit the complete JSON report"
    )
    args = parser.parse_args(argv)
    report = analyze_blocks(load_blocks(args.inputs))
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else _human_summary(report)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
