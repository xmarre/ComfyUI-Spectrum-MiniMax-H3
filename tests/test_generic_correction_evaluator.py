from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "analyze_generic_correction.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "spectrum_generic_correction_tool",
    _TOOL_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)


def _row(step: int, stream: str, trace: str, region: str | None = None):
    a = 1.0 + 0.05 * step
    b = 0.3 + 0.02 * step
    c = 0.8
    row = {
        "schema_version": 1,
        "trace_fingerprint": trace,
        "target_step_id": step,
        "stream": stream,
        "region_id": region,
        "sample_count": 4 if region is None else 2,
        "A": a,
        "B": b,
        "C": c,
        "legacy_A": a,
        "legacy_B": b,
        "legacy_C": c,
        "ratio_denominator_rms": 1.0,
        "ratio_epsilon": 1.0e-6,
        "bounded_legacy_gain": 0.1,
        "general_forecast_confidence": 0.8,
    }
    return row


def _block(label: str, *, sampler: str = "sample_euler"):
    trace = f"trace-{label}"
    rows = []
    for step in (2, 4, 6):
        rows.append(_row(step, "audio", trace))
        rows.append(_row(step, "video", trace))
        rows.append(_row(step, "video", trace, "t0"))
        rows.append(_row(step, "video", trace, "t1"))
    return {
        "schema_version": 1,
        "kind": "spectrum_h3_generic_correction_calibration",
        "compatible": True,
        "provenance": {
            "trace_fingerprint": trace,
            "source_schema_revision": "generic-correction-v1",
            "package_version": "0.2.8",
            "source_revision": "abc",
        },
        "metadata": {"sampler": sampler, "steps": 8},
        "config": {
            "degree": 1,
            "generic_correction_mode": "legacy",
            "generic_correction_limiter": "rational",
            "generic_correction_limit": 0.25,
            "debug": True,
        },
        "target_rows": rows,
    }


def test_parser_accepts_raw_json_and_complete_log_markers():
    block = _block("a")
    assert evaluator.parse_calibration_text(json.dumps(block)) == [block]
    log = f"noise\n{evaluator.LOG_PREFIX}{json.dumps(block)}\nmore noise"
    assert evaluator.parse_calibration_text(log) == [block]


def test_single_run_is_explicitly_development_only():
    report = evaluator.analyze_blocks([_block("a")])
    group = report["groups"][0]["report"]
    assert group["cross_validation"]["status"] == "development_only_non_confirmatory"
    assert group["candidate_ranking"]
    best = group["candidate_ranking"][0]
    assert group["candidates"][best]["video_regional"]["targets"] == 3


def test_multiple_runs_use_whole_run_leave_one_out():
    report = evaluator.analyze_blocks([_block("a"), _block("b"), _block("c")])
    cv = report["groups"][0]["report"]["cross_validation"]
    assert cv["status"] == "whole_run_leave_one_out_generalization"
    assert len(cv["folds"]) == 3


def test_incompatible_sampler_traces_are_reported_as_separate_groups():
    report = evaluator.analyze_blocks(
        [_block("a", sampler="sample_euler"), _block("b", sampler="sample_er_sde")]
    )
    assert report["compatibility_group_count"] == 2


def test_duplicate_trace_is_rejected(tmp_path):
    block = _block("same")
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps(block), encoding="utf-8")
    second.write_text(json.dumps(block), encoding="utf-8")
    with pytest.raises(evaluator.CalibrationError, match="duplicate"):
        evaluator.load_blocks([first, second])


def test_post_target_row_order_never_changes_earlier_candidate_score():
    block = _block("causal")
    original = evaluator._evaluate_run(block)
    mutated = json.loads(json.dumps(block))
    for row in mutated["target_rows"]:
        if row["target_step_id"] > 2:
            row["B"] *= -1.0
    changed = evaluator._evaluate_run(mutated)
    candidate = evaluator._candidate_names()[0]
    first_original = original["candidates"][candidate][0]
    first_changed = changed["candidates"][candidate][0]
    assert first_original["ratio"] == first_changed["ratio"]
