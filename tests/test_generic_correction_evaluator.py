from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from comfyui_spectrum_h3.generic_correction_controller import (
    GenericCorrectionController,
)
from comfyui_spectrum_h3.generic_correction_core import ScalarMoments

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


def _row(step: int, stream: str, region: str | None = None):
    region_offset = {
        None: 0.0,
        "t0": -0.02,
        "t1": 0.03,
        "audio_start": 0.08,
        "audio_middle": -0.01,
        "audio_end": 0.12,
    }[region]
    a = 1.0 + 0.05 * step + region_offset
    b = 0.3 + 0.02 * step + region_offset
    c = 0.8 + abs(region_offset)
    row = {
        "target_step_id": step,
        "stream": stream,
        "region_id": region,
        "sample_count": 6 if stream == "audio" and region is None else 4 if region is None else 2,
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
        rows.append(_row(step, "audio"))
        rows.extend(
            _row(step, "audio", band)
            for band in ("audio_start", "audio_middle", "audio_end")
        )
        rows.append(_row(step, "video"))
        rows.append(_row(step, "video", "t0"))
        rows.append(_row(step, "video", "t1"))
    return {
        "schema_version": 1,
        "kind": "spectrum_h3_generic_correction_calibration",
        "compatible": True,
        "provenance": {
            "trace_fingerprint": trace,
            "source_schema_revision": "generic-correction-v1",
            "package_version": "0.2.8",
            "source_revision": "abc",
            "topology_fingerprint": "topology-av",
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
    second = _block("b")
    log = (
        f"noise\n{evaluator.LOG_PREFIX}   {json.dumps(block)}\n"
        f"more noise\n{evaluator.LOG_PREFIX}\t{json.dumps(second)}\n"
    )
    assert evaluator.parse_calibration_text(log) == [block, second]


def test_single_run_is_explicitly_development_only():
    report = evaluator.analyze_blocks([_block("a")])
    group = report["groups"][0]["report"]
    assert group["cross_validation"]["status"] == "development_only_non_confirmatory"
    assert group["candidate_ranking"]
    best = group["candidate_ranking"][0]
    assert group["candidates"][best]["video_regional"]["targets"] == 3
    assert group["candidates"][best]["audio_start"]["targets"] == 3
    assert group["candidates"][best]["audio_middle"]["targets"] == 3
    assert group["candidates"][best]["audio_end"]["targets"] == 3


def test_multiple_runs_use_whole_run_leave_one_out():
    preliminary = evaluator.analyze_blocks([_block("a"), _block("b")])
    assert (
        preliminary["groups"][0]["report"]["cross_validation"]["status"]
        == "whole_run_leave_one_out_preliminary"
    )
    report = evaluator.analyze_blocks([_block("a"), _block("b"), _block("c")])
    cv = report["groups"][0]["report"]["cross_validation"]
    assert cv["status"] == "whole_run_leave_one_out_generalization"
    assert len(cv["folds"]) == 3


def test_incompatible_sampler_traces_are_reported_as_separate_groups():
    report = evaluator.analyze_blocks(
        [_block("a", sampler="sample_euler"), _block("b", sampler="sample_er_sde")]
    )
    assert report["compatibility_group_count"] == 2


def test_incompatible_topologies_are_reported_as_separate_groups():
    first = _block("a")
    second = _block("b")
    second["provenance"]["topology_fingerprint"] = "topology-video-only"
    report = evaluator.analyze_blocks([first, second])
    assert report["compatibility_group_count"] == 2


def test_duplicate_trace_is_rejected(tmp_path):
    block = _block("same")
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps(block), encoding="utf-8")
    second.write_text(json.dumps(block), encoding="utf-8")
    with pytest.raises(evaluator.CalibrationError, match="duplicate"):
        evaluator.load_blocks([first, second])


def test_later_targets_never_change_any_candidate_score_at_an_earlier_target():
    block = _block("causal")
    original = evaluator._evaluate_run(block)
    mutated = json.loads(json.dumps(block))
    for row in mutated["target_rows"]:
        if row["target_step_id"] > 2:
            row["B"] *= -1.0
    changed = evaluator._evaluate_run(mutated)
    for candidate in evaluator._candidate_names():
        original_step = next(
            item for item in original["candidates"][candidate] if item["target_step_id"] == 2
        )
        changed_step = next(
            item for item in changed["candidates"][candidate] if item["target_step_id"] == 2
        )
        assert original_step == changed_step


def test_reliability_alignment_matches_runtime_threshold_and_clamp():
    state = evaluator._OnlineState(forgetting=0.9)
    clamped = _row(2, "audio")
    clamped.update({"A": 1.0, "B": 2.0, "C": 1.0, "ratio_epsilon": 1.0e-6})
    state.update(clamped, 0.0)
    assert state.alignment == pytest.approx(0.5)

    degenerate = evaluator._OnlineState(forgetting=0.9)
    row = _row(2, "audio")
    row.update({"A": 1.0, "B": 0.5, "C": 1.0, "ratio_epsilon": 2.0})
    degenerate.update(row, 0.0)
    assert degenerate.alignment == 0.0
    assert degenerate.nondegenerate == 0


def test_empty_candidate_scope_is_rejected_explicitly():
    candidate = evaluator._candidate_names()[0]
    run = {
        "candidates": {candidate: []},
        "regional_video_candidates": {candidate: []},
    }
    with pytest.raises(evaluator.CalibrationError, match="no scoreable entries"):
        evaluator._candidate_run_score(run, candidate, regional=True)


def _moments_from_row(row):
    return ScalarMoments(
        sample_count=int(row["sample_count"]),
        residual_sq_mean=float(row["A"]),
        residual_dot_direction_mean=float(row["B"]),
        direction_sq_mean=float(row["C"]),
        hold_error_sq_mean=float(row["ratio_denominator_rms"]) ** 2,
        actual_sq_mean=1.0,
        ratio_epsilon=float(row["ratio_epsilon"]),
        ratio_denominator_rms=float(row["ratio_denominator_rms"]),
    ).validate()


def test_evaluator_recommendation_mapping_matches_live_gain_step_for_step():
    block = _block("parity")
    candidate = "rls0.90__no_attenuation__hard_clip__L0.40"
    mapping = evaluator.live_configuration_for_candidate(candidate)
    assert mapping["live_reproducible"]
    controller = GenericCorrectionController(
        mapping["generic_correction_mode"],
        mapping["generic_correction_limiter"],
        mapping["generic_correction_limit"],
        attenuation=mapping["generic_correction_attenuation"],
    )
    evaluated = evaluator._evaluate_run(block)["candidates"][candidate]
    expected = {
        (item["stream"], item["target_step_id"]): item["gain"]
        for item in evaluated
    }
    rows = [
        row
        for row in evaluator._ordered_rows(block)
        if row.get("region_id") is None
    ]
    for row in rows:
        application = controller.application(
            row["stream"],
            general_confidence=float(row["general_forecast_confidence"]),
        )
        assert application.bounded_gain == pytest.approx(
            expected[(row["stream"], int(row["target_step_id"]))],
            rel=1e-13,
            abs=1e-13,
        )
        controller.observe_stream(
            row["stream"],
            _moments_from_row(row),
            application.bounded_gain,
        )


def test_saturated_rls_lambdas_are_reported_as_one_canonical_tie():
    run = evaluator._evaluate_run(_block("tied"))
    candidates = [
        f"rls{forgetting:.2f}__no_attenuation__hard_clip__L0.40"
        for forgetting in (0.75, 0.90, 0.97, 1.00)
    ]
    groups = evaluator._candidate_equivalence_groups([run], candidates)
    assert len(groups) == 1
    assert groups[0]["numerically_equivalent"]
    assert groups[0]["representative"].startswith("rls0.90__")
    assert groups[0]["members"][0].startswith("rls0.90__")
    assert evaluator.live_configuration_for_candidate(candidates[0])[
        "live_reproducible"
    ] is False
    assert evaluator.live_configuration_for_candidate(candidates[1])[
        "live_reproducible"
    ] is True


def test_cli_normalizes_malformed_numeric_input(tmp_path, capsys):
    block = _block("bad")
    block["target_rows"][0]["A"] = None
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(block), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        evaluator.main([str(path)])
    assert raised.value.code == 2
    assert "malformed numeric data" in capsys.readouterr().err


def _mutate_matching_row(block, predicate, **updates):
    index = next(
        index for index, row in enumerate(block["target_rows"]) if predicate(row)
    )
    block["target_rows"][index] = {**block["target_rows"][index], **updates}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda block: _mutate_matching_row(
                block,
                lambda row: row["target_step_id"] == 2
                and row["stream"] == "video"
                and row["region_id"] is None,
                region_id="missing-global",
            ),
            "lacks a global",
        ),
        (
            lambda block: _mutate_matching_row(
                block,
                lambda row: row["target_step_id"] == 6
                and row["stream"] == "video"
                and row["region_id"] == "t1",
                region_id="changed-region",
            ),
            "topology changed",
        ),
        (
            lambda block: _mutate_matching_row(
                block,
                lambda row: row["target_step_id"] == 2
                and row["stream"] == "video"
                and row["region_id"] == "t0",
                sample_count=1,
            ),
            "exactly cover",
        ),
    ),
)
def test_regional_candidate_rejects_incomplete_geometry(mutation, message):
    block = _block("bad-regional")
    if message == "lacks a global":
        block["target_rows"] = [
            row
            for row in block["target_rows"]
            if not (row["target_step_id"] == 2 and row["stream"] == "video" and row["region_id"] is None)
        ]
    else:
        mutation(block)
    with pytest.raises(evaluator.CalibrationError, match=message):
        evaluator._evaluate_regional_candidate(
            block,
            evaluator._candidate_names()[0],
        )
