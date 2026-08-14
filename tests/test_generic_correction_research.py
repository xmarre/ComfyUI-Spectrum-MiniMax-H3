from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import generic_correction_evaluator as evaluator
from comfyui_spectrum_h3 import generic_correction_research as research


def _row(step: int, stream: str, region: str | None = None) -> dict[str, object]:
    region_offset = {None: 0.0, "t0": -0.02, "t1": 0.03}[region]
    a = 1.0 + 0.05 * step + region_offset
    b = 0.3 + 0.02 * step + region_offset
    c = 0.8 + abs(region_offset)
    return {
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


def _block(
    label: str,
    *,
    seed: int | None = None,
    sampler: str = "sample_euler",
    schedule: str = "schedule-a",
    topology: str = "topology-av",
    source: str = "source-a",
    package: str = "0.2.8",
    degree: int = 1,
    mode: str = "legacy",
    attenuation: str = "mode_default",
) -> dict[str, object]:
    trace = f"trace-{label}"
    rows: list[dict[str, object]] = []
    for step in (2, 4, 6):
        rows.extend(
            (
                _row(step, "audio"),
                _row(step, "video"),
                _row(step, "video", "t0"),
                _row(step, "video", "t1"),
            )
        )
    return {
        "schema_version": 1,
        "kind": "spectrum_h3_generic_correction_calibration",
        "compatible": True,
        "provenance": {
            "trace_fingerprint": trace,
            "seed": seed,
            "source_schema_revision": "generic-correction-v1",
            "package_version": package,
            "source_revision": source,
            "schedule_fingerprint": schedule,
            "topology_fingerprint": topology,
        },
        "metadata": {"sampler": sampler, "steps": 8},
        "config": {
            "degree": degree,
            "generic_correction_mode": mode,
            "generic_correction_attenuation": attenuation,
            "generic_correction_limiter": "rational",
            "generic_correction_limit": 0.25,
            "debug": True,
            "model_aware_mode": "full",
            "offline_smoothing_replay": False,
        },
        "target_rows": rows,
    }


def test_default_store_uses_comfyui_internal_user_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_system_user_directory=lambda name: tmp_path / f"__{name}"),
    )
    assert research.default_store_root() == (
        tmp_path / "__cache" / "spectrum_h3" / "generic_correction" / "v1"
    )


@pytest.mark.parametrize(
    ("count", "validation"),
    (
        (1, "development only / non-confirmatory"),
        (2, "preliminary whole-run leave-one-out"),
        (3, "whole-run leave-one-out generalization"),
    ),
)
def test_automatic_progression_reports_and_console(tmp_path, count, validation):
    result = None
    for index in range(count):
        result = research.persist_and_analyze(
            _block(str(index), seed=100 + index),
            root=tmp_path,
        )
    assert result is not None
    assert result.run_count == count
    assert result.report["validation_label"] == validation
    assert result.json_report_path.is_file()
    assert result.markdown_report_path.is_file()
    assert "Generic correction research" in result.console_summary
    assert "VIDEO" in result.console_summary
    assert "AUDIO" in result.console_summary
    assert "generic_correction_mode=" in result.console_summary
    assert "generic_correction_attenuation=" in result.console_summary
    assert "Runtime status: coordinate_rls + no_attenuation + hard_clip + 0.40" in result.console_summary
    assert "Legacy reproduction: legacy + mode_default + rational + 0.25" in result.console_summary
    machine = json.loads(result.json_report_path.read_text(encoding="utf-8"))
    assert len(machine["analysis"]["candidate_ranking"]) == 144
    assert machine["production_default"]["generic_correction_mode"] == "coordinate_rls"
    assert machine["production_default"]["generic_correction_attenuation"] == "no_attenuation"
    assert machine["production_default"]["generic_correction_limiter"] == "hard_clip"
    assert machine["production_default"]["generic_correction_limit"] == 0.40
    assert machine["production_default"]["hidden_space_alone_triggered_promotion"] is False
    assert machine["legacy_reproduction"]["generic_correction_mode"] == "legacy"
    markdown = result.markdown_report_path.read_text(encoding="utf-8")
    assert "Complete global candidate ranking" in markdown
    assert "Hidden-space results rank scalar reconstruction candidates" in markdown
    assert "Promotion used separate three-seed decoded-media" in markdown


def test_duplicate_trace_and_seed_identity_do_not_add_evidence(tmp_path):
    first = research.persist_and_analyze(_block("a", seed=7), root=tmp_path)
    exact = research.persist_and_analyze(_block("a", seed=7), root=tmp_path)
    same_seed = research.persist_and_analyze(_block("b", seed=7), root=tmp_path)
    assert first.run_count == exact.run_count == same_seed.run_count == 1
    assert exact.duplicate
    assert same_seed.duplicate
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sampler", "sample_er_sde"),
        ("schedule", "schedule-b"),
        ("topology", "topology-video"),
        ("source", "source-b"),
        ("package", "0.2.9"),
        ("degree", 2),
    ),
)
def test_incompatible_evidence_isolated_into_another_group(tmp_path, field, value):
    first = research.persist_and_analyze(_block("a", seed=1), root=tmp_path)
    second = research.persist_and_analyze(
        _block("b", seed=2, **{field: value}),
        root=tmp_path,
    )
    assert first.group_id != second.group_id
    assert second.run_count == 1
    assert len(list((tmp_path / "reports").glob("*.json"))) == 2


def test_execution_changing_correction_mode_isolated_into_another_group(tmp_path):
    first = research.persist_and_analyze(
        _block("a", seed=1, mode="legacy"), root=tmp_path
    )
    second = research.persist_and_analyze(
        _block("b", seed=2, mode="coordinate_rls"), root=tmp_path
    )
    assert first.group_id != second.group_id
    assert second.run_count == 1


def test_execution_changing_attenuation_isolated_into_another_group(tmp_path):
    first = research.persist_and_analyze(
        _block("a", seed=1, mode="coordinate_rls", attenuation="mode_default"),
        root=tmp_path,
    )
    second = research.persist_and_analyze(
        _block(
            "b",
            seed=2,
            mode="coordinate_rls",
            attenuation="no_attenuation",
        ),
        root=tmp_path,
    )
    assert first.group_id != second.group_id
    assert second.run_count == 1


def test_corruption_is_quarantined_and_does_not_poison_analysis(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    (runs / "broken.json").write_text('{"not":', encoding="utf-8")
    result = research.persist_and_analyze(_block("good", seed=1), root=tmp_path)
    assert result.run_count == 1
    assert not (runs / "broken.json").exists()
    assert len(list((tmp_path / "corrupt").glob("broken.*.json"))) == 1


def test_retention_is_bounded_and_keeps_latest_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "MAX_STORED_RUNS", 2)
    for index in range(3):
        research.persist_and_analyze(
            _block(str(index), seed=index, schedule=f"schedule-{index}"),
            root=tmp_path,
        )
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2


def test_each_compatibility_group_has_its_own_run_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "MAX_RUNS_PER_GROUP", 2)
    for index in range(3):
        research.persist_and_analyze(
            _block(str(index), seed=index),
            root=tmp_path,
        )
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2


def test_report_group_retention_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "MAX_REPORT_GROUPS", 2)
    for index in range(3):
        research.persist_and_analyze(
            _block(str(index), seed=index, schedule=f"schedule-{index}"),
            root=tmp_path,
        )
    assert len(list((tmp_path / "reports").glob("*.json"))) == 2
    assert len(list((tmp_path / "reports").glob("*.md"))) == 2


def test_atomic_failure_leaves_no_partial_run(tmp_path, monkeypatch):
    def fail_replace(_source, _target):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(research.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        research.persist_and_analyze(_block("a", seed=1), root=tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list((tmp_path / "runs").glob("*.json"))


def test_duplicate_rerun_is_deterministic(tmp_path):
    block = _block("a", seed=1)
    first = research.persist_and_analyze(block, root=tmp_path)
    before = first.json_report_path.read_bytes()
    second = research.persist_and_analyze(block, root=tmp_path)
    assert second.json_report_path.read_bytes() == before
    assert second.report == first.report


def test_research_reporting_never_mutates_live_configuration_input(tmp_path):
    block = _block("immutable", seed=23)
    original = copy.deepcopy(block)

    result = research.persist_and_analyze(block, root=tmp_path)

    assert block == original
    assert block["config"]["generic_correction_mode"] == "legacy"
    assert result.report["production_default"]["generic_correction_mode"] == (
        "coordinate_rls"
    )


def test_cli_and_runtime_import_the_same_evaluator_implementation():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "analyze_generic_correction.py"
    spec = importlib.util.spec_from_file_location("generic_tool_parity", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    blocks = [_block("a", seed=1), _block("b", seed=2)]
    assert tool.analyze_blocks(blocks) == evaluator.analyze_blocks(blocks)
    assert tool.analyze_blocks.__code__.co_filename == evaluator.analyze_blocks.__code__.co_filename


def test_evaluator_rejects_repeated_seed_inside_one_group():
    with pytest.raises(evaluator.CalibrationError, match="seed/run identity"):
        evaluator.analyze_blocks([_block("a", seed=5), _block("b", seed=5)])


def test_recommendation_serializes_exact_attenuation_and_canonical_tie():
    tied = [
        "rls0.75__no_attenuation__hard_clip__L0.40",
        "rls0.90__no_attenuation__hard_clip__L0.40",
        "rls0.97__no_attenuation__hard_clip__L0.40",
        "rls1.00__no_attenuation__hard_clip__L0.40",
    ]
    metrics = {
        "targets": 6,
        "mean_normalized_hidden_error": 0.5,
    }
    recommendation = research._recommendation(
        {
            "run_count": 3,
            "candidate_equivalence_groups": [
                {
                    "representative": tied[1],
                    "members": [tied[1], tied[0], tied[2], tied[3]],
                    "numerically_equivalent": True,
                    "tie_breaker": "canonical runtime RLS lambda 0.90, then candidate name",
                }
            ],
            "candidates": {name: {"aggregate": metrics} for name in tied},
        }
    )
    assert recommendation["available"]
    assert recommendation["candidate"] == tied[1]
    assert recommendation["generic_correction_mode"] == "coordinate_rls"
    assert recommendation["generic_correction_attenuation"] == "no_attenuation"
    assert recommendation["generic_correction_limiter"] == "hard_clip"
    assert recommendation["generic_correction_limit"] == 0.4
    assert recommendation["rls_lambda"] == 0.9
    assert recommendation["numerical_tie"]
