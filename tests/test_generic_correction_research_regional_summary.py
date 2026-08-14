from __future__ import annotations

from pathlib import Path

from comfyui_spectrum_h3 import generic_correction_research as research


CANDIDATE = "rls0.90__no_attenuation__hard_clip__L0.40"


def _metric(error: float = 1.0) -> dict[str, float | int]:
    return {
        "targets": 1,
        "mean_normalized_hidden_error": error,
        "relative_improvement_over_legacy": 0.1,
        "wins_vs_legacy": 1,
        "losses_vs_legacy": 0,
        "worst_regression_vs_legacy": 0.0,
        "oracle_headroom_captured": 0.5,
    }


def _report() -> dict:
    metric = _metric()
    candidate = {
        "video_global": _metric(1.1),
        "video_regional": _metric(0.9),
        "audio": _metric(1.0),
        "audio_start": _metric(1.0),
        "audio_middle": _metric(1.0),
        "audio_end": _metric(1.0),
        "live_configuration": {
            "live_reproducible": True,
        },
    }
    return {
        "compatibility_group_id": "fixture",
        "compatible_independent_runs": 1,
        "validation_label": "development only / non-confirmatory",
        "hidden_space_recommendation": {
            "available": False,
            "reason": "fixture",
        },
        "analysis": {
            "baselines": {
                "legacy": {
                    "video": metric,
                    "audio": metric,
                }
            },
            "candidates": {CANDIDATE: candidate},
            "candidate_ranking": [CANDIDATE],
            "regional_candidate_ranking": [CANDIDATE],
        },
    }


def test_regional_stream_candidate_keeps_base_identity_and_regional_live_mapping():
    entries = research._stream_candidates(_report()["analysis"], "video")
    regional = next(entry for entry in entries if entry[0].startswith("regional::"))
    display_name, metrics, live = regional
    assert display_name == f"regional::{CANDIDATE}"
    assert metrics["mean_normalized_hidden_error"] == 0.9
    assert live["live_reproducible"] is True
    assert live["generic_correction_mode"] == "regional"
    assert live["scope"] == "regional"


def test_console_summary_handles_regional_candidate_without_synthetic_key_lookup():
    summary = research.render_console_summary(
        _report(),
        Path("report.json"),
        Path("report.md"),
    )
    assert f"regional::{CANDIDATE}" in summary
    assert "exact live" in summary
    assert "Hidden-space ranking" in summary
