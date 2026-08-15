from __future__ import annotations

import copy

import torch

from comfyui_spectrum_h3.objective_media_bounded import (
    PROFILE_NAME,
    evaluate_objective_media_bounded,
)


def _provenance():
    return {
        "compatibility": {
            "model": "MiniMax-H3",
            "model_weights": "fixture",
            "precision": "bf16",
            "sampler": "er_sde",
            "scheduler": "fixture",
            "steps": 20,
            "conditioning": "fixture",
            "video_vae": "fixture",
            "audio_decoder": "fixture",
            "generation_settings": {
                "objective_metric_profile": PROFILE_NAME,
            },
        },
        "R": {"spectrum": "bypassed"},
        "A": {"generic_correction_mode": "legacy"},
        "B": {"generic_correction_mode": "coordinate_rls"},
    }


def _video(seed: int, frames: int = 6, height: int = 24, width: int = 32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((frames, height, width, 3), generator=generator)


def test_bounded_evaluator_produces_compatible_report_and_prefers_closer_candidate():
    reference = _video(1)
    noise = _video(2) - 0.5
    legacy = (reference + noise * 0.08).clamp(0.0, 1.0)
    candidate = (reference + noise * 0.02).clamp(0.0, 1.0)
    source_metadata = {
        "frame_count": 6,
        "height": 96,
        "width": 128,
        "channels": 3,
    }

    report = evaluate_objective_media_bounded(
        reference,
        legacy,
        candidate,
        fps=24.0,
        benchmark_id="bounded-fixture",
        seed=123,
        provenance=_provenance(),
        source_video_metadata=source_metadata,
        chunk_size=2,
    )

    assert report["kind"] == "spectrum_h3_objective_media_comparison"
    assert report["evaluator_profile"]["name"] == PROFILE_NAME
    assert report["video"]["metadata"]["source_width"] == 128
    assert report["video"]["metadata"]["source_height"] == 96
    assert report["compatibility"]["metric_profile"] == PROFILE_NAME
    assert report["boundaries"]["raw_media_persisted"] is False
    rows = {row["metric"]: row for row in report["comparisons"]}
    assert rows["video_ms_ssim"]["winner"] == "candidate"
    assert rows["video_temporal_derivative_error"]["winner"] == "candidate"
    assert rows["video_motion_weighted_detail_error"]["winner"] == "candidate"


def test_bounded_evaluator_is_deterministic():
    reference = _video(10)
    legacy = (_video(11) * 0.1 + reference * 0.9).clamp(0.0, 1.0)
    candidate = (_video(12) * 0.05 + reference * 0.95).clamp(0.0, 1.0)
    source_metadata = {
        "frame_count": 6,
        "height": 24,
        "width": 32,
        "channels": 3,
    }
    kwargs = dict(
        fps=24.0,
        benchmark_id="deterministic",
        seed=7,
        provenance=_provenance(),
        source_video_metadata=source_metadata,
        chunk_size=3,
    )

    first = evaluate_objective_media_bounded(
        reference,
        legacy,
        candidate,
        **copy.deepcopy(kwargs),
    )
    second = evaluate_objective_media_bounded(
        reference,
        legacy,
        candidate,
        **copy.deepcopy(kwargs),
    )

    assert first["comparisons"] == second["comparisons"]
    assert first["uncertainty"] == second["uncertainty"]
    assert first["compatibility_signature"] == second["compatibility_signature"]
