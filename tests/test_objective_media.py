from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from comfyui_spectrum_h3.objective_media import (
    ObjectiveMediaError,
    aggregate_objective_reports,
    evaluate_objective_media,
    persist_objective_report,
)
from comfyui_spectrum_h3.objective_media_nodes import (
    NODE_CLASS_MAPPINGS,
    SpectrumH3ObjectiveMediaStage,
    SpectrumH3ObjectiveQualityCompare,
    SpectrumH3ObjectiveStagedQualityCompare,
)


def _video(frames: int = 8, height: int = 32, width: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    base = torch.rand((frames, height, width, 3), generator=generator) * 0.2
    for index in range(frames):
        x = 2 + index * 2
        base[index, 10:18, x : x + 6] += torch.linspace(0.0, 0.8, 6)[None, :, None]
    return base.clamp(0.0, 1.0)


def _audio(sample_rate: int = 8000, seconds: float = 1.0) -> dict[str, object]:
    samples = round(sample_rate * seconds)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = 0.5 * torch.sin(2.0 * math.pi * 220.0 * time)
    waveform += 0.2 * torch.sin(2.0 * math.pi * 1300.0 * time)
    return {"waveform": waveform[None, None, :], "sample_rate": sample_rate}


def _evaluate(
    reference: torch.Tensor,
    legacy: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_audio=None,
    legacy_audio=None,
    candidate_audio=None,
    benchmark_id: str = "case-a",
    seed: int = 1,
):
    provenance = {
        "compatibility": {
            "model": "fixture",
            "model_weights": "fixture-sha",
            "precision": "fp32",
            "sampler": "er_sde",
            "scheduler": "fixture",
            "steps": 20,
            "conditioning": "fixture",
            "video_vae": "fixture",
            "audio_decoder": "fixture",
            "generation_settings": {"resolution_family": "fixture"},
        },
        "R": {"spectrum": "bypassed"},
        "A": {"generic_correction_mode": "legacy"},
        "B": {"generic_correction_mode": "coordinate_rls"},
    }
    return evaluate_objective_media(
        reference,
        legacy,
        candidate,
        fps=24.0,
        benchmark_id=benchmark_id,
        seed=seed,
        provenance=provenance,
        reference_audio=reference_audio,
        legacy_audio=legacy_audio,
        candidate_audio=candidate_audio,
        chunk_size=3,
    )


def _rows(report):
    return {row["metric"]: row for row in report["comparisons"]}


def test_identical_media_is_an_exact_tie_and_perfect_reference():
    video = _video()
    audio = _audio()
    report = _evaluate(
        video,
        video.clone(),
        video.clone(),
        reference_audio=audio,
        legacy_audio={"waveform": audio["waveform"].clone(), "sample_rate": 8000},
        candidate_audio={"waveform": audio["waveform"].clone(), "sample_rate": 8000},
    )
    rows = _rows(report)
    assert report["verdict"]["value"] == "mixed_or_inconclusive"
    assert report["video"]["metrics"]["legacy"]["ms_ssim"]["mean"] == pytest.approx(1.0)
    assert report["video"]["metrics"]["legacy"]["temporal_derivative_error"]["mean"] == 0.0
    assert report["audio"]["metrics"]["legacy"]["normalized_correlation"]["value"] == pytest.approx(1.0)
    assert report["audio"]["metrics"]["legacy"]["si_sdr_db"]["value"] == 120.0
    assert all(row["winner"] == "tie" for row in rows.values())


def test_swapping_roles_swaps_pairwise_advantages_and_favored_verdict():
    reference = _video()
    legacy_noise = torch.randn(reference.shape, generator=torch.Generator().manual_seed(5))
    candidate_noise = torch.randn(reference.shape, generator=torch.Generator().manual_seed(6))
    legacy = (reference + 0.08 * legacy_noise).clamp(0.0, 1.0)
    candidate = (reference + 0.01 * candidate_noise).clamp(0.0, 1.0)
    forward = _evaluate(reference, legacy, candidate)
    reverse = _evaluate(reference, candidate, legacy, benchmark_id="case-b")
    forward_rows = _rows(forward)
    reverse_rows = _rows(reverse)
    assert forward["verdict"]["value"] == "candidate_favored"
    assert reverse["verdict"]["value"] == "legacy_favored"
    for name in forward_rows:
        assert forward_rows[name]["candidate_relative_advantage"] == pytest.approx(
            -reverse_rows[name]["candidate_relative_advantage"],
            rel=1.0e-5,
            abs=1.0e-8,
        )


def test_repeatability_is_byte_stable_except_backend_environment():
    reference = _video()
    legacy = reference.roll(1, dims=0)
    first = _evaluate(reference, legacy, reference.clone())
    second = _evaluate(reference, legacy, reference.clone())
    assert first == second


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value[:-1],
        lambda value: value[:, :-1],
        lambda value: torch.cat((value, torch.ones_like(value[..., :1])), dim=-1),
    ),
)
def test_video_topology_mismatch_is_rejected(mutation):
    reference = _video()
    with pytest.raises(ObjectiveMediaError, match="identical frame count, resolution, and channels"):
        _evaluate(reference, mutation(reference), reference)


def test_missing_audio_is_supported_and_partial_audio_is_rejected():
    video = _video()
    assert _evaluate(video, video, video)["audio"] is None
    with pytest.raises(ObjectiveMediaError, match="requires reference, legacy, and candidate"):
        _evaluate(video, video, video, reference_audio=_audio())


def test_audio_resampling_is_deterministic_and_recorded():
    video = _video()
    reference = _audio(8000)
    legacy = _audio(16000)
    candidate = _audio(8000)
    first = _evaluate(
        video,
        video,
        video,
        reference_audio=reference,
        legacy_audio=legacy,
        candidate_audio=candidate,
    )
    second = _evaluate(
        video,
        video,
        video,
        reference_audio=reference,
        legacy_audio=legacy,
        candidate_audio=candidate,
    )
    assert first["audio"] == second["audio"]
    assert first["audio"]["metadata"]["original_sample_rates"]["legacy"] == 16000
    assert first["audio"]["metadata"]["resampler"] == "torch_linear_align_corners_false"


def test_blur_noise_and_local_detail_loss_move_video_metrics():
    reference = _video()
    channel_first = reference.movedim(-1, 1)
    blurred = F.avg_pool2d(channel_first, 5, stride=1, padding=2).movedim(1, -1)
    noise = torch.randn(reference.shape, generator=torch.Generator().manual_seed(9))
    noisy = (reference + 0.08 * noise).clamp(0.0, 1.0)
    detail_loss = reference.clone()
    detail_loss[:, 10:18, 2:24] = blurred[:, 10:18, 2:24]
    blur_report = _evaluate(reference, blurred, reference)
    noise_report = _evaluate(reference, noisy, reference)
    detail_report = _evaluate(reference, detail_loss, reference)
    for report in (blur_report, noise_report, detail_report):
        metrics = report["video"]["metrics"]
        assert metrics["legacy"]["ms_ssim"]["mean"] < metrics["candidate"]["ms_ssim"]["mean"]
        assert metrics["legacy"]["global_detail_error"]["mean"] > metrics["candidate"]["global_detail_error"]["mean"]


def test_freeze_and_temporal_jitter_move_temporal_metric():
    reference = _video()
    frozen = reference.clone()
    frozen[3:6] = frozen[2]
    jittered = reference.clone()
    jittered[1::2] = jittered[1::2].roll(2, dims=2)
    frozen_report = _evaluate(reference, frozen, reference)
    jitter_report = _evaluate(reference, jittered, reference)
    for report in (frozen_report, jitter_report):
        temporal = report["video"]["metrics"]
        assert temporal["legacy"]["temporal_derivative_error"]["mean"] > 0.0
        assert temporal["candidate"]["temporal_derivative_error"]["mean"] == 0.0


def test_moving_detail_metric_amplifies_small_local_degradation():
    reference = _video(frames=10, height=64, width=64)
    degraded = reference.clone()
    for index in range(reference.shape[0]):
        x = 2 + index * 2
        patch = degraded[index : index + 1, 10:18, x : x + 6].movedim(-1, 1)
        degraded[index, 10:18, x : x + 6] = F.avg_pool2d(patch, 3, stride=1, padding=1).movedim(1, -1)[0]
    report = _evaluate(reference, degraded, reference)
    legacy = report["video"]["metrics"]["legacy"]
    assert legacy["motion_weighted_detail_error"]["mean"] > legacy["global_detail_error"]["mean"]
    assert legacy["motion_weighted_detail_error"]["worst_value"] > 0.0


def test_audio_noise_lowpass_click_shift_and_boundary_corruption_are_detected():
    video = _video()
    reference = _audio(8000, 1.5)
    waveform = reference["waveform"]
    degradations = []
    noise = torch.randn(waveform.shape, generator=torch.Generator().manual_seed(12))
    degradations.append((waveform + 0.05 * noise).clamp(-1.0, 1.0))
    degradations.append(F.avg_pool1d(waveform, 15, stride=1, padding=7))
    clicked = waveform.clone()
    clicked[..., 3000] += 1.0
    degradations.append(clicked)
    degradations.append(torch.roll(waveform, shifts=16, dims=-1))
    boundary = waveform.clone()
    boundary[..., :400] = 0.0
    boundary[..., -400:] = 0.0
    degradations.append(boundary)
    for index, degraded in enumerate(degradations):
        degraded_audio = {"waveform": degraded, "sample_rate": 8000}
        report = _evaluate(
            video,
            video,
            video,
            reference_audio=reference,
            legacy_audio=degraded_audio,
            candidate_audio=reference,
            benchmark_id=f"audio-{index}",
        )
        metrics = report["audio"]["metrics"]
        assert metrics["legacy"]["mrstft_log_magnitude_error"]["value"] > 0.0
        assert metrics["legacy"]["windowed_spectral_error"]["worst_value"] > 0.0
        assert metrics["candidate"]["mrstft_log_magnitude_error"]["value"] == 0.0


def test_bounded_alignment_reports_lag_without_changing_primary_metrics():
    video = _video()
    reference = _audio(8000, 1.0)
    small_shift = {"waveform": torch.roll(reference["waveform"], 24, -1), "sample_rate": 8000}
    report = _evaluate(
        video,
        video,
        video,
        reference_audio=reference,
        legacy_audio=small_shift,
        candidate_audio=reference,
    )
    legacy = report["audio"]["metrics"]["legacy"]
    assert abs(legacy["bounded_alignment_diagnostic"]["lag_ms"]) <= 20.0
    assert legacy["bounded_alignment_diagnostic"]["primary_metrics_use_alignment"] is False
    assert legacy["mrstft_log_magnitude_error"]["value"] > 0.0


def test_bounded_alignment_cannot_erase_a_larger_timing_error():
    video = _video()
    reference = _audio(8000, 1.0)
    large_shift = {"waveform": torch.roll(reference["waveform"], 333, -1), "sample_rate": 8000}
    report = _evaluate(
        video,
        video,
        video,
        reference_audio=reference,
        legacy_audio=large_shift,
        candidate_audio=reference,
    )
    legacy = report["audio"]["metrics"]["legacy"]
    assert abs(legacy["bounded_alignment_diagnostic"]["lag_ms"]) <= 20.0
    assert legacy["mrstft_log_magnitude_error"]["value"] > 0.0
    assert legacy["normalized_correlation"]["value"] < 1.0


def test_si_sdr_treats_silence_as_a_failure_for_non_silent_reference():
    video = _video()
    reference = _audio(8000, 1.0)
    silence = {"waveform": torch.zeros_like(reference["waveform"]), "sample_rate": 8000}
    report = _evaluate(
        video,
        video,
        video,
        reference_audio=reference,
        legacy_audio=silence,
        candidate_audio=reference,
    )
    assert report["audio"]["metrics"]["legacy"]["si_sdr_db"]["value"] == -120.0


def test_persistence_is_report_only_atomic_and_aggregates_complete_triads(tmp_path):
    reference = _video()
    first = _evaluate(reference, reference.roll(1, 0), reference, benchmark_id="seed-1", seed=1)
    second = _evaluate(reference, reference.roll(1, 0), reference, benchmark_id="seed-2", seed=2)
    persisted_first = persist_objective_report(first, root=tmp_path)
    persisted_second = persist_objective_report(second, root=tmp_path)
    assert persisted_first.json_path.is_file()
    assert persisted_second.markdown_path.is_file()
    assert persisted_second.run_count == 2
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.png"))
    aggregate = json.loads(persisted_second.aggregate_json_path.read_text(encoding="utf-8"))
    assert aggregate["independent_case_count"] == 2
    assert aggregate["cross_validation"].startswith("none")


def test_aggregate_rejects_incompatible_or_duplicate_cases():
    reference = _video()
    first = _evaluate(reference, reference, reference, benchmark_id="a", seed=1)
    duplicate = _evaluate(reference, reference, reference, benchmark_id="a", seed=1)
    incompatible = _evaluate(reference[:, :, :-1], reference[:, :, :-1], reference[:, :, :-1], benchmark_id="b", seed=2)
    with pytest.raises(ObjectiveMediaError, match="duplicate"):
        aggregate_objective_reports([first, duplicate])
    with pytest.raises(ObjectiveMediaError, match="incompatible"):
        aggregate_objective_reports([first, incompatible])


def test_node_registration_and_optional_audio_schema():
    assert NODE_CLASS_MAPPINGS["SpectrumH3ObjectiveQualityCompare"] is SpectrumH3ObjectiveQualityCompare
    schema = SpectrumH3ObjectiveQualityCompare.INPUT_TYPES()
    assert schema["required"]["reference_video"][0] == "IMAGE"
    assert schema["optional"]["reference_audio"][0] == "AUDIO"
    assert SpectrumH3ObjectiveQualityCompare.OUTPUT_NODE is True
    assert NODE_CLASS_MAPPINGS["SpectrumH3ObjectiveMediaStage"] is SpectrumH3ObjectiveMediaStage
    assert (
        NODE_CLASS_MAPPINGS["SpectrumH3ObjectiveStagedQualityCompare"]
        is SpectrumH3ObjectiveStagedQualityCompare
    )


def test_stage_moves_video_and_audio_to_cpu_without_disk_io():
    video = _video()
    audio = _audio()
    (staged,) = SpectrumH3ObjectiveMediaStage().stage(video, audio)
    assert staged["video"].device.type == "cpu"
    assert staged["audio"]["waveform"].device.type == "cpu"
    assert staged["audio"]["sample_rate"] == 8000
