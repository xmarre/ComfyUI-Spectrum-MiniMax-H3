from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from . import objective_media as base

PROFILE_NAME = "sequential_bounded_luma_block_ssim_v1"
BLOCK_SIZE = 8


def _video_chunk(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
    chunk = value[start:end, ..., :3].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(chunk).all():
        raise base.ObjectiveMediaError("video contains NaN or infinite values")
    if base._safe_float(chunk.min()) < -1.0e-4 or base._safe_float(chunk.max()) > 1.0001:
        raise base.ObjectiveMediaError("IMAGE values must be in the decoded ComfyUI [0, 1] range")
    return chunk.clamp(0.0, 1.0).movedim(-1, 1).contiguous()


def _luma(value: torch.Tensor) -> torch.Tensor:
    return (
        value[:, 0:1] * 0.2126
        + value[:, 1:2] * 0.7152
        + value[:, 2:3] * 0.0722
    )


def _block_ssim_and_cs(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = max(1, min(BLOCK_SIZE, int(x.shape[-2]), int(x.shape[-1])))
    mean_x = F.avg_pool2d(x, block, stride=block, ceil_mode=True)
    mean_y = F.avg_pool2d(y, block, stride=block, ceil_mode=True)
    second_x = F.avg_pool2d(x.square(), block, stride=block, ceil_mode=True)
    second_y = F.avg_pool2d(y.square(), block, stride=block, ceil_mode=True)
    cross = F.avg_pool2d(x * y, block, stride=block, ceil_mode=True)
    var_x = (second_x - mean_x.square()).clamp_min(0.0)
    var_y = (second_y - mean_y.square()).clamp_min(0.0)
    covariance = cross - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2.0 * mean_x * mean_y + c1) / (
        mean_x.square() + mean_y.square() + c1
    )
    contrast_structure = (2.0 * covariance + c2) / (var_x + var_y + c2)
    ssim = (luminance * contrast_structure).mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)
    cs = contrast_structure.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    return ssim, cs


def _fast_ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    weights = (0.0448, 0.2856, 0.3001, 0.2363)
    levels = min(
        len(weights),
        max(1, math.floor(math.log2(max(1, min(x.shape[-2:])))) - 2),
    )
    levels = max(1, levels)
    active_weights = torch.tensor(weights[:levels], dtype=x.dtype, device=x.device)
    active_weights = active_weights / active_weights.sum()
    components: list[torch.Tensor] = []
    current_x = x
    current_y = y
    for level in range(levels):
        ssim, cs = _block_ssim_and_cs(current_x, current_y)
        component = ssim.add(1.0).mul(0.5) if level == levels - 1 else cs
        components.append(component.clamp_min(1.0e-8))
        if level < levels - 1:
            current_x = F.avg_pool2d(current_x, 2, stride=2, ceil_mode=True)
            current_y = F.avg_pool2d(current_y, 2, stride=2, ceil_mode=True)
    stacked = torch.stack(components, dim=0)
    return torch.prod(stacked ** active_weights[:, None], dim=0).clamp(0.0, 1.0)


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    kernel = value.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    return F.conv2d(value, kernel, padding=1)


def _multiscale_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    per_scale: list[torch.Tensor] = []
    current_x = x
    current_y = y
    for _ in range(3):
        per_scale.append((current_x - current_y).abs().mean(dim=(1, 2, 3)))
        if min(current_x.shape[-2:]) < 4:
            break
        current_x = F.avg_pool2d(current_x, 2, stride=2, ceil_mode=True)
        current_y = F.avg_pool2d(current_y, 2, stride=2, ceil_mode=True)
    return torch.stack(per_scale).mean(dim=0)


def _video_metrics(
    reference: torch.Tensor,
    legacy: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    reference = base._validate_video("reference_video", reference)
    legacy = base._validate_video("legacy_video", legacy)
    candidate = base._validate_video("candidate_video", candidate)
    if reference.shape != legacy.shape or reference.shape != candidate.shape:
        raise base.ObjectiveMediaError(
            "reference, legacy, and candidate analysis tensors must have identical shape"
        )
    frames = int(reference.shape[0])
    if frames > base.PER_FRAME_LIMIT:
        raise base.ObjectiveMediaError(
            f"video frame count exceeds the {base.PER_FRAME_LIMIT} frame report bound"
        )
    chunk_size = max(1, min(int(chunk_size), 16))
    names = ("legacy", "candidate")
    series: dict[str, dict[str, list[float]]] = {
        name: {
            "ssim": [],
            "ms_ssim": [],
            "psnr_db": [],
            "global_detail_error": [],
            "motion_weighted_detail_error": [],
            "temporal_derivative_error": [],
        }
        for name in names
    }
    previous_reference: torch.Tensor | None = None
    previous_values: dict[str, torch.Tensor | None] = {name: None for name in names}
    for start in range(0, frames, chunk_size):
        end = min(frames, start + chunk_size)
        ref_rgb = _video_chunk(reference, start, end)
        ref = _luma(ref_rgb)
        values_rgb = {
            "legacy": _video_chunk(legacy, start, end),
            "candidate": _video_chunk(candidate, start, end),
        }
        reference_detail = _laplacian(ref)
        for name, value_rgb in values_rgb.items():
            value = _luma(value_rgb)
            ssim, _ = _block_ssim_and_cs(ref, value)
            ms_ssim = _fast_ms_ssim(ref, value)
            mse = (ref_rgb - value_rgb).square().mean(dim=(1, 2, 3))
            psnr = torch.where(
                mse <= 1.0e-12,
                torch.full_like(mse, base.PERFECT_PSNR_DB),
                (-10.0 * torch.log10(mse.clamp_min(1.0e-12))).clamp_max(
                    base.PERFECT_PSNR_DB
                ),
            )
            detail_error_map = (reference_detail - _laplacian(value)).abs()
            global_detail = detail_error_map.mean(dim=(1, 2, 3))
            previous_value = previous_values[name]
            if previous_reference is None:
                sequence_ref = ref
                sequence_value = value
            else:
                sequence_ref = torch.cat((previous_reference, ref), dim=0)
                sequence_value = torch.cat((previous_value, value), dim=0)
            delta_ref = sequence_ref[1:] - sequence_ref[:-1]
            delta_value = sequence_value[1:] - sequence_value[:-1]
            temporal = _multiscale_l1(delta_ref, delta_value)
            motion = delta_ref.abs()
            motion = motion / motion.mean(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1.0e-6)
            motion = motion.clamp_max(8.0)
            if previous_reference is None:
                motion_detail = detail_error_map[1:]
            else:
                motion_detail = detail_error_map
            if motion.shape[0]:
                weighted = (
                    motion_detail * (1.0 + 4.0 * motion)
                ).sum(dim=(1, 2, 3)) / (
                    (1.0 + 4.0 * motion)
                    .sum(dim=(1, 2, 3))
                    .clamp_min(1.0e-6)
                )
            else:
                weighted = global_detail.new_empty((0,))
            for metric, tensor in (
                ("ssim", ssim),
                ("ms_ssim", ms_ssim),
                ("psnr_db", psnr),
                ("global_detail_error", global_detail),
            ):
                series[name][metric].extend(float(item) for item in tensor.tolist())
            temporal_values = [float(item) for item in temporal.tolist()]
            weighted_values = [float(item) for item in weighted.tolist()]
            if previous_reference is None:
                temporal_values.insert(0, 0.0)
                weighted_values.insert(0, float(global_detail[0]))
            series[name]["temporal_derivative_error"].extend(temporal_values)
            series[name]["motion_weighted_detail_error"].extend(weighted_values)
            previous_values[name] = value[-1:].clone()
        previous_reference = ref[-1:].clone()

    directions = {
        "ssim": True,
        "ms_ssim": True,
        "psnr_db": True,
        "global_detail_error": False,
        "motion_weighted_detail_error": False,
        "temporal_derivative_error": False,
    }
    metrics: dict[str, Any] = {}
    for name in names:
        metrics[name] = {}
        for metric, values in series[name].items():
            if len(values) != frames:
                raise base.ObjectiveMediaError(
                    f"internal bounded {metric} frame accounting mismatch"
                )
            summary = base._series_summary(
                values, higher_is_better=directions[metric]
            )
            summary["direction"] = (
                "higher_is_better" if directions[metric] else "lower_is_better"
            )
            summary["worst_window"] = base._rolling_worst(
                values,
                window=min(5, frames),
                higher_is_better=directions[metric],
            )
            metrics[name][metric] = summary
    return {
        "metadata": {
            "frame_count": frames,
            "height": int(reference.shape[1]),
            "width": int(reference.shape[2]),
            "input_channels": int(reference.shape[3]),
            "evaluated_channels": 3,
            "structural_channels": 1,
            "chunk_size": chunk_size,
            "value_range": [0.0, 1.0],
            "metric_profile": PROFILE_NAME,
            "ssim_estimator": "nonoverlapping_block_luma",
            "ssim_block_size": BLOCK_SIZE,
        },
        "metrics": metrics,
    }


def evaluate_objective_media_bounded(
    reference_video: torch.Tensor,
    legacy_video: torch.Tensor,
    candidate_video: torch.Tensor,
    *,
    fps: float,
    benchmark_id: str,
    seed: int | None,
    provenance: dict[str, Any],
    source_video_metadata: dict[str, Any],
    reference_audio: Any = None,
    legacy_audio: Any = None,
    candidate_audio: Any = None,
    chunk_size: int = 4,
) -> dict[str, Any]:
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise base.ObjectiveMediaError("benchmark_id must be a non-empty string")
    if (
        not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise base.ObjectiveMediaError("fps must be positive and finite")
    if not isinstance(provenance, dict):
        raise base.ObjectiveMediaError("provenance must be a JSON object")
    base._validate_provenance(provenance)
    audio_values = (reference_audio, legacy_audio, candidate_audio)
    if any(value is not None for value in audio_values) and not all(
        value is not None for value in audio_values
    ):
        raise base.ObjectiveMediaError(
            "audio comparison requires reference, legacy, and candidate AUDIO inputs together"
        )

    started = time.perf_counter()
    print(
        "Spectrum H3 objective bounded evaluator: VIDEO start "
        f"profile={PROFILE_NAME} analysis_shape={tuple(reference_video.shape)}"
    )
    video = _video_metrics(
        reference_video,
        legacy_video,
        candidate_video,
        chunk_size=chunk_size,
    )
    video["metadata"]["fps"] = float(fps)
    video["metadata"]["duration_seconds"] = (
        int(source_video_metadata["frame_count"]) / float(fps)
    )
    video["metadata"]["source_frame_count"] = int(
        source_video_metadata["frame_count"]
    )
    video["metadata"]["source_height"] = int(source_video_metadata["height"])
    video["metadata"]["source_width"] = int(source_video_metadata["width"])
    video["metadata"]["source_channels"] = int(source_video_metadata["channels"])
    print(
        "Spectrum H3 objective bounded evaluator: VIDEO done "
        f"elapsed={time.perf_counter() - started:.3f}s"
    )

    audio = None
    if reference_audio is not None:
        audio_started = time.perf_counter()
        print("Spectrum H3 objective bounded evaluator: AUDIO start")
        audio = base._audio_metrics(
            reference_audio,
            legacy_audio,
            candidate_audio,
        )
        print(
            "Spectrum H3 objective bounded evaluator: AUDIO done "
            f"elapsed={time.perf_counter() - audio_started:.3f}s"
        )

    comparisons = base._paired_comparisons(video, audio)
    uncertainty = {
        "video_ms_ssim": base._paired_block_bootstrap(
            video["metrics"]["legacy"]["ms_ssim"]["values"],
            video["metrics"]["candidate"]["ms_ssim"]["values"],
            higher_is_better=True,
        ),
        "video_temporal_derivative_error": base._paired_block_bootstrap(
            video["metrics"]["legacy"]["temporal_derivative_error"]["values"],
            video["metrics"]["candidate"]["temporal_derivative_error"]["values"],
            higher_is_better=False,
        ),
        "video_motion_weighted_detail_error": base._paired_block_bootstrap(
            video["metrics"]["legacy"]["motion_weighted_detail_error"]["values"],
            video["metrics"]["candidate"]["motion_weighted_detail_error"]["values"],
            higher_is_better=False,
        ),
    }
    compatibility = {
        "declared": provenance.get("compatibility", {}),
        "source_frame_count": int(source_video_metadata["frame_count"]),
        "source_resolution": [
            int(source_video_metadata["width"]),
            int(source_video_metadata["height"]),
        ],
        "analysis_resolution": [
            video["metadata"]["width"],
            video["metadata"]["height"],
        ],
        "fps": float(fps),
        "metric_profile": PROFILE_NAME,
        "audio": None
        if audio is None
        else {
            "sample_rate": audio["metadata"]["sample_rate"],
            "channels": audio["metadata"]["channels"],
            "samples": audio["metadata"]["samples"],
        },
    }
    signature = base._canonical_json(compatibility)
    report = {
        "schema_version": base.SCHEMA_VERSION,
        "kind": "spectrum_h3_objective_media_comparison",
        "benchmark_id": benchmark_id.strip(),
        "seed": seed,
        "scientific_reference": (
            "full native MiniMax H3 decoded output with Spectrum bypassed, "
            "evaluated through the deterministic bounded sequential metric transform"
        ),
        "roles": {
            "R": "native_reference",
            "A": "accelerated_legacy",
            "B": "accelerated_candidate",
        },
        "provenance": provenance,
        "compatibility": compatibility,
        "compatibility_signature": signature,
        "group_id": hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12],
        "video": video,
        "audio": audio,
        "comparisons": comparisons,
        "uncertainty": uncertainty,
        "verdict": base._verdict(comparisons, audio_present=audio is not None),
        "optional_backends": base.inspect_optional_backends(),
        "evaluator_profile": {
            "name": PROFILE_NAME,
            "purpose": (
                "Bound post-generation CPU/RAM cost for sequential ComfyUI R/A/B testing."
            ),
            "structural_metric": (
                "luma block-SSIM/MS-SSIM on deterministic downscaled analysis frames"
            ),
            "temporal_metric": "three-scale luma temporal-derivative error",
            "detail_metric": "luma Laplacian with native-motion weighting",
            "full_resolution_raw_media_retained": False,
        },
        "boundaries": {
            "media_stage": "decoded tensors before video encoding or audio muxing",
            "hidden_space_evidence_is_separate": True,
            "production_defaults_changed": False,
            "transformer_calls": 0,
            "raw_media_persisted": False,
        },
    }
    print(
        "Spectrum H3 objective bounded evaluator: complete "
        f"elapsed={time.perf_counter() - started:.3f}s"
    )
    return report


__all__ = [
    "BLOCK_SIZE",
    "PROFILE_NAME",
    "evaluate_objective_media_bounded",
]
