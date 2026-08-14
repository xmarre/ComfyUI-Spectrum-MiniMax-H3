from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

SCHEMA_VERSION = 1
STORE_DIRECTORY_NAME = "spectrum_h3/objective_media/v1"
MAX_REPORTS_PER_GROUP = 24
MAX_REPORT_GROUPS = 24
PER_FRAME_LIMIT = 4096
PER_WINDOW_LIMIT = 4096
PERFECT_PSNR_DB = 120.0
PERFECT_SI_SDR_DB = 120.0
PRIMARY_IMPROVEMENT_FRACTION = 0.01
MATERIAL_REGRESSION_FRACTION = 0.02
WORST_CASE_REGRESSION_FRACTION = 0.05
ABSOLUTE_TOLERANCE = 1.0e-8
REQUIRED_COMPATIBILITY_FIELDS = frozenset(
    {
        "model",
        "model_weights",
        "precision",
        "sampler",
        "scheduler",
        "steps",
        "conditioning",
        "video_vae",
        "audio_decoder",
        "generation_settings",
    }
)

_COMPARISON_METADATA = {
    "video_ms_ssim": ("verdict_primary", "score_points", "relative_and_absolute"),
    "video_psnr_db": ("diagnostic", "dB", "absolute_delta"),
    "video_temporal_derivative_error": (
        "verdict_primary",
        "error_units",
        "relative_and_absolute",
    ),
    "video_motion_weighted_detail_error": (
        "verdict_primary",
        "error_units",
        "relative_and_absolute",
    ),
    "video_worst_frame_ms_ssim": (
        "verdict_guardrail",
        "score_points",
        "relative_and_absolute",
    ),
    "audio_mrstft_log_magnitude_error": (
        "verdict_primary",
        "error_units",
        "relative_and_absolute",
    ),
    "audio_normalized_correlation": (
        "diagnostic",
        "correlation_points",
        "absolute_delta",
    ),
    "audio_si_sdr_db": ("diagnostic", "dB", "absolute_delta"),
    "audio_worst_window_spectral_error": (
        "verdict_guardrail",
        "error_units",
        "relative_and_absolute",
    ),
    "audio_absolute_bounded_lag_ms": (
        "verdict_guardrail",
        "ms",
        "absolute_delta",
    ),
}


class ObjectiveMediaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PersistedObjectiveReport:
    group_id: str
    run_count: int
    json_path: Path
    markdown_path: Path
    aggregate_json_path: Path
    aggregate_markdown_path: Path


def default_store_root() -> Path:
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
        base = Path.cwd() / "user" / "__cache"
    return base / STORE_DIRECTORY_NAME


def _atomic_write_text(path: Path, value: str) -> None:
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
            handle.write(value)
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_provenance(provenance: dict[str, Any]) -> None:
    compatibility = provenance.get("compatibility")
    if not isinstance(compatibility, dict):
        raise ObjectiveMediaError("provenance.compatibility must be a JSON object")
    missing = sorted(REQUIRED_COMPATIBILITY_FIELDS.difference(compatibility))
    if missing:
        raise ObjectiveMediaError(
            "provenance.compatibility is missing required grouping fields: " + ", ".join(missing)
        )
    empty = sorted(
        name
        for name in REQUIRED_COMPATIBILITY_FIELDS.difference({"generation_settings"})
        if compatibility[name] is None
        or (isinstance(compatibility[name], str) and not compatibility[name].strip())
    )
    if empty:
        raise ObjectiveMediaError(
            "provenance.compatibility has empty required grouping fields: " + ", ".join(empty)
        )
    if not isinstance(compatibility["generation_settings"], dict):
        raise ObjectiveMediaError("provenance.compatibility.generation_settings must be a JSON object")
    for role in ("R", "A", "B"):
        value = provenance.get(role)
        if not isinstance(value, dict) or not value:
            raise ObjectiveMediaError(f"provenance.{role} must be a non-empty role-specific JSON object")


def _safe_float(value: torch.Tensor | float) -> float:
    result = float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value)
    if not math.isfinite(result):
        raise ObjectiveMediaError("metric calculation produced a non-finite value")
    return result


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ObjectiveMediaError("cannot summarize an empty metric series")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _series_summary(values: list[float], *, higher_is_better: bool) -> dict[str, Any]:
    if not values:
        raise ObjectiveMediaError("cannot summarize an empty metric series")
    if len(values) > PER_FRAME_LIMIT:
        raise ObjectiveMediaError(f"metric series exceeds the {PER_FRAME_LIMIT} value bound")
    ordered = sorted(values, reverse=not higher_is_better)
    decile_count = max(1, math.ceil(len(ordered) * 0.1))
    worst_index = min(range(len(values)), key=values.__getitem__) if higher_is_better else max(
        range(len(values)), key=values.__getitem__
    )
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "worst_decile_mean": statistics.fmean(ordered[:decile_count]),
        "worst_value": values[worst_index],
        "worst_index": worst_index,
        "values": values,
    }


def _rolling_worst(values: list[float], *, window: int, higher_is_better: bool) -> dict[str, Any]:
    if not values:
        raise ObjectiveMediaError("cannot find a worst window in an empty metric series")
    width = max(1, min(window, len(values)))
    means = [statistics.fmean(values[index : index + width]) for index in range(len(values) - width + 1)]
    worst_start = min(range(len(means)), key=means.__getitem__) if higher_is_better else max(
        range(len(means)), key=means.__getitem__
    )
    return {
        "window_size": width,
        "start_index": worst_start,
        "end_index_inclusive": worst_start + width - 1,
        "mean": means[worst_start],
    }


def _validate_video(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ObjectiveMediaError(f"{name} must be a torch IMAGE tensor")
    if value.ndim != 4:
        raise ObjectiveMediaError(f"{name} must have shape [frames, height, width, channels]")
    if value.shape[0] < 2:
        raise ObjectiveMediaError(f"{name} must contain at least two frames")
    if value.shape[-1] < 3:
        raise ObjectiveMediaError(f"{name} must contain at least three color channels")
    if not value.is_floating_point():
        raise ObjectiveMediaError(f"{name} must be floating point")
    return value


def _video_chunk(value: torch.Tensor, start: int, end: int) -> torch.Tensor:
    chunk = value[start:end, ..., :3].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(chunk).all():
        raise ObjectiveMediaError("video contains NaN or infinite values")
    if _safe_float(chunk.min()) < -1.0e-4 or _safe_float(chunk.max()) > 1.0001:
        raise ObjectiveMediaError("IMAGE values must be in the decoded ComfyUI [0, 1] range")
    return chunk.clamp(0.0, 1.0).movedim(-1, 1).contiguous()


def _ssim_and_cs(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = x.shape[-2:]
    kernel = min(11, height, width)
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(1, kernel)
    padding = kernel // 2
    mean_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mean_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    var_x = F.avg_pool2d(x * x, kernel, stride=1, padding=padding) - mean_x.square()
    var_y = F.avg_pool2d(y * y, kernel, stride=1, padding=padding) - mean_y.square()
    covariance = F.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (2.0 * mean_x * mean_y + c1) / (mean_x.square() + mean_y.square() + c1)
    contrast_structure = (2.0 * covariance + c2) / (var_x.clamp_min(0.0) + var_y.clamp_min(0.0) + c2)
    ssim = (luminance * contrast_structure).mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)
    cs = contrast_structure.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    return ssim, cs


def _ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    levels = min(len(weights), math.floor(math.log2(min(x.shape[-2:]))) + 1)
    levels = max(1, levels)
    active_weights = torch.tensor(weights[:levels], dtype=x.dtype)
    active_weights = active_weights / active_weights.sum()
    components: list[torch.Tensor] = []
    current_x = x
    current_y = y
    for level in range(levels):
        ssim, cs = _ssim_and_cs(current_x, current_y)
        components.append((ssim.add(1.0).mul(0.5) if level == levels - 1 else cs).clamp_min(1.0e-8))
        if level < levels - 1:
            current_x = F.avg_pool2d(current_x, 2, stride=2, ceil_mode=True)
            current_y = F.avg_pool2d(current_y, 2, stride=2, ceil_mode=True)
    stacked = torch.stack(components, dim=0)
    return torch.prod(stacked ** active_weights[:, None], dim=0).clamp(0.0, 1.0)


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    kernel = value.new_tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)))
    kernel = kernel.view(1, 1, 3, 3).repeat(value.shape[1], 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=value.shape[1])


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
    reference = _validate_video("reference_video", reference)
    legacy = _validate_video("legacy_video", legacy)
    candidate = _validate_video("candidate_video", candidate)
    if reference.shape != legacy.shape or reference.shape != candidate.shape:
        raise ObjectiveMediaError(
            "reference, legacy, and candidate IMAGE tensors must have identical frame count, resolution, and channels"
        )
    frames = int(reference.shape[0])
    if frames > PER_FRAME_LIMIT:
        raise ObjectiveMediaError(f"video frame count exceeds the {PER_FRAME_LIMIT} frame report bound")
    chunk_size = max(1, min(int(chunk_size), 32))
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
        ref = _video_chunk(reference, start, end)
        values = {
            "legacy": _video_chunk(legacy, start, end),
            "candidate": _video_chunk(candidate, start, end),
        }
        reference_detail = _laplacian(ref)
        for name, value in values.items():
            ssim, _ = _ssim_and_cs(ref, value)
            ms_ssim = _ms_ssim(ref, value)
            mse = (ref - value).square().mean(dim=(1, 2, 3))
            psnr = torch.where(
                mse <= 1.0e-12,
                torch.full_like(mse, PERFECT_PSNR_DB),
                (-10.0 * torch.log10(mse.clamp_min(1.0e-12))).clamp_max(PERFECT_PSNR_DB),
            )
            detail_error_map = (reference_detail - _laplacian(value)).abs().mean(dim=1, keepdim=True)
            global_detail = detail_error_map.mean(dim=(1, 2, 3))
            sequence_ref = ref if previous_reference is None else torch.cat((previous_reference, ref), dim=0)
            previous_value = previous_values[name]
            sequence_value = value if previous_value is None else torch.cat((previous_value, value), dim=0)
            delta_ref = sequence_ref[1:] - sequence_ref[:-1]
            delta_value = sequence_value[1:] - sequence_value[:-1]
            temporal = _multiscale_l1(delta_ref, delta_value)
            motion = delta_ref.abs().mean(dim=1, keepdim=True)
            motion = motion / motion.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
            motion = motion.clamp_max(8.0)
            detail_for_motion = detail_error_map if previous_reference is None else torch.cat(
                (detail_error_map[:1], detail_error_map), dim=0
            )
            detail_for_motion = detail_for_motion[-motion.shape[0] :]
            weighted_detail = (detail_for_motion * (1.0 + 4.0 * motion)).sum(dim=(1, 2, 3)) / (
                (1.0 + 4.0 * motion).sum(dim=(1, 2, 3)).clamp_min(1.0e-6)
            )
            for metric, tensor in (
                ("ssim", ssim),
                ("ms_ssim", ms_ssim),
                ("psnr_db", psnr),
                ("global_detail_error", global_detail),
            ):
                series[name][metric].extend(float(item) for item in tensor.tolist())
            temporal_values = [float(item) for item in temporal.tolist()]
            weighted_values = [float(item) for item in weighted_detail.tolist()]
            if previous_reference is None:
                temporal_values.insert(0, 0.0)
                weighted_values.insert(0, float(global_detail[0]))
            series[name]["temporal_derivative_error"].extend(temporal_values)
            series[name]["motion_weighted_detail_error"].extend(weighted_values)
            previous_values[name] = value[-1:].clone()
        previous_reference = ref[-1:].clone()
    metrics: dict[str, Any] = {}
    directions = {
        "ssim": True,
        "ms_ssim": True,
        "psnr_db": True,
        "global_detail_error": False,
        "motion_weighted_detail_error": False,
        "temporal_derivative_error": False,
    }
    for name in names:
        metrics[name] = {}
        for metric, values in series[name].items():
            if len(values) != frames:
                raise ObjectiveMediaError(f"internal {metric} frame accounting mismatch")
            metrics[name][metric] = _series_summary(values, higher_is_better=directions[metric])
            metrics[name][metric]["direction"] = "higher_is_better" if directions[metric] else "lower_is_better"
            metrics[name][metric]["worst_window"] = _rolling_worst(
                values,
                window=min(5, frames),
                higher_is_better=directions[metric],
            )
    return {
        "metadata": {
            "frame_count": frames,
            "height": int(reference.shape[1]),
            "width": int(reference.shape[2]),
            "input_channels": int(reference.shape[3]),
            "evaluated_channels": 3,
            "chunk_size": chunk_size,
            "value_range": [0.0, 1.0],
        },
        "metrics": metrics,
    }


def _validate_audio(name: str, value: Any) -> tuple[torch.Tensor, int]:
    if not isinstance(value, dict):
        raise ObjectiveMediaError(f"{name} must be a ComfyUI AUDIO dictionary")
    waveform = value.get("waveform")
    sample_rate = value.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ObjectiveMediaError(f"{name}.waveform must have shape [batch, channels, samples]")
    if waveform.shape[0] != 1 or waveform.shape[1] < 1 or waveform.shape[2] < 32:
        raise ObjectiveMediaError(f"{name}.waveform must contain one non-empty batch")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ObjectiveMediaError(f"{name}.sample_rate must be a positive integer")
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ObjectiveMediaError(f"{name}.waveform contains NaN or infinite values")
    return waveform, sample_rate


def _resample_linear(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_length = round(waveform.shape[-1] * target_rate / source_rate)
    if target_length < 32:
        raise ObjectiveMediaError("resampled audio would be too short")
    return F.interpolate(waveform, size=target_length, mode="linear", align_corners=False)


def _prepare_audio_triplet(
    reference_audio: Any,
    legacy_audio: Any,
    candidate_audio: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    reference, reference_rate = _validate_audio("reference_audio", reference_audio)
    legacy, legacy_rate = _validate_audio("legacy_audio", legacy_audio)
    candidate, candidate_rate = _validate_audio("candidate_audio", candidate_audio)
    if reference.shape[1] != legacy.shape[1] or reference.shape[1] != candidate.shape[1]:
        raise ObjectiveMediaError("audio channel counts must match; implicit downmixing is not allowed")
    original_rates = {
        "reference": reference_rate,
        "legacy": legacy_rate,
        "candidate": candidate_rate,
    }
    legacy = _resample_linear(legacy, legacy_rate, reference_rate)
    candidate = _resample_linear(candidate, candidate_rate, reference_rate)
    lengths = (reference.shape[-1], legacy.shape[-1], candidate.shape[-1])
    if max(lengths) - min(lengths) > 1:
        raise ObjectiveMediaError(
            "audio durations differ after deterministic resampling; timing differences must not be cropped away"
        )
    common_length = min(lengths)
    reference = reference[..., :common_length]
    legacy = legacy[..., :common_length]
    candidate = candidate[..., :common_length]
    return reference, legacy, candidate, {
        "sample_rate": reference_rate,
        "original_sample_rates": original_rates,
        "resampler": "torch_linear_align_corners_false",
        "channels": int(reference.shape[1]),
        "samples": common_length,
        "duration_seconds": common_length / reference_rate,
        "length_adjustment_samples": {
            "reference": lengths[0] - common_length,
            "legacy": lengths[1] - common_length,
            "candidate": lengths[2] - common_length,
        },
    }


def _stft_log_distance(reference: torch.Tensor, value: torch.Tensor, fft_size: int) -> torch.Tensor:
    fft_size = min(int(fft_size), int(reference.shape[-1]))
    if fft_size < 16:
        return (reference - value).abs().mean()
    hop = max(1, fft_size // 4)
    window = torch.hann_window(fft_size, dtype=reference.dtype)
    reference_stft = torch.stft(
        reference.reshape(-1, reference.shape[-1]),
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=True,
    )
    value_stft = torch.stft(
        value.reshape(-1, value.shape[-1]),
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=True,
    )
    return (torch.log1p(reference_stft.abs()) - torch.log1p(value_stft.abs())).abs().mean()


def _mrstft_distance(reference: torch.Tensor, value: torch.Tensor) -> tuple[float, dict[str, float]]:
    scale_values: dict[str, float] = {}
    for fft_size in (256, 512, 1024, 2048):
        scale_values[str(fft_size)] = _safe_float(_stft_log_distance(reference, value, fft_size))
    return statistics.fmean(scale_values.values()), scale_values


def _normalized_correlation(reference: torch.Tensor, value: torch.Tensor) -> float:
    reference_flat = reference.reshape(-1)
    value_flat = value.reshape(-1)
    reference_flat = reference_flat - reference_flat.mean()
    value_flat = value_flat - value_flat.mean()
    denominator = torch.linalg.vector_norm(reference_flat) * torch.linalg.vector_norm(value_flat)
    if _safe_float(denominator) <= 1.0e-12:
        return 1.0 if torch.equal(reference, value) else 0.0
    return max(-1.0, min(1.0, _safe_float(torch.dot(reference_flat, value_flat) / denominator)))


def _si_sdr(reference: torch.Tensor, value: torch.Tensor) -> float:
    reference_flat = reference.reshape(-1).to(torch.float64)
    value_flat = value.reshape(-1).to(torch.float64)
    reference_energy = torch.dot(reference_flat, reference_flat)
    if _safe_float(reference_energy) <= 1.0e-20:
        return PERFECT_SI_SDR_DB if torch.equal(reference, value) else -PERFECT_SI_SDR_DB
    scale = torch.dot(value_flat, reference_flat) / reference_energy
    target = scale * reference_flat
    noise = value_flat - target
    noise_energy = torch.dot(noise, noise)
    if _safe_float(noise_energy) <= 1.0e-20:
        return PERFECT_SI_SDR_DB if _safe_float(torch.dot(target, target)) > 1.0e-20 else -PERFECT_SI_SDR_DB
    score = 10.0 * torch.log10(torch.dot(target, target).clamp_min(1.0e-20) / noise_energy)
    return max(-PERFECT_SI_SDR_DB, min(PERFECT_SI_SDR_DB, _safe_float(score)))


def _bounded_lag(reference: torch.Tensor, value: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    maximum_lag_ms = 20.0
    stride = max(1, math.ceil(sample_rate / 8000))
    reference_mono = reference.mean(dim=1)[0, ::stride]
    value_mono = value.mean(dim=1)[0, ::stride]
    maximum_lag = max(1, round(maximum_lag_ms * sample_rate / 1000 / stride))
    best_lag = 0
    best_score = -math.inf
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            ref_part = reference_mono[-lag:]
            value_part = value_mono[: value_mono.shape[0] + lag]
        elif lag > 0:
            ref_part = reference_mono[: reference_mono.shape[0] - lag]
            value_part = value_mono[lag:]
        else:
            ref_part = reference_mono
            value_part = value_mono
        score = _normalized_correlation(ref_part, value_part)
        if score > best_score:
            best_score = score
            best_lag = lag
    lag_samples = best_lag * stride
    return {
        "search_bound_ms": maximum_lag_ms,
        "search_stride_samples": stride,
        "lag_samples": lag_samples,
        "lag_ms": lag_samples * 1000.0 / sample_rate,
        "aligned_correlation_diagnostic": best_score,
        "primary_metrics_use_alignment": False,
    }


def _audio_window_scores(
    reference: torch.Tensor,
    value: torch.Tensor,
    sample_rate: int,
) -> list[float]:
    window_samples = max(256, round(sample_rate * 0.5))
    scores: list[float] = []
    for start in range(0, reference.shape[-1], window_samples):
        ref_window = reference[..., start : start + window_samples]
        value_window = value[..., start : start + window_samples]
        if ref_window.shape[-1] < 16:
            continue
        scores.append(_safe_float(_stft_log_distance(ref_window, value_window, 1024)))
    if not scores:
        scores.append(_safe_float((reference - value).abs().mean()))
    if len(scores) > PER_WINDOW_LIMIT:
        raise ObjectiveMediaError(f"audio window count exceeds the {PER_WINDOW_LIMIT} report bound")
    return scores


def _third_summaries(values: list[float]) -> dict[str, float]:
    boundaries = [round(index * len(values) / 3) for index in range(4)]
    labels = ("start", "middle", "end")
    result: dict[str, float] = {}
    for index, label in enumerate(labels):
        segment = values[boundaries[index] : boundaries[index + 1]]
        if not segment:
            segment = [values[min(index, len(values) - 1)]]
        result[label] = statistics.fmean(segment)
    return result


def _audio_metrics(
    reference_audio: Any,
    legacy_audio: Any,
    candidate_audio: Any,
) -> dict[str, Any]:
    reference, legacy, candidate, metadata = _prepare_audio_triplet(
        reference_audio,
        legacy_audio,
        candidate_audio,
    )
    metrics: dict[str, Any] = {}
    for name, value in (("legacy", legacy), ("candidate", candidate)):
        mrstft, scales = _mrstft_distance(reference, value)
        window_values = _audio_window_scores(reference, value, metadata["sample_rate"])
        window_summary = _series_summary(window_values, higher_is_better=False)
        window_summary["direction"] = "lower_is_better"
        window_summary["window_seconds"] = 0.5
        window_summary["thirds"] = _third_summaries(window_values)
        metrics[name] = {
            "mrstft_log_magnitude_error": {
                "value": mrstft,
                "direction": "lower_is_better",
                "fft_sizes": scales,
            },
            "windowed_spectral_error": window_summary,
            "normalized_correlation": {
                "value": _normalized_correlation(reference, value),
                "direction": "higher_is_better",
            },
            "si_sdr_db": {
                "value": _si_sdr(reference, value),
                "direction": "higher_is_better",
                "perfect_cap_db": PERFECT_SI_SDR_DB,
            },
            "bounded_alignment_diagnostic": _bounded_lag(
                reference,
                value,
                metadata["sample_rate"],
            ),
        }
    return {"metadata": metadata, "metrics": metrics}


def _relative_advantage(legacy: float, candidate: float, *, higher_is_better: bool) -> float:
    denominator = max(abs(legacy), abs(candidate), ABSOLUTE_TOLERANCE)
    return ((candidate - legacy) if higher_is_better else (legacy - candidate)) / denominator


def _format_delta_value(delta: float, unit: str) -> str:
    if unit == "dB":
        return f"{delta:+.3f} dB"
    if unit == "ms":
        return f"{delta:+.3f} ms"
    if unit == "correlation_points":
        return f"{delta:+.5f} correlation points"
    if unit == "score_points":
        return f"{delta:+.6f} score points"
    return f"{delta:+.6g}"


def _comparison_row(
    name: str,
    legacy: float,
    candidate: float,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    advantage = _relative_advantage(legacy, candidate, higher_is_better=higher_is_better)
    if math.isclose(legacy, candidate, rel_tol=0.0, abs_tol=ABSOLUTE_TOLERANCE):
        winner = "tie"
    elif advantage > 0.0:
        winner = "candidate"
    else:
        winner = "legacy"
    role, display_unit, display_kind = _COMPARISON_METADATA[name]
    absolute_delta = candidate - legacy
    return {
        "metric": name,
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "legacy": legacy,
        "candidate": candidate,
        "absolute_candidate_delta": absolute_delta,
        "candidate_relative_advantage": advantage,
        "metric_role": role,
        "display": {
            "kind": display_kind,
            "unit": display_unit,
            "candidate_delta": absolute_delta,
            "headline": _format_delta_value(absolute_delta, display_unit),
        },
        "winner": winner,
    }


def _normalize_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add current display metadata to an existing schema-v1 comparison row."""
    name = str(row["metric"])
    legacy = float(row["legacy"])
    candidate = float(row["candidate"])
    direction = str(row["direction"])
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ObjectiveMediaError(f"objective comparison {name!r} has an invalid direction")
    if not all(math.isfinite(value) for value in (legacy, candidate)):
        raise ObjectiveMediaError(f"objective comparison {name!r} contains non-finite values")
    role, display_unit, display_kind = _COMPARISON_METADATA.get(
        name,
        ("diagnostic", "metric_units", "absolute_delta"),
    )
    normalized = dict(row)
    normalized["legacy"] = legacy
    normalized["candidate"] = candidate
    normalized["absolute_candidate_delta"] = candidate - legacy
    normalized.setdefault(
        "candidate_relative_advantage",
        _relative_advantage(
            legacy,
            candidate,
            higher_is_better=direction == "higher_is_better",
        ),
    )
    normalized["candidate_relative_advantage"] = float(
        normalized["candidate_relative_advantage"]
    )
    if not math.isfinite(normalized["candidate_relative_advantage"]):
        raise ObjectiveMediaError(
            f"objective comparison {name!r} has a non-finite relative advantage"
        )
    normalized["metric_role"] = role
    normalized["display"] = {
        "kind": display_kind,
        "unit": display_unit,
        "candidate_delta": candidate - legacy,
        "headline": _format_delta_value(candidate - legacy, display_unit),
    }
    if "winner" not in normalized:
        advantage = normalized["candidate_relative_advantage"]
        normalized["winner"] = (
            "tie"
            if math.isclose(
                legacy,
                candidate,
                rel_tol=0.0,
                abs_tol=ABSOLUTE_TOLERANCE,
            )
            else ("candidate" if advantage > 0.0 else "legacy")
        )
    return normalized


def _display_delta(row: dict[str, Any]) -> str:
    normalized = _normalize_comparison_row(row)
    return str(normalized["display"]["headline"])


def _display_decision_fraction(row: dict[str, Any]) -> str:
    normalized = _normalize_comparison_row(row)
    if normalized["metric_role"] == "diagnostic":
        return "diagnostic only"
    return f"{normalized['candidate_relative_advantage']:+.3%}"


def _paired_comparisons(video: dict[str, Any], audio: dict[str, Any] | None) -> list[dict[str, Any]]:
    legacy_video = video["metrics"]["legacy"]
    candidate_video = video["metrics"]["candidate"]
    rows = [
        _comparison_row("video_ms_ssim", legacy_video["ms_ssim"]["mean"], candidate_video["ms_ssim"]["mean"], higher_is_better=True),
        _comparison_row("video_psnr_db", legacy_video["psnr_db"]["mean"], candidate_video["psnr_db"]["mean"], higher_is_better=True),
        _comparison_row("video_temporal_derivative_error", legacy_video["temporal_derivative_error"]["mean"], candidate_video["temporal_derivative_error"]["mean"], higher_is_better=False),
        _comparison_row("video_motion_weighted_detail_error", legacy_video["motion_weighted_detail_error"]["mean"], candidate_video["motion_weighted_detail_error"]["mean"], higher_is_better=False),
        _comparison_row("video_worst_frame_ms_ssim", legacy_video["ms_ssim"]["worst_value"], candidate_video["ms_ssim"]["worst_value"], higher_is_better=True),
    ]
    if audio is not None:
        legacy_audio = audio["metrics"]["legacy"]
        candidate_audio = audio["metrics"]["candidate"]
        rows.extend(
            (
                _comparison_row(
                    "audio_mrstft_log_magnitude_error",
                    legacy_audio["mrstft_log_magnitude_error"]["value"],
                    candidate_audio["mrstft_log_magnitude_error"]["value"],
                    higher_is_better=False,
                ),
                _comparison_row(
                    "audio_normalized_correlation",
                    legacy_audio["normalized_correlation"]["value"],
                    candidate_audio["normalized_correlation"]["value"],
                    higher_is_better=True,
                ),
                _comparison_row(
                    "audio_si_sdr_db",
                    legacy_audio["si_sdr_db"]["value"],
                    candidate_audio["si_sdr_db"]["value"],
                    higher_is_better=True,
                ),
                _comparison_row(
                    "audio_worst_window_spectral_error",
                    legacy_audio["windowed_spectral_error"]["worst_value"],
                    candidate_audio["windowed_spectral_error"]["worst_value"],
                    higher_is_better=False,
                ),
                _comparison_row(
                    "audio_absolute_bounded_lag_ms",
                    abs(legacy_audio["bounded_alignment_diagnostic"]["lag_ms"]),
                    abs(candidate_audio["bounded_alignment_diagnostic"]["lag_ms"]),
                    higher_is_better=False,
                ),
            )
        )
    return rows


def _verdict(comparisons: list[dict[str, Any]], *, audio_present: bool) -> dict[str, Any]:
    by_name = {row["metric"]: row for row in comparisons}
    primary_names = [
        "video_ms_ssim",
        "video_temporal_derivative_error",
        "video_motion_weighted_detail_error",
    ]
    guardrail_names = ["video_worst_frame_ms_ssim"]
    if audio_present:
        primary_names.append("audio_mrstft_log_magnitude_error")
        guardrail_names.extend(
            ("audio_worst_window_spectral_error", "audio_absolute_bounded_lag_ms")
        )

    def decide(role: str) -> bool:
        sign = 1.0 if role == "candidate" else -1.0
        primary = [sign * by_name[name]["candidate_relative_advantage"] for name in primary_names]
        guardrails = [sign * by_name[name]["candidate_relative_advantage"] for name in guardrail_names]
        return (
            any(value >= PRIMARY_IMPROVEMENT_FRACTION for value in primary)
            and all(value >= -MATERIAL_REGRESSION_FRACTION for value in primary)
            and all(value >= -WORST_CASE_REGRESSION_FRACTION for value in guardrails)
        )

    candidate_favored = decide("candidate")
    legacy_favored = decide("legacy")
    if candidate_favored and not legacy_favored:
        verdict = "candidate_favored"
    elif legacy_favored and not candidate_favored:
        verdict = "legacy_favored"
    else:
        verdict = "mixed_or_inconclusive"
    return {
        "value": verdict,
        "rule": {
            "primary_metrics": primary_names,
            "worst_case_guardrails": guardrail_names,
            "minimum_primary_relative_improvement": PRIMARY_IMPROVEMENT_FRACTION,
            "maximum_primary_relative_regression": MATERIAL_REGRESSION_FRACTION,
            "maximum_worst_case_relative_regression": WORST_CASE_REGRESSION_FRACTION,
            "description": (
                "A role is favored only when at least one primary metric improves materially, "
                "no primary metric materially regresses, and no worst-case guardrail regresses beyond its limit."
            ),
        },
    }


def _paired_block_bootstrap(
    legacy: list[float],
    candidate: list[float],
    *,
    higher_is_better: bool,
    draws: int = 512,
) -> dict[str, Any]:
    if len(legacy) != len(candidate) or not legacy:
        raise ObjectiveMediaError("paired bootstrap requires equal non-empty temporal series")
    differences = [
        (candidate_value - legacy_value) if higher_is_better else (legacy_value - candidate_value)
        for legacy_value, candidate_value in zip(legacy, candidate, strict=True)
    ]
    block_size = max(1, round(math.sqrt(len(differences))))
    generator = random.Random(0)
    estimates: list[float] = []
    for _ in range(draws):
        sampled: list[float] = []
        while len(sampled) < len(differences):
            start = generator.randrange(len(differences))
            sampled.extend(differences[(start + offset) % len(differences)] for offset in range(block_size))
        estimates.append(statistics.fmean(sampled[: len(differences)]))
    return {
        "method": "paired_circular_temporal_block_bootstrap",
        "draws": draws,
        "block_size": block_size,
        "mean_advantage": statistics.fmean(differences),
        "confidence_interval_95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "positive_fraction": sum(value > 0.0 for value in estimates) / draws,
        "random_seed": 0,
    }


@lru_cache(maxsize=1)
def inspect_optional_backends() -> dict[str, Any]:
    try:
        lpips_installed = importlib.util.find_spec("lpips") is not None
    except (ImportError, ValueError):
        lpips_installed = False
    ffmpeg = shutil.which("ffmpeg")
    libvmaf_available = False
    if ffmpeg is not None:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            libvmaf_available = result.returncode == 0 and "libvmaf" in result.stdout
        except (OSError, subprocess.SubprocessError):
            libvmaf_available = False
    visqol = shutil.which("visqol")
    return {
        "lpips": {
            "package_detected": lpips_installed,
            "executed": False,
            "reason": (
                "LPIPS is not instantiated automatically because the official AlexNet trunk can request pretrained weights; "
                "the dependency-free metric panel remains authoritative until a local-weight adapter is configured."
            ),
        },
        "vmaf": {
            "ffmpeg_detected": ffmpeg is not None,
            "libvmaf_filter_detected": libvmaf_available,
            "executed": False,
            "reason": "VMAF is an optional encoded/raw-video forensic backend and is not applied to in-memory decoded tensors.",
        },
        "visqol": {
            "binary_detected": visqol is not None,
            "executed": False,
            "reason": (
                "ViSQOL remains an optional file/API forensic backend; general-audio mode requires 48 kHz and downmixes to mono."
            ),
        },
    }


def evaluate_objective_media(
    reference_video: torch.Tensor,
    legacy_video: torch.Tensor,
    candidate_video: torch.Tensor,
    *,
    fps: float,
    benchmark_id: str,
    seed: int | None,
    provenance: dict[str, Any],
    reference_audio: Any = None,
    legacy_audio: Any = None,
    candidate_audio: Any = None,
    chunk_size: int = 4,
) -> dict[str, Any]:
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ObjectiveMediaError("benchmark_id must be a non-empty string")
    if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ObjectiveMediaError("fps must be positive and finite")
    if not isinstance(provenance, dict):
        raise ObjectiveMediaError("provenance must be a JSON object")
    _validate_provenance(provenance)
    audio_values = (reference_audio, legacy_audio, candidate_audio)
    if any(value is not None for value in audio_values) and not all(value is not None for value in audio_values):
        raise ObjectiveMediaError("audio comparison requires reference, legacy, and candidate AUDIO inputs together")
    video = _video_metrics(reference_video, legacy_video, candidate_video, chunk_size=chunk_size)
    video["metadata"]["fps"] = float(fps)
    video["metadata"]["duration_seconds"] = video["metadata"]["frame_count"] / float(fps)
    audio = _audio_metrics(reference_audio, legacy_audio, candidate_audio) if reference_audio is not None else None
    comparisons = _paired_comparisons(video, audio)
    uncertainty = {
        "video_ms_ssim": _paired_block_bootstrap(
            video["metrics"]["legacy"]["ms_ssim"]["values"],
            video["metrics"]["candidate"]["ms_ssim"]["values"],
            higher_is_better=True,
        ),
        "video_temporal_derivative_error": _paired_block_bootstrap(
            video["metrics"]["legacy"]["temporal_derivative_error"]["values"],
            video["metrics"]["candidate"]["temporal_derivative_error"]["values"],
            higher_is_better=False,
        ),
        "video_motion_weighted_detail_error": _paired_block_bootstrap(
            video["metrics"]["legacy"]["motion_weighted_detail_error"]["values"],
            video["metrics"]["candidate"]["motion_weighted_detail_error"]["values"],
            higher_is_better=False,
        ),
    }
    compatibility = {
        "declared": provenance.get("compatibility", {}),
        "frame_count": video["metadata"]["frame_count"],
        "resolution": [video["metadata"]["width"], video["metadata"]["height"]],
        "fps": float(fps),
        "audio": None if audio is None else {
            "sample_rate": audio["metadata"]["sample_rate"],
            "channels": audio["metadata"]["channels"],
            "samples": audio["metadata"]["samples"],
        },
    }
    signature = _canonical_json(compatibility)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "spectrum_h3_objective_media_comparison",
        "benchmark_id": benchmark_id.strip(),
        "seed": seed,
        "scientific_reference": "full native MiniMax H3 decoded output with Spectrum bypassed",
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
        "verdict": _verdict(comparisons, audio_present=audio is not None),
        "optional_backends": inspect_optional_backends(),
        "boundaries": {
            "media_stage": "decoded tensors before video encoding or audio muxing",
            "hidden_space_evidence_is_separate": True,
            "production_defaults_changed": False,
            "transformer_calls": 0,
            "raw_media_persisted": False,
        },
    }


def _trim_paths(paths: Iterable[Path], keep: int) -> None:
    def order(path: Path) -> tuple[int, str]:
        try:
            return path.stat().st_mtime_ns, path.name
        except OSError:
            return 0, path.name

    ordered = sorted(paths, key=order)
    for path in ordered[: max(0, len(ordered) - keep)]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _trim_report_groups(root: Path, keep: int) -> None:
    runs_directory = root / "runs"
    if not runs_directory.exists():
        return

    def order(path: Path) -> tuple[int, str]:
        modified = 0
        try:
            for child in path.iterdir():
                try:
                    modified = max(modified, child.stat().st_mtime_ns)
                except OSError:
                    continue
        except OSError:
            pass
        return modified, path.name

    groups = sorted((path for path in runs_directory.iterdir() if path.is_dir()), key=order)
    aggregate_directory = root / "aggregates"
    for directory in groups[: max(0, len(groups) - keep)]:
        shutil.rmtree(directory, ignore_errors=True)
        for suffix in (".json", ".md"):
            try:
                (aggregate_directory / f"{directory.name}{suffix}").unlink()
            except FileNotFoundError:
                pass


def _metric_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["metric"]: _normalize_comparison_row(row)
        for row in report["comparisons"]
    }


def _independent_case_bootstrap(values: list[float], *, draws: int = 2000) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "available": False,
            "reason": "at least two independent complete triads are required",
        }
    generator = random.Random(0)
    estimates = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return {
        "available": True,
        "method": "independent_complete_triad_bootstrap",
        "draws": draws,
        "confidence_interval_95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "positive_fraction": sum(value > 0.0 for value in estimates) / draws,
        "random_seed": 0,
    }


def aggregate_objective_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ObjectiveMediaError("at least one objective-media report is required")
    signature = reports[0].get("compatibility_signature")
    if any(report.get("compatibility_signature") != signature for report in reports):
        raise ObjectiveMediaError("cannot aggregate incompatible objective-media reports")
    identities = {(report.get("benchmark_id"), report.get("seed")) for report in reports}
    if len(identities) != len(reports):
        raise ObjectiveMediaError("duplicate benchmark_id/seed evidence cannot be aggregated")
    metric_names = list(_metric_map(reports[0]))
    aggregate_metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        rows = [_metric_map(report)[metric_name] for report in reports]
        advantages = [float(row["candidate_relative_advantage"]) for row in rows]
        absolute_deltas = [float(row["absolute_candidate_delta"]) for row in rows]
        first = rows[0]
        aggregate_metrics[metric_name] = {
            "direction": first["direction"],
            "metric_role": first["metric_role"],
            "display": {
                "kind": first["display"]["kind"],
                "unit": first["display"]["unit"],
            },
            "mean_candidate_relative_advantage": statistics.fmean(advantages),
            "median_candidate_relative_advantage": statistics.median(advantages),
            "mean_absolute_candidate_delta": statistics.fmean(absolute_deltas),
            "median_absolute_candidate_delta": statistics.median(absolute_deltas),
            "wins": sum(value > ABSOLUTE_TOLERANCE for value in advantages),
            "losses": sum(value < -ABSOLUTE_TOLERANCE for value in advantages),
            "ties": sum(abs(value) <= ABSOLUTE_TOLERANCE for value in advantages),
            "worst_regression": min(advantages),
            "per_case": advantages,
            "per_case_absolute_candidate_delta": absolute_deltas,
            "independent_case_bootstrap": _independent_case_bootstrap(advantages),
        }
    verdicts = [report["verdict"]["value"] for report in reports]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "spectrum_h3_objective_media_aggregate",
        "group_id": reports[0]["group_id"],
        "compatibility": reports[0]["compatibility"],
        "independent_case_count": len(reports),
        "benchmark_ids": [report["benchmark_id"] for report in reports],
        "seeds": [report.get("seed") for report in reports],
        "per_case_verdicts": verdicts,
        "verdict_counts": {value: verdicts.count(value) for value in sorted(set(verdicts))},
        "metrics": aggregate_metrics,
        "cross_validation": "none; aggregation is by independent complete triads",
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Spectrum H3 objective decoded-media comparison",
        "",
        f"- Benchmark: `{report['benchmark_id']}`",
        f"- Seed: `{report.get('seed')}`",
        f"- Compatibility group: `{report['group_id']}`",
        f"- Verdict: **{report['verdict']['value']}**",
        "- Reference: full native MiniMax H3 decoded output with Spectrum bypassed",
        "- Media stage: decoded IMAGE/AUDIO tensors before encoding and muxing",
        "",
        "## Pairwise panel",
        "",
        "| Metric | Role | Direction | Legacy | Candidate | Human delta | Decision-relative | Winner |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["comparisons"]:
        row = _normalize_comparison_row(row)
        lines.append(
            f"| {row['metric']} | {row['metric_role']} | {row['direction']} | "
            f"{row['legacy']:.8g} | {row['candidate']:.8g} | {_display_delta(row)} | "
            f"{_display_decision_fraction(row)} | {row['winner']} |"
        )
    lines.extend(
        (
            "",
            "## Decision rule",
            "",
            report["verdict"]["rule"]["description"],
            "",
            (
                f"Primary improvement threshold: {PRIMARY_IMPROVEMENT_FRACTION:.1%}; "
                f"primary regression guardrail: {MATERIAL_REGRESSION_FRACTION:.1%}; "
                f"worst-case guardrail: {WORST_CASE_REGRESSION_FRACTION:.1%}."
            ),
            "",
            (
                "Raw per-frame and per-window series, temporal block-bootstrap intervals, provenance, "
                "and optional-backend availability are in the sibling JSON report."
            ),
        )
    )
    return "\n".join(lines) + "\n"


def _aggregate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Spectrum H3 objective decoded-media aggregate",
        "",
        f"- Compatibility group: `{report['group_id']}`",
        f"- Independent triads: **{report['independent_case_count']}**",
        f"- Verdict counts: `{json.dumps(report['verdict_counts'], sort_keys=True)}`",
        "",
        "| Metric | Role | Mean human delta | Median human delta | Decision-relative mean / median / worst | Wins | Losses | Ties |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"].items():
        display_row = {
            "metric": name,
            "legacy": 0.0,
            "candidate": metric["mean_absolute_candidate_delta"],
            "direction": metric["direction"],
            "candidate_relative_advantage": metric[
                "mean_candidate_relative_advantage"
            ],
        }
        median_row = {
            **display_row,
            "candidate": metric["median_absolute_candidate_delta"],
            "candidate_relative_advantage": metric[
                "median_candidate_relative_advantage"
            ],
        }
        decision_summary = (
            "diagnostic only"
            if metric["metric_role"] == "diagnostic"
            else (
                f"{metric['mean_candidate_relative_advantage']:+.3%} / "
                f"{metric['median_candidate_relative_advantage']:+.3%} / "
                f"{metric['worst_regression']:+.3%}"
            )
        )
        lines.append(
            f"| {name} | {metric['metric_role']} | {_display_delta(display_row)} | "
            f"{_display_delta(median_row)} | {decision_summary} | {metric['wins']} | "
            f"{metric['losses']} | {metric['ties']} |"
        )
    return "\n".join(lines) + "\n"


def persist_objective_report(
    report: dict[str, Any],
    *,
    root: Path | None = None,
) -> PersistedObjectiveReport:
    root = Path(root) if root is not None else default_store_root()
    group_id = str(report["group_id"])
    safe_benchmark = "".join(character if character.isalnum() or character in "-_" else "_" for character in report["benchmark_id"])
    digest = hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()[:12]
    group_directory = root / "runs" / group_id
    json_path = group_directory / f"{safe_benchmark}.{digest}.json"
    markdown_path = group_directory / f"{safe_benchmark}.{digest}.md"
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write_text(markdown_path, _report_markdown(report))
    _trim_paths(list(group_directory.glob("*.json")), MAX_REPORTS_PER_GROUP)
    _trim_paths(list(group_directory.glob("*.md")), MAX_REPORTS_PER_GROUP)
    loaded: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for path in sorted(group_directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("kind") != "spectrum_h3_objective_media_comparison"
                or value.get("group_id") != group_id
                or value.get("compatibility_signature") != report["compatibility_signature"]
                or not isinstance(value.get("comparisons"), list)
            ):
                continue
            _metric_map(value)
        except (
            KeyError,
            ObjectiveMediaError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
        identity = (value.get("benchmark_id"), value.get("seed"))
        if identity in seen:
            continue
        seen.add(identity)
        loaded.append(value)
        # Existing schema-v1 JSON remains authoritative and unchanged. Refresh
        # its human report through the additive metric-aware renderer so old
        # completed R/A/B evidence does not require regeneration.
        _atomic_write_text(path.with_suffix(".md"), _report_markdown(value))
    aggregate = aggregate_objective_reports(loaded)
    aggregate_directory = root / "aggregates"
    aggregate_json_path = aggregate_directory / f"{group_id}.json"
    aggregate_markdown_path = aggregate_directory / f"{group_id}.md"
    _atomic_write_text(aggregate_json_path, json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write_text(aggregate_markdown_path, _aggregate_markdown(aggregate))
    _trim_report_groups(root, MAX_REPORT_GROUPS)
    _trim_paths(list(aggregate_directory.glob("*.json")), MAX_REPORT_GROUPS)
    _trim_paths(list(aggregate_directory.glob("*.md")), MAX_REPORT_GROUPS)
    return PersistedObjectiveReport(
        group_id=group_id,
        run_count=len(loaded),
        json_path=json_path,
        markdown_path=markdown_path,
        aggregate_json_path=aggregate_json_path,
        aggregate_markdown_path=aggregate_markdown_path,
    )


__all__ = [
    "ObjectiveMediaError",
    "PersistedObjectiveReport",
    "aggregate_objective_reports",
    "default_store_root",
    "evaluate_objective_media",
    "inspect_optional_backends",
    "persist_objective_report",
]
