from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

_PROFILE_CACHE_LIMIT = 16
_BASE_SAMPLE_ELEMENTS = 4096
_EXACT_LORA_FACTOR_ELEMENTS = 4_000_000
_EXACT_LORA_COMBINED_RANK = 512


@dataclass(frozen=True, slots=True)
class ModelForecastabilityProfile:
    cache_key: tuple[Any, ...]
    base_model_identity: str
    patch_identity: str
    active_patch_count: int
    active_patch_keys: int
    recognized_lora_count: int
    unknown_patch_count: int
    sampled_base_tensors: int
    profile_confidence: float
    aggregate_sensitivity: float
    patch_perturbation: float
    final_block_perturbation: float
    audio_sensitivity: float
    video_sensitivity: float
    forecast_risk_prior: float
    build_seconds: float
    estimated_bytes: int
    transient_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class ProfileLookup:
    profile: ModelForecastabilityProfile
    cache_hit: bool
    lookup_seconds: float


@dataclass(frozen=True, slots=True)
class ModelAwareForecastDecision:
    trajectory_risk: float
    model_risk: float
    patch_risk: float
    combined_risk: float
    confidence: float
    ridge_lambda: float
    degree: int
    audio_blend_weight: float
    video_blend_weight: float
    audio_correction_gain: float
    video_correction_gain: float
    forecast_horizon: float
    force_actual: bool


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    forecast_ratio: float
    curvature_ratio: float
    fit_condition: float
    audio_projection: float
    video_projection: float
    model_corrected_ratio: float
    generic_corrected_ratio: float


_PROFILE_CACHE: OrderedDict[tuple[Any, ...], ModelForecastabilityProfile] = OrderedDict()
_PROFILE_CACHE_LOCK = threading.RLock()


def clear_model_profile_cache() -> None:
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return high
    return max(low, min(high, float(value)))


def _locate_inner(model_patcher: Any) -> Any | None:
    model = getattr(model_patcher, "model", None)
    inner = getattr(model, "diffusion_model", None)
    if inner is not None:
        return inner
    return getattr(model_patcher, "diffusion_model", None)


def _architecture_signature(inner: Any) -> tuple[Any, ...]:
    blocks = getattr(inner, "blocks", ())
    return (
        type(inner).__module__,
        type(inner).__name__,
        len(blocks),
        int(getattr(inner, "hidden_size", 0)),
        tuple(int(v) for v in getattr(inner, "patch_size", ())),
        int(getattr(inner, "latents_dim", 0)),
        int(getattr(inner, "audio_latents_dim", 0)),
        bool(getattr(inner, "use_adaln_curves", False)),
    )


def _tensor_signature(value: Any) -> tuple[Any, ...]:
    if not torch.is_tensor(value):
        return (type(value).__module__, type(value).__name__)
    return (
        tuple(int(v) for v in value.shape),
        str(value.dtype),
        str(value.device),
        int(value.data_ptr()) if value.device.type != "meta" else 0,
    )


def _adapter_signature(adapter: Any) -> tuple[Any, ...]:
    weights = getattr(adapter, "weights", ())
    if not isinstance(weights, (tuple, list)):
        weights = (weights,)
    return (
        type(adapter).__module__,
        type(adapter).__name__,
        tuple(_tensor_signature(value) for value in weights),
    )


def _patch_payload_signature(payload: Any) -> tuple[Any, ...]:
    if hasattr(payload, "weights"):
        return _adapter_signature(payload)
    if isinstance(payload, (tuple, list)):
        values = []
        for value in payload[:8]:
            if torch.is_tensor(value):
                values.append(_tensor_signature(value))
            elif isinstance(value, (tuple, list)):
                values.append(
                    tuple(
                        _tensor_signature(item)
                        if torch.is_tensor(item)
                        else (type(item).__module__, type(item).__name__, repr(item)[:96])
                        for item in value[:8]
                    )
                )
            else:
                values.append((type(value).__module__, type(value).__name__, repr(value)[:96]))
        return (type(payload).__name__, tuple(values))
    return (type(payload).__module__, type(payload).__name__, repr(payload)[:96])


def _patch_metadata_digest(model_patcher: Any) -> str:
    digest = hashlib.sha256()
    patches = getattr(model_patcher, "patches", {}) or {}
    for key in sorted(patches, key=str):
        digest.update(str(key).encode("utf-8", "backslashreplace"))
        for patch in patches[key] or ():
            if not isinstance(patch, (tuple, list)) or len(patch) < 5:
                signature = (type(patch).__module__, type(patch).__name__, repr(patch)[:96])
            else:
                strength, payload, strength_model, offset, function = patch[:5]
                signature = (
                    repr(strength),
                    repr(strength_model),
                    _patch_payload_signature(payload),
                    repr(offset)[:96],
                    None
                    if function is None
                    else (type(function).__module__, type(function).__name__, id(function)),
                )
            digest.update(repr(signature).encode("utf-8", "backslashreplace"))
    return digest.hexdigest()


def _injection_metadata(model_patcher: Any) -> tuple[tuple[Any, ...], list[tuple[str, Any, float]]]:
    descriptors: list[tuple[Any, ...]] = []
    adapters: list[tuple[str, Any, float]] = []
    seen_managers: set[int] = set()
    injections = getattr(model_patcher, "injections", {}) or {}
    for injection_key in sorted(injections, key=str):
        entries = injections[injection_key] or ()
        descriptors.append((str(injection_key), len(entries)))
        for entry in entries:
            functions = (getattr(entry, "inject", None), getattr(entry, "eject", None))
            for function in functions:
                closure = getattr(function, "__closure__", None) or ()
                for cell in closure:
                    try:
                        captured = cell.cell_contents
                    except ValueError:
                        continue
                    registered = getattr(captured, "adapters", None)
                    if not isinstance(registered, dict):
                        continue
                    manager_identity = id(captured)
                    if manager_identity in seen_managers:
                        continue
                    seen_managers.add(manager_identity)
                    for key, value in sorted(registered.items(), key=lambda item: str(item[0])):
                        if not isinstance(value, tuple) or len(value) != 2:
                            continue
                        adapter, strength = value
                        try:
                            strength_value = float(strength)
                        except (TypeError, ValueError):
                            strength_value = 1.0
                        adapters.append((str(key), adapter, strength_value))
                        descriptors.append(
                            (str(injection_key), str(key), strength_value, _adapter_signature(adapter))
                        )
    return tuple(descriptors), adapters


def _profile_cache_key(model_patcher: Any, inner: Any) -> tuple[Any, ...]:
    injection_signature, _ = _injection_metadata(model_patcher)
    return (
        str(getattr(model_patcher, "clone_base_uuid", "unknown-base")),
        str(getattr(model_patcher, "patches_uuid", "unknown-patches")),
        _patch_metadata_digest(model_patcher),
        _architecture_signature(inner),
        hashlib.sha256(repr(injection_signature).encode("utf-8", "backslashreplace")).hexdigest(),
    )


def _sample_tensor_rms(value: Any, *, limit: int = _BASE_SAMPLE_ELEMENTS) -> tuple[float | None, int]:
    if not torch.is_tensor(value) or not value.dtype.is_floating_point or value.device.type == "meta":
        return None, 0
    detached = value.detach()
    if not detached.is_contiguous() or detached.numel() == 0:
        return None, 0
    flat = detached.view(-1)
    stride = max(1, flat.numel() // max(1, int(limit)))
    sample = flat[::stride][:limit]
    try:
        rms = float(torch.sqrt(torch.mean(sample.to(torch.float32).square())).item())
    except (RuntimeError, TypeError):
        return None, 0
    if not math.isfinite(rms):
        return None, 0
    return rms, int(sample.numel())


def _operator_gain(weight: Any) -> tuple[float | None, int]:
    rms, samples = _sample_tensor_rms(weight)
    if rms is None or not torch.is_tensor(weight) or weight.ndim < 2:
        return None, samples
    in_features = math.prod(int(v) for v in weight.shape[1:])
    return rms * math.sqrt(max(1, in_features)), samples


def _get_base_weight(model_patcher: Any, key: str) -> Any | None:
    backup = (getattr(model_patcher, "backup", {}) or {}).get(key)
    if backup is not None:
        return getattr(backup, "weight", None)
    try:
        return model_patcher.get_model_object(key)
    except (AttributeError, KeyError, IndexError, RuntimeError, TypeError):
        return None


def _selected_base_keys(inner: Any) -> tuple[str, ...]:
    last = max(0, len(getattr(inner, "blocks", ())) - 1)
    prefix = f"diffusion_model.blocks.{last}"
    return (
        f"{prefix}.attn.qkv_proj.weight",
        f"{prefix}.attn.out_proj.weight",
        f"{prefix}.mlp.fc1.weight",
        f"{prefix}.mlp.fc2.weight",
        f"{prefix}.adaln_proj.linear.weight",
        "diffusion_model.final_layer.adaln_proj.linear.weight",
        "diffusion_model.final_layer.video_out.weight",
        "diffusion_model.final_layer.audio_out.weight",
    )


def _key_group(key: str, last_block_index: int) -> str:
    final_prefix = f"diffusion_model.blocks.{last_block_index}."
    if key.startswith(final_prefix):
        return "final_block"
    if key.startswith("diffusion_model.final_layer."):
        return "final_head"
    if key.startswith("diffusion_model.blocks."):
        return "transformer"
    if "time_embed" in key or "adaln" in key:
        return "conditioning"
    return "other"


def _classic_lora_factors(adapter: Any, strength: float) -> tuple[torch.Tensor, torch.Tensor] | None:
    if str(getattr(adapter, "name", "")).lower() != "lora":
        return None
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (tuple, list)) or len(weights) < 6:
        return None
    up, down, alpha, mid, dora_scale, reshape = weights[:6]
    if (
        not torch.is_tensor(up)
        or not torch.is_tensor(down)
        or mid is not None
        or dora_scale is not None
        or reshape is not None
    ):
        return None
    try:
        up_2d = up.detach().flatten(start_dim=1)
        down_2d = down.detach().flatten(start_dim=1)
    except RuntimeError:
        return None
    if up_2d.ndim != 2 or down_2d.ndim != 2 or up_2d.shape[1] != down_2d.shape[0]:
        return None
    scale = float(strength)
    if alpha is not None:
        scale *= float(alpha) / max(1, int(down_2d.shape[0]))
    return up_2d, down_2d * scale


def _combined_low_rank_norm(
    factors: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float | None, int]:
    if not factors:
        return None, 0
    device = factors[0][0].device
    if any(up.device != device or down.device != device for up, down in factors):
        return None, 0
    factor_elements = sum(up.numel() + down.numel() for up, down in factors)
    combined_rank = sum(int(up.shape[1]) for up, _ in factors)
    if (
        factor_elements > _EXACT_LORA_FACTOR_ELEMENTS
        or combined_rank > _EXACT_LORA_COMBINED_RANK
    ):
        bound = 0.0
        for up, down in factors:
            up_rms, _ = _sample_tensor_rms(up)
            down_rms, _ = _sample_tensor_rms(down)
            if up_rms is None or down_rms is None:
                return None, factor_elements
            bound += (
                up_rms
                * math.sqrt(up.numel())
                * down_rms
                * math.sqrt(down.numel())
            )
        return bound, factor_elements
    try:
        up = torch.cat([value.to(torch.float32) for value, _ in factors], dim=1)
        down = torch.cat([value.to(torch.float32) for _, value in factors], dim=0)
        gram_up = up.transpose(0, 1) @ up
        gram_down = down @ down.transpose(0, 1)
        squared = torch.sum(gram_up * gram_down.transpose(0, 1)).clamp_min(0.0)
        norm = float(torch.sqrt(squared).item())
    except (RuntimeError, TypeError):
        return None, factor_elements
    return (norm if math.isfinite(norm) else None), factor_elements


def _sample_diff_norm(value: Any) -> tuple[float | None, int]:
    if not torch.is_tensor(value):
        return None, 0
    rms, samples = _sample_tensor_rms(value)
    if rms is None:
        return None, samples
    return rms * math.sqrt(max(1, value.numel())), samples


def _build_profile(model_patcher: Any, inner: Any, cache_key: tuple[Any, ...]) -> ModelForecastabilityProfile:
    started = time.perf_counter()
    base_gains: dict[str, float] = {}
    sampled_base_tensors = 0
    sampled_values = 0
    for key in _selected_base_keys(inner):
        weight = _get_base_weight(model_patcher, key)
        gain, samples = _operator_gain(weight)
        sampled_values += samples
        if gain is not None:
            base_gains[key] = gain
            sampled_base_tensors += 1

    gain_values = sorted(value for value in base_gains.values() if value > 0.0)
    base_gain = gain_values[len(gain_values) // 2] if gain_values else 1.0
    audio_gain = base_gains.get("diffusion_model.final_layer.audio_out.weight", base_gain)
    video_gain = base_gains.get("diffusion_model.final_layer.video_out.weight", base_gain)
    stream_mean = max((audio_gain + video_gain) * 0.5, 1e-12)
    audio_sensitivity = _clamp(audio_gain / stream_mean, 0.25, 2.0)
    video_sensitivity = _clamp(video_gain / stream_mean, 0.25, 2.0)

    patches = getattr(model_patcher, "patches", {}) or {}
    _, injection_adapters = _injection_metadata(model_patcher)
    last_block_index = max(0, len(getattr(inner, "blocks", ())) - 1)
    active_patch_count = 0
    active_patch_keys: set[str] = set()
    recognized_lora_count = 0
    unknown_patch_count = 0
    patch_values: list[tuple[float, float]] = []
    final_values: list[tuple[float, float]] = []
    transient_workspace_bytes = sampled_values * 4

    for key in sorted(patches, key=str):
        patch_list = patches[key] or ()
        factors: list[tuple[torch.Tensor, torch.Tensor]] = []
        fallback_norm = 0.0
        fallback_known = False
        for patch in patch_list:
            if not isinstance(patch, (tuple, list)) or len(patch) < 5:
                active_patch_count += 1
                active_patch_keys.add(str(key))
                unknown_patch_count += 1
                continue
            strength, payload, strength_model, offset, function = patch[:5]
            numeric_strengths = True
            try:
                strength_value = float(strength)
                strength_model_value = float(strength_model)
            except (TypeError, ValueError):
                strength_value = 1.0
                strength_model_value = 1.0
                numeric_strengths = False
            if strength_value == 0.0:
                continue
            active_patch_count += 1
            active_patch_keys.add(str(key))
            precise = numeric_strengths
            if offset is not None or function is not None or not math.isclose(strength_model_value, 1.0):
                precise = False
            lora = _classic_lora_factors(payload, strength_value)
            if lora is not None and precise:
                factors.append(lora)
                recognized_lora_count += 1
                transient_workspace_bytes = max(
                    transient_workspace_bytes,
                    int(lora[0].shape[1] ** 2) * 8,
                )
                continue
            patch_type = payload[0] if isinstance(payload, tuple) and payload else None
            patch_data = payload[1] if isinstance(payload, tuple) and len(payload) > 1 else None
            if patch_type == "diff" and isinstance(patch_data, tuple) and patch_data:
                norm, _ = _sample_diff_norm(patch_data[0])
                if norm is not None:
                    fallback_norm += abs(strength_value) * norm
                    fallback_known = True
                    continue
            unknown_patch_count += 1

        low_rank_norm, factor_elements = _combined_low_rank_norm(factors)
        transient_workspace_bytes = max(
            transient_workspace_bytes,
            min(factor_elements, _EXACT_LORA_FACTOR_ELEMENTS) * 4,
        )
        patch_norm = (low_rank_norm or 0.0) + fallback_norm
        if patch_norm <= 0.0 and not fallback_known and not factors:
            continue
        base_weight = _get_base_weight(model_patcher, str(key))
        base_rms, samples = _sample_tensor_rms(base_weight)
        sampled_values += samples
        if base_rms is not None and torch.is_tensor(base_weight):
            denominator = base_rms * math.sqrt(max(1, base_weight.numel()))
        else:
            shape_numel = math.prod(int(v) for v in factors[0][0].shape[:1] + factors[0][1].shape[1:]) if factors else 1
            denominator = max(base_gain * math.sqrt(max(1, shape_numel)), 1e-12)
        relative = patch_norm / max(denominator, 1e-12)
        group = _key_group(str(key), last_block_index)
        group_weight = {
            "final_head": 1.20,
            "final_block": 1.00,
            "transformer": 0.45,
            "conditioning": 0.35,
            "other": 0.25,
        }[group]
        patch_values.append((relative, group_weight))
        if group in {"final_head", "final_block"}:
            final_values.append((relative, group_weight))

    for key, adapter, strength in injection_adapters:
        if strength == 0.0:
            continue
        active_patch_count += 1
        active_patch_keys.add(key)
        lora = _classic_lora_factors(adapter, strength)
        if lora is None:
            unknown_patch_count += 1
            continue
        norm, factor_elements = _combined_low_rank_norm([lora])
        if norm is None:
            unknown_patch_count += 1
            continue
        recognized_lora_count += 1
        group = _key_group(key, last_block_index)
        group_weight = 1.0 if group == "final_block" else 1.2 if group == "final_head" else 0.45
        up, down = lora
        denominator = max(base_gain * math.sqrt(max(1, up.shape[0] * down.shape[1])), 1e-12)
        relative = norm / denominator
        patch_values.append((relative, group_weight))
        if group in {"final_head", "final_block"}:
            final_values.append((relative, group_weight))
        transient_workspace_bytes = max(
            transient_workspace_bytes,
            min(factor_elements, _EXACT_LORA_FACTOR_ELEMENTS) * 4,
        )

    def aggregate(values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        numerator = sum(weight * math.log1p(max(0.0, value)) for value, weight in values)
        denominator = sum(weight for _, weight in values)
        return math.expm1(numerator / max(denominator, 1e-12))

    patch_perturbation = aggregate(patch_values)
    final_perturbation = aggregate(final_values)
    base_sensitivity = _clamp(math.log1p(max(base_gain, 0.0)) / math.log(8.0))
    patch_risk = _clamp(math.log1p(8.0 * patch_perturbation) / math.log(9.0))
    final_risk = _clamp(math.log1p(8.0 * final_perturbation) / math.log(9.0))
    aggregate_sensitivity = _clamp(0.35 * base_sensitivity + 0.35 * patch_risk + 0.30 * final_risk)
    coverage = recognized_lora_count / max(1, active_patch_count)
    base_coverage = sampled_base_tensors / max(1, len(_selected_base_keys(inner)))
    profile_confidence = _clamp(0.55 * base_coverage + 0.45 * coverage if active_patch_count else base_coverage)
    unknown_risk = _clamp(unknown_patch_count / max(1, active_patch_count))
    forecast_risk_prior = _clamp(
        profile_confidence * aggregate_sensitivity
        + (1.0 - profile_confidence) * (0.35 + 0.35 * unknown_risk)
    )
    patch_identity = str(getattr(model_patcher, "patches_uuid", "unknown-patches"))
    base_identity = f"{_architecture_signature(inner)[0]}.{_architecture_signature(inner)[1]}:{getattr(model_patcher, 'clone_base_uuid', 'unknown-base')}"
    elapsed = time.perf_counter() - started
    return ModelForecastabilityProfile(
        cache_key=cache_key,
        base_model_identity=base_identity,
        patch_identity=patch_identity,
        active_patch_count=active_patch_count,
        active_patch_keys=len(active_patch_keys),
        recognized_lora_count=recognized_lora_count,
        unknown_patch_count=unknown_patch_count,
        sampled_base_tensors=sampled_base_tensors,
        profile_confidence=profile_confidence,
        aggregate_sensitivity=aggregate_sensitivity,
        patch_perturbation=patch_perturbation,
        final_block_perturbation=final_perturbation,
        audio_sensitivity=audio_sensitivity,
        video_sensitivity=video_sensitivity,
        forecast_risk_prior=forecast_risk_prior,
        build_seconds=elapsed,
        # Samples and low-rank Gram matrices are temporary; the LRU retains
        # only this immutable scalar/string record.
        estimated_bytes=max(512, min(4096, 512 + 16 * len(cache_key))),
        transient_workspace_bytes=int(transient_workspace_bytes),
    )


def get_model_forecastability_profile(model_patcher: Any) -> ProfileLookup:
    started = time.perf_counter()
    inner = _locate_inner(model_patcher)
    if inner is None:
        raise TypeError("model patcher does not contain a diffusion model")
    cache_key = _profile_cache_key(model_patcher, inner)
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(cache_key)
        if cached is not None:
            _PROFILE_CACHE.move_to_end(cache_key)
            return ProfileLookup(cached, True, time.perf_counter() - started)
        profile = _build_profile(model_patcher, inner, cache_key)
        _PROFILE_CACHE[cache_key] = profile
        _PROFILE_CACHE.move_to_end(cache_key)
        while len(_PROFILE_CACHE) > _PROFILE_CACHE_LIMIT:
            _PROFILE_CACHE.popitem(last=False)
    return ProfileLookup(profile, False, time.perf_counter() - started)


class ModelAwareController:
    def __init__(self, mode: str, risk_threshold: float) -> None:
        if mode not in {"off", "schedule", "schedule_confidence", "full"}:
            raise ValueError("invalid model-aware mode")
        self.mode = mode
        self.risk_threshold = float(risk_threshold)
        self.profile: ModelForecastabilityProfile | None = None
        self.reset()

    def reset(self) -> None:
        self.anchor_count = 0
        self.forecast_ratio_ewma = 1.0
        self.curvature_ratio_ewma = 0.0
        self.fit_condition_ewma = 1.0
        self.audio_projection_ewma = 0.0
        self.video_projection_ewma = 0.0
        self.model_corrected_ratio_sum = 0.0
        self.generic_corrected_ratio_sum = 0.0
        self.ablation_count = 0

    def set_profile(self, profile: ModelForecastabilityProfile | None) -> None:
        self.profile = profile

    def snapshot(self) -> dict[str, float | int]:
        return {
            "anchor_count": self.anchor_count,
            "forecast_ratio_ewma": self.forecast_ratio_ewma,
            "curvature_ratio_ewma": self.curvature_ratio_ewma,
            "fit_condition_ewma": self.fit_condition_ewma,
            "audio_projection_ewma": self.audio_projection_ewma,
            "video_projection_ewma": self.video_projection_ewma,
            "model_corrected_ratio_sum": self.model_corrected_ratio_sum,
            "generic_corrected_ratio_sum": self.generic_corrected_ratio_sum,
            "ablation_count": self.ablation_count,
        }

    def restore(self, state: dict[str, float | int]) -> None:
        self.anchor_count = int(state["anchor_count"])
        self.forecast_ratio_ewma = float(state["forecast_ratio_ewma"])
        self.curvature_ratio_ewma = float(state["curvature_ratio_ewma"])
        self.fit_condition_ewma = float(state["fit_condition_ewma"])
        self.audio_projection_ewma = float(state["audio_projection_ewma"])
        self.video_projection_ewma = float(state["video_projection_ewma"])
        self.model_corrected_ratio_sum = float(state["model_corrected_ratio_sum"])
        self.generic_corrected_ratio_sum = float(state["generic_corrected_ratio_sum"])
        self.ablation_count = int(state["ablation_count"])

    def observe_anchor(self, evidence: AnchorEvidence) -> None:
        alpha = 0.5 if self.anchor_count < 2 else 0.3
        self.forecast_ratio_ewma = (1.0 - alpha) * self.forecast_ratio_ewma + alpha * _clamp(
            evidence.forecast_ratio, 0.0, 8.0
        )
        self.curvature_ratio_ewma = (1.0 - alpha) * self.curvature_ratio_ewma + alpha * _clamp(
            evidence.curvature_ratio, 0.0, 8.0
        )
        self.fit_condition_ewma = (1.0 - alpha) * self.fit_condition_ewma + alpha * max(
            1.0, min(float(evidence.fit_condition), 1e8)
        )
        self.audio_projection_ewma = (1.0 - alpha) * self.audio_projection_ewma + alpha * _clamp(
            evidence.audio_projection, -2.0, 2.0
        )
        self.video_projection_ewma = (1.0 - alpha) * self.video_projection_ewma + alpha * _clamp(
            evidence.video_projection, -2.0, 2.0
        )
        if math.isfinite(evidence.model_corrected_ratio) and math.isfinite(evidence.generic_corrected_ratio):
            self.model_corrected_ratio_sum += evidence.model_corrected_ratio
            self.generic_corrected_ratio_sum += evidence.generic_corrected_ratio
            self.ablation_count += 1
        self.anchor_count += 1

    def _risks(self, forecast_horizon: float) -> tuple[float, float, float, float, float]:
        profile = self.profile
        model_risk = profile.forecast_risk_prior if profile is not None else 0.5
        patch_risk = (
            _clamp(math.log1p(8.0 * profile.patch_perturbation) / math.log(9.0))
            if profile is not None
            else 0.5
        )
        residual_risk = _clamp((self.forecast_ratio_ewma - 0.5) / 1.5)
        curvature_risk = _clamp(self.curvature_ratio_ewma / 2.0)
        condition_risk = _clamp(math.log10(max(1.0, self.fit_condition_ewma)) / 6.0)
        horizon_risk = _clamp((max(1.0, forecast_horizon) - 1.0) / 2.0)
        trajectory_risk = _clamp(
            0.42 * residual_risk
            + 0.28 * curvature_risk
            + 0.18 * horizon_risk
            + 0.12 * condition_risk
        )
        reliability = min(1.0, self.anchor_count / 3.0)
        evidence_risk = max(residual_risk, 0.65 * curvature_risk)
        calibrated_model = (1.0 - 0.70 * reliability) * model_risk + 0.70 * reliability * min(
            1.0, 0.35 * model_risk + 0.65 * evidence_risk
        )
        combined = _clamp(
            0.68 * trajectory_risk
            + 0.22 * calibrated_model
            + 0.10 * patch_risk
            + 0.12 * trajectory_risk * calibrated_model
        )
        confidence = _clamp(1.0 - combined)
        return trajectory_risk, calibrated_model, patch_risk, combined, confidence

    def decision(
        self,
        *,
        forecast_horizon: float,
        history_length: int,
        configured_degree: int,
        configured_ridge_lambda: float,
        configured_audio_blend: float,
        configured_video_blend: float,
    ) -> ModelAwareForecastDecision:
        trajectory, model_risk, patch_risk, combined, confidence = self._risks(forecast_horizon)
        adaptive = self.mode in {"schedule_confidence", "full"}
        degree = max(1, min(int(configured_degree), max(1, int(history_length) - 1)))
        ridge = float(configured_ridge_lambda)
        audio_blend = float(configured_audio_blend)
        video_blend = float(configured_video_blend)
        if adaptive:
            if combined >= 0.72:
                degree = max(1, degree - 2)
            elif combined >= 0.48:
                degree = max(1, degree - 1)
            base_ridge = max(ridge, 1e-5)
            ridge = max(1e-6, min(10.0, base_ridge * (2.0 ** (4.0 * (combined - 0.35)))))
            spectral_scale = math.sqrt(confidence)
            audio_blend *= spectral_scale
            video_blend *= spectral_scale

        audio_correction = 0.0
        video_correction = 0.0
        if self.mode == "full" and self.anchor_count > 0 and self.profile is not None:
            horizon_scale = min(1.5, max(0.5, float(forecast_horizon)))
            audio_correction = _clamp(
                self.audio_projection_ewma
                * self.profile.audio_sensitivity
                * confidence
                * horizon_scale,
                -0.25,
                0.25,
            )
            video_correction = _clamp(
                self.video_projection_ewma
                * self.profile.video_sensitivity
                * confidence
                * horizon_scale,
                -0.25,
                0.25,
            )
        return ModelAwareForecastDecision(
            trajectory_risk=trajectory,
            model_risk=model_risk,
            patch_risk=patch_risk,
            combined_risk=combined,
            confidence=confidence,
            ridge_lambda=ridge,
            degree=degree,
            audio_blend_weight=audio_blend,
            video_blend_weight=video_blend,
            audio_correction_gain=audio_correction,
            video_correction_gain=video_correction,
            forecast_horizon=float(forecast_horizon),
            force_actual=combined >= self.risk_threshold,
        )

    def generic_correction_gains(
        self,
        decision: ModelAwareForecastDecision,
    ) -> tuple[float, float]:
        if self.mode != "full" or self.anchor_count == 0:
            return 0.0, 0.0
        horizon_scale = min(1.5, max(0.5, decision.forecast_horizon))
        scale = decision.confidence * horizon_scale
        return (
            _clamp(self.audio_projection_ewma * scale, -0.25, 0.25),
            _clamp(self.video_projection_ewma * scale, -0.25, 0.25),
        )

    @property
    def model_corrected_ratio_mean(self) -> float:
        return self.model_corrected_ratio_sum / self.ablation_count if self.ablation_count else 0.0

    @property
    def generic_corrected_ratio_mean(self) -> float:
        return self.generic_corrected_ratio_sum / self.ablation_count if self.ablation_count else 0.0


__all__ = [
    "AnchorEvidence",
    "ModelAwareController",
    "ModelAwareForecastDecision",
    "ModelForecastabilityProfile",
    "ProfileLookup",
    "clear_model_profile_cache",
    "get_model_forecastability_profile",
]
