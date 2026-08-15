from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


LOG = logging.getLogger(__name__)

EXTERNAL_PATCH_CONTRACTS_KEY = "spectrum_h3_external_patch_contracts"
EXTERNAL_PATCH_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"
EXTERNAL_PATCH_SCHEMA_VERSION = 1
EXTERNAL_PATCH_KIND = "text_activation_modulation"
EXTERNAL_PATCH_ARCHITECTURE = "minimax_h3"
EXTERNAL_PATCH_SCOPE = "native_mod_segments_tag_1_only"

_PROFILE_CACHE_LIMIT = 16
_PROFILE_CACHE: OrderedDict[tuple[Any, ...], "ExternalAwareProfile"] = OrderedDict()
_PROFILE_CACHE_LOCK = threading.RLock()
_RUNTIME_STATE_ATTR = "_spectrum_h3_external_patch_compat"
_CURRENT_DECISION_ATTR = "_spectrum_h3_external_patch_decision"

_INSTALLED = False
_ORIGINAL_OUTER_SAMPLE_WRAPPER = None
_ORIGINAL_PREDICT_NOISE_WRAPPER = None
_ORIGINAL_DIFFUSION_MODEL_WRAPPER = None
_ORIGINAL_PROFILE_LOOKUP = None
_ORIGINAL_START_RUN = None
_ORIGINAL_END_RUN = None
_ORIGINAL_BEGIN_STEP = None
_ORIGINAL_FINALIZE_STEP = None
_ORIGINAL_ABORT_STEP = None
_ORIGINAL_CREATE_ROLLBACK_SNAPSHOT = None
_ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT = None
_ORIGINAL_DEBUG_SUMMARY = None


class ExternalPatchContractError(ValueError):
    """Declared external forecast-compatibility metadata is invalid."""


@dataclass(frozen=True, slots=True)
class ExternalPatchDescriptor:
    schema_version: int
    provider: str
    kind: str
    architecture: str
    instance_id: str
    block_indices_0based: tuple[int, ...]
    model_block_count: int
    strength: float
    sigma_start: float
    sigma_end: float
    sigma_ramp: float
    token_weight_mode: str
    token_tail: float
    cond_only: bool
    scope: str

    @property
    def inert(self) -> bool:
        return self.strength == 0.0

    @property
    def affects_final_block(self) -> bool:
        return (self.model_block_count - 1) in self.block_indices_0based

    @property
    def has_hard_temporal_transition(self) -> bool:
        return bool(
            not self.inert
            and self.sigma_ramp == 0.0
            and (self.sigma_start != 0.0 or self.sigma_end != 1.0)
        )

    def active_at(self, normalized_sigma: float) -> bool:
        # Diff-Aid hard-window semantics are inclusive at both boundaries.
        return self.sigma_start <= normalized_sigma <= self.sigma_end

    @property
    def canonical(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.provider,
            self.kind,
            self.architecture,
            self.instance_id,
            self.block_indices_0based,
            self.model_block_count,
            self.strength,
            self.sigma_start,
            self.sigma_end,
            self.sigma_ramp,
            self.token_weight_mode,
            self.token_tail,
            self.cond_only,
            self.scope,
        )


@dataclass(frozen=True, slots=True)
class ParsedExternalPatchContracts:
    descriptors: tuple[ExternalPatchDescriptor, ...]
    canonical: tuple[tuple[Any, ...], ...]
    fingerprint: str

    @property
    def active_descriptors(self) -> tuple[ExternalPatchDescriptor, ...]:
        return tuple(value for value in self.descriptors if not value.inert)

    @property
    def hard_descriptors(self) -> tuple[ExternalPatchDescriptor, ...]:
        return tuple(value for value in self.descriptors if value.has_hard_temporal_transition)


@dataclass(frozen=True, slots=True)
class ExternalAwareProfile:
    """Duck-typed extension of ModelForecastabilityProfile with runtime-patch fields."""

    base: Any
    cache_key: tuple[Any, ...]
    patch_identity: str
    active_patch_count: int
    active_patch_keys: int
    profile_confidence: float
    aggregate_sensitivity: float
    patch_perturbation: float
    final_block_perturbation: float
    forecast_risk_prior: float
    build_seconds: float
    estimated_bytes: int
    transient_workspace_bytes: int
    recognized_runtime_patch_count: int
    runtime_patch_kinds: tuple[str, ...]
    runtime_patch_perturbation: float
    runtime_final_block_perturbation: float
    external_contract_fingerprint: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass(frozen=True, slots=True)
class _ExternalRollbackState:
    committed_active: tuple[bool, ...] | None
    committed_sigma: tuple[float, ...] | None
    committed_step_id: int | None


@dataclass(slots=True)
class _ExternalRunState:
    run_id: int
    parsed: ParsedExternalPatchContracts
    replay: bool
    contract_failure: str | None = None
    committed_active: tuple[bool, ...] | None = None
    committed_sigma: tuple[float, ...] | None = None
    committed_step_id: int | None = None
    pending_step_id: int | None = None
    pending_active: tuple[bool, ...] | None = None
    pending_sigma: tuple[float, ...] | None = None
    pending_transition_indices: tuple[int, ...] = ()
    transitions: int = 0
    forced_actuals: int = 0
    contract_failures: int = 0
    failed_safe: bool = False


@dataclass(slots=True)
class _RuntimeCompatState:
    parsed: ParsedExternalPatchContracts = field(
        default_factory=lambda: ParsedExternalPatchContracts((), (), "none")
    )
    contract_failure: str | None = None
    run: _ExternalRunState | None = None
    rollback_snapshots: dict[int, _ExternalRollbackState] = field(default_factory=dict)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExternalPatchContractError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ExternalPatchContractError(f"{name} must be finite")
    return resolved


def _required_string(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ExternalPatchContractError(f"{name} must be a non-empty string")
    return value.strip()


def _parse_descriptor(raw: Any, *, block_count: int, position: int) -> ExternalPatchDescriptor:
    if not isinstance(raw, dict):
        raise ExternalPatchContractError(f"contract[{position}] must be a dictionary")
    schema = raw.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ExternalPatchContractError(f"contract[{position}].schema_version must be an integer")
    if schema != EXTERNAL_PATCH_SCHEMA_VERSION:
        raise ExternalPatchContractError(
            f"contract[{position}] uses unsupported schema_version={schema}"
        )
    provider = _required_string(raw, "provider")
    kind = _required_string(raw, "kind")
    architecture = _required_string(raw, "architecture")
    instance_id = _required_string(raw, "instance_id")
    scope = _required_string(raw, "scope")
    if kind != EXTERNAL_PATCH_KIND:
        raise ExternalPatchContractError(f"contract[{position}] uses unsupported kind={kind!r}")
    if architecture != EXTERNAL_PATCH_ARCHITECTURE:
        raise ExternalPatchContractError(
            f"contract[{position}] architecture={architecture!r} is not MiniMax H3"
        )
    if scope != EXTERNAL_PATCH_SCOPE:
        raise ExternalPatchContractError(
            f"contract[{position}] uses unsupported activation scope={scope!r}"
        )

    declared_count = raw.get("model_block_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count <= 0:
        raise ExternalPatchContractError(f"contract[{position}].model_block_count must be a positive integer")
    if declared_count != int(block_count):
        raise ExternalPatchContractError(
            f"contract[{position}] declares {declared_count} blocks but detected H3 has {block_count}"
        )
    raw_indices = raw.get("block_indices_0based")
    if not isinstance(raw_indices, (tuple, list)) or not raw_indices:
        raise ExternalPatchContractError(
            f"contract[{position}].block_indices_0based must be a non-empty sequence"
        )
    indices: list[int] = []
    for index in raw_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ExternalPatchContractError(
                f"contract[{position}] block indices must be integers"
            )
        if index < 0 or index >= block_count:
            raise ExternalPatchContractError(
                f"contract[{position}] block index {index} is outside 0..{block_count - 1}"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ExternalPatchContractError(f"contract[{position}] contains duplicate block indices")

    strength = _finite_number(raw.get("strength"), name=f"contract[{position}].strength")
    sigma_start = _finite_number(raw.get("sigma_start"), name=f"contract[{position}].sigma_start")
    sigma_end = _finite_number(raw.get("sigma_end"), name=f"contract[{position}].sigma_end")
    sigma_ramp = _finite_number(raw.get("sigma_ramp"), name=f"contract[{position}].sigma_ramp")
    token_tail = _finite_number(raw.get("token_tail"), name=f"contract[{position}].token_tail")
    if not 0.0 <= sigma_start <= sigma_end <= 1.0:
        raise ExternalPatchContractError(
            f"contract[{position}] requires 0 <= sigma_start <= sigma_end <= 1"
        )
    if sigma_ramp < 0.0:
        raise ExternalPatchContractError(f"contract[{position}].sigma_ramp must be >= 0")
    if not 0.0 <= token_tail <= 1.0:
        raise ExternalPatchContractError(f"contract[{position}].token_tail must be in [0, 1]")
    token_weight_mode = _required_string(raw, "token_weight_mode")
    if token_weight_mode not in {"none", "linear", "exponential"}:
        raise ExternalPatchContractError(
            f"contract[{position}] has unsupported token_weight_mode={token_weight_mode!r}"
        )
    cond_only = raw.get("cond_only")
    if not isinstance(cond_only, bool):
        raise ExternalPatchContractError(f"contract[{position}].cond_only must be a boolean")

    return ExternalPatchDescriptor(
        schema_version=schema,
        provider=provider,
        kind=kind,
        architecture=architecture,
        instance_id=instance_id,
        block_indices_0based=tuple(indices),
        model_block_count=declared_count,
        strength=strength,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        sigma_ramp=sigma_ramp,
        token_weight_mode=token_weight_mode,
        token_tail=token_tail,
        cond_only=cond_only,
        scope=scope,
    )


def parse_external_patch_contracts(
    model_options: dict[str, Any] | None,
    *,
    block_count: int,
) -> ParsedExternalPatchContracts:
    options = model_options or {}
    raw = options.get(EXTERNAL_PATCH_CONTRACTS_KEY)
    if raw is None:
        return ParsedExternalPatchContracts((), (), "none")
    if not isinstance(raw, (tuple, list)):
        raise ExternalPatchContractError(
            f"{EXTERNAL_PATCH_CONTRACTS_KEY} must be a sequence of descriptors"
        )
    descriptors = tuple(
        _parse_descriptor(value, block_count=int(block_count), position=index)
        for index, value in enumerate(raw)
    )
    instance_ids = [value.instance_id for value in descriptors]
    if len(set(instance_ids)) != len(instance_ids):
        raise ExternalPatchContractError("external patch instance_id values must be unique")
    canonical = tuple(value.canonical for value in descriptors)
    fingerprint = hashlib.sha256(
        repr(canonical).encode("utf-8", "backslashreplace")
    ).hexdigest()
    return ParsedExternalPatchContracts(descriptors, canonical, fingerprint)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return high
    return max(low, min(high, float(value)))


def _risk_from_perturbation(value: float) -> float:
    return _clamp(math.log1p(8.0 * max(0.0, float(value))) / math.log(9.0))


def _runtime_structural_perturbation(
    parsed: ParsedExternalPatchContracts,
) -> tuple[float, float]:
    all_log = 0.0
    final_log = 0.0
    for descriptor in parsed.active_descriptors:
        magnitude = abs(descriptor.strength)
        if magnitude == 0.0:
            continue
        for block_index in descriptor.block_indices_0based:
            weight = 1.0 if block_index == descriptor.model_block_count - 1 else 0.45
            all_log += weight * math.log1p(magnitude)
            if block_index == descriptor.model_block_count - 1:
                final_log += math.log1p(magnitude)
    return math.expm1(all_log), math.expm1(final_log)


def _external_profile_from_base(
    base_lookup: Any,
    parsed: ParsedExternalPatchContracts,
    *,
    adjustment_started: float,
) -> ExternalAwareProfile:
    base = base_lookup.profile
    runtime_count = len(parsed.active_descriptors)
    runtime_perturbation, runtime_final = _runtime_structural_perturbation(parsed)
    combined_perturbation = math.expm1(
        math.log1p(max(0.0, float(base.patch_perturbation)))
        + math.log1p(runtime_perturbation)
    )
    combined_final = math.expm1(
        math.log1p(max(0.0, float(base.final_block_perturbation)))
        + math.log1p(runtime_final)
    )

    old_patch_risk = _risk_from_perturbation(base.patch_perturbation)
    old_final_risk = _risk_from_perturbation(base.final_block_perturbation)
    base_sensitivity = _clamp(
        (
            float(base.aggregate_sensitivity)
            - 0.35 * old_patch_risk
            - 0.30 * old_final_risk
        )
        / 0.35
    )
    new_patch_risk = _risk_from_perturbation(combined_perturbation)
    new_final_risk = _risk_from_perturbation(combined_final)
    aggregate_sensitivity = _clamp(
        0.35 * base_sensitivity + 0.35 * new_patch_risk + 0.30 * new_final_risk
    )

    old_active = int(base.active_patch_count)
    old_recognized = int(base.recognized_lora_count)
    old_coverage = old_recognized / max(1, old_active)
    if old_active:
        base_coverage = _clamp((float(base.profile_confidence) - 0.45 * old_coverage) / 0.55)
    else:
        base_coverage = _clamp(base.profile_confidence)
    active_patch_count = old_active + runtime_count
    recognized_coverage = (old_recognized + runtime_count) / max(1, active_patch_count)
    profile_confidence = _clamp(
        0.55 * base_coverage + 0.45 * recognized_coverage
        if active_patch_count
        else base_coverage
    )
    unknown_risk = _clamp(int(base.unknown_patch_count) / max(1, active_patch_count))
    forecast_risk_prior = _clamp(
        profile_confidence * aggregate_sensitivity
        + (1.0 - profile_confidence) * (0.35 + 0.35 * unknown_risk)
    )
    cache_key = tuple(base.cache_key) + (
        "external_patch_contracts_v1",
        parsed.fingerprint,
    )
    kinds = tuple(dict.fromkeys(value.kind for value in parsed.active_descriptors))
    adjustment_elapsed = max(0.0, time.perf_counter() - adjustment_started)
    return ExternalAwareProfile(
        base=base,
        cache_key=cache_key,
        patch_identity=f"{base.patch_identity}:external:{parsed.fingerprint[:16]}",
        active_patch_count=active_patch_count,
        active_patch_keys=int(base.active_patch_keys) + runtime_count,
        profile_confidence=profile_confidence,
        aggregate_sensitivity=aggregate_sensitivity,
        patch_perturbation=combined_perturbation,
        final_block_perturbation=combined_final,
        forecast_risk_prior=forecast_risk_prior,
        build_seconds=(0.0 if base_lookup.cache_hit else float(base.build_seconds)) + adjustment_elapsed,
        estimated_bytes=int(base.estimated_bytes) + 256 * len(parsed.descriptors),
        transient_workspace_bytes=int(base.transient_workspace_bytes),
        recognized_runtime_patch_count=runtime_count,
        runtime_patch_kinds=kinds,
        runtime_patch_perturbation=runtime_perturbation,
        runtime_final_block_perturbation=runtime_final,
        external_contract_fingerprint=parsed.fingerprint,
    )


def get_model_forecastability_profile_with_external_patches(model_patcher: Any) -> Any:
    assert _ORIGINAL_PROFILE_LOOKUP is not None
    started = time.perf_counter()
    base_lookup = _ORIGINAL_PROFILE_LOOKUP(model_patcher)
    from . import minimax_h3
    from . import model_aware

    inner, _ = minimax_h3.locate_minimax_h3_inner(model_patcher)
    block_count = len(getattr(inner, "blocks", ())) if inner is not None else 0
    parsed = parse_external_patch_contracts(
        getattr(model_patcher, "model_options", None),
        block_count=block_count,
    )
    if not parsed.descriptors:
        return base_lookup
    effective_key = tuple(base_lookup.profile.cache_key) + (
        "external_patch_contracts_v1",
        parsed.fingerprint,
    )
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(effective_key)
        if cached is not None:
            _PROFILE_CACHE.move_to_end(effective_key)
            return model_aware.ProfileLookup(cached, True, time.perf_counter() - started)
        profile = _external_profile_from_base(
            base_lookup,
            parsed,
            adjustment_started=started,
        )
        _PROFILE_CACHE[effective_key] = profile
        _PROFILE_CACHE.move_to_end(effective_key)
        while len(_PROFILE_CACHE) > _PROFILE_CACHE_LIMIT:
            _PROFILE_CACHE.popitem(last=False)
    return model_aware.ProfileLookup(profile, False, time.perf_counter() - started)


def clear_external_profile_cache() -> None:
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()


def _compat_state(runtime: Any) -> _RuntimeCompatState:
    state = getattr(runtime, _RUNTIME_STATE_ATTR, None)
    if not isinstance(state, _RuntimeCompatState):
        state = _RuntimeCompatState()
        setattr(runtime, _RUNTIME_STATE_ATTR, state)
    return state


def configure_runtime_external_patches(
    runtime: Any,
    model_options: dict[str, Any] | None,
    *,
    block_count: int,
) -> None:
    state = _compat_state(runtime)
    state.run = None
    state.rollback_snapshots.clear()
    try:
        state.parsed = parse_external_patch_contracts(
            model_options,
            block_count=block_count,
        )
        state.contract_failure = None
    except ExternalPatchContractError as exc:
        state.parsed = ParsedExternalPatchContracts((), (), "invalid")
        state.contract_failure = str(exc)

    if runtime.config.debug and state.contract_failure is None:
        for descriptor in state.parsed.active_descriptors:
            LOG.warning(
                "Spectrum H3 external patch profile provider=%s instance=%s kind=%s "
                "strength=%.3f blocks=%s final_block=%s sigma_window=%.3f..%.3f "
                "sigma_ramp=%.3f",
                descriptor.provider,
                descriptor.instance_id,
                descriptor.kind,
                descriptor.strength,
                ",".join(str(index + 1) for index in descriptor.block_indices_0based),
                descriptor.affects_final_block,
                descriptor.sigma_start,
                descriptor.sigma_end,
                descriptor.sigma_ramp,
            )


def _runtime_entries(
    transformer_options: dict[str, Any],
    parsed: ParsedExternalPatchContracts,
) -> tuple[float, ...]:
    raw = transformer_options.get(EXTERNAL_PATCH_RUNTIME_KEY)
    if not parsed.descriptors:
        if raw not in (None, (), []):
            raise ExternalPatchContractError(
                "external patch runtime state exists without a declared static contract"
            )
        return ()
    if not isinstance(raw, (tuple, list)):
        raise ExternalPatchContractError(
            f"{EXTERNAL_PATCH_RUNTIME_KEY} is missing or is not a sequence"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ExternalPatchContractError(f"runtime_state[{position}] must be a dictionary")
        schema = value.get("schema_version")
        if isinstance(schema, bool) or not isinstance(schema, int) or schema != EXTERNAL_PATCH_SCHEMA_VERSION:
            raise ExternalPatchContractError(
                f"runtime_state[{position}] has unsupported schema_version"
            )
        provider = _required_string(value, "provider")
        instance_id = _required_string(value, "instance_id")
        if instance_id in by_id:
            raise ExternalPatchContractError(
                f"duplicate runtime state for external patch instance {instance_id!r}"
            )
        by_id[instance_id] = value
        by_id[instance_id]["__provider_checked"] = provider

    normalized: list[float] = []
    expected_ids = {value.instance_id for value in parsed.descriptors}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise ExternalPatchContractError(
            f"external patch runtime/static instance mismatch missing={missing} extra={extra}"
        )
    for descriptor in parsed.descriptors:
        value = by_id[descriptor.instance_id]
        if value["__provider_checked"] != descriptor.provider:
            raise ExternalPatchContractError(
                f"external patch provider changed for instance {descriptor.instance_id!r}"
            )
        sigma = _finite_number(
            value.get("normalized_sigma"),
            name=f"runtime_state[{descriptor.instance_id}].normalized_sigma",
        )
        if not 0.0 <= sigma <= 1.0:
            raise ExternalPatchContractError(
                f"runtime normalized sigma for {descriptor.instance_id!r} is outside [0, 1]"
            )
        normalized.append(sigma)
    return tuple(normalized)


def _set_effective_actual(runtime: Any, reason: str) -> None:
    step = getattr(runtime, "_step", None)
    if step is None:
        raise RuntimeError("external patch transition arrived outside an active Spectrum step")
    step.mode = "actual"
    step.reason = str(reason)
    step.adaptive_recompute = False
    step.bootstrap_forecast = False
    step.model_aware_decision = None
    step.model_aware_forced_actual = False
    decision = getattr(runtime, _CURRENT_DECISION_ATTR, None)
    if isinstance(decision, dict):
        decision["actual"] = True
        decision["reason"] = str(reason)


def _fail_safe_current_step(runtime: Any, run_state: _ExternalRunState, reason: str) -> None:
    if not run_state.failed_safe:
        run_state.contract_failures += 1
        run_state.failed_safe = True
        LOG.warning(
            "Spectrum H3 external patch compatibility failed; forcing native/all-actual "
            "sampling for this run: %s",
            reason,
        )
    runtime._disable_forecasting(f"external patch compatibility metadata invalid: {reason}")
    _set_effective_actual(runtime, "external patch compatibility all-actual fallback")


def observe_external_patch_runtime(
    runtime: Any,
    transformer_options: dict[str, Any],
) -> bool:
    state = _compat_state(runtime)
    run_state = state.run
    if run_state is None:
        return False
    if run_state.failed_safe:
        return getattr(getattr(runtime, "_step", None), "mode", None) == "actual"
    if run_state.contract_failure is not None:
        _fail_safe_current_step(runtime, run_state, run_state.contract_failure)
        return True

    try:
        normalized = _runtime_entries(transformer_options, run_state.parsed)
    except ExternalPatchContractError as exc:
        if run_state.replay:
            from .runtime import OfflineReplayAbort

            raise OfflineReplayAbort(
                f"external patch runtime state is invalid during offline replay: {exc}"
            ) from exc
        _fail_safe_current_step(runtime, run_state, str(exc))
        return True

    if run_state.replay or not run_state.parsed.descriptors:
        return False
    active = tuple(
        descriptor.active_at(normalized[index])
        for index, descriptor in enumerate(run_state.parsed.descriptors)
    )
    step = getattr(runtime, "_step", None)
    if step is None:
        raise RuntimeError("external patch runtime state arrived without an active step")
    step_id = int(step.step_id)

    if run_state.pending_step_id == step_id:
        if run_state.pending_active != active or run_state.pending_sigma != normalized:
            _fail_safe_current_step(
                runtime,
                run_state,
                "external patch state changed across repeated model calls in one solver step",
            )
            return True
        return step.mode == "actual"

    transition_indices: tuple[int, ...] = ()
    if run_state.committed_active is not None:
        transition_indices = tuple(
            index
            for index, descriptor in enumerate(run_state.parsed.descriptors)
            if descriptor.has_hard_temporal_transition
            and run_state.committed_active[index] != active[index]
        )
    run_state.pending_step_id = step_id
    run_state.pending_active = active
    run_state.pending_sigma = normalized
    run_state.pending_transition_indices = transition_indices

    if transition_indices:
        run_state.transitions += len(transition_indices)
        if step.mode == "forecast":
            reason = "external patch hard sigma transition"
            _set_effective_actual(runtime, reason)
            run_state.forced_actuals += 1
            action = "force_actual"
        else:
            action = "already_actual"
        if runtime.config.debug:
            details = []
            for index in transition_indices:
                descriptor = run_state.parsed.descriptors[index]
                details.append(
                    f"{descriptor.provider}:{descriptor.instance_id}:"
                    f"{str(run_state.committed_active[index]).lower()}->"
                    f"{str(active[index]).lower()}@{normalized[index]:.6f}"
                )
            LOG.warning(
                "Spectrum H3 external patch transition step=%s transitions=%s action=%s",
                step_id,
                ",".join(details),
                action,
            )
    return step.mode == "actual"


def _runtime_start_run(self, *args, **kwargs):
    assert _ORIGINAL_START_RUN is not None
    run_id = _ORIGINAL_START_RUN(self, *args, **kwargs)
    compat = _compat_state(self)
    compat.rollback_snapshots.clear()
    compat.run = _ExternalRunState(
        run_id=int(run_id),
        parsed=compat.parsed,
        replay=getattr(self, "_offline_phase", None) == "replay",
        contract_failure=compat.contract_failure,
    )
    if compat.contract_failure is not None:
        compat.run.contract_failures = 1
        compat.run.failed_safe = True
        self.disable_forecasting_for_run(
            f"external patch compatibility metadata invalid: {compat.contract_failure}"
        )
        LOG.warning(
            "Spectrum H3 external patch contract rejected; forcing native/all-actual "
            "sampling for this run: %s",
            compat.contract_failure,
        )
    return run_id


def _runtime_end_run(self, run_id: int) -> None:
    assert _ORIGINAL_END_RUN is not None
    _ORIGINAL_END_RUN(self, run_id)
    compat = _compat_state(self)
    compat.run = None
    compat.rollback_snapshots.clear()
    setattr(self, _CURRENT_DECISION_ATTR, None)


def _runtime_begin_step(self, timestep: Any) -> dict[str, Any]:
    assert _ORIGINAL_BEGIN_STEP is not None
    decision = _ORIGINAL_BEGIN_STEP(self, timestep)
    setattr(self, _CURRENT_DECISION_ATTR, decision)
    return decision


def _runtime_finalize_step(self, run_id: int, step_id: int) -> None:
    assert _ORIGINAL_FINALIZE_STEP is not None
    compat = _compat_state(self)
    run_state = compat.run
    pending_matches = bool(
        run_state is not None and run_state.pending_step_id == int(step_id)
    )
    _ORIGINAL_FINALIZE_STEP(self, run_id, step_id)
    if pending_matches and run_state is not None and not run_state.replay:
        run_state.committed_active = run_state.pending_active
        run_state.committed_sigma = run_state.pending_sigma
        run_state.committed_step_id = int(step_id)
        run_state.pending_step_id = None
        run_state.pending_active = None
        run_state.pending_sigma = None
        run_state.pending_transition_indices = ()
    setattr(self, _CURRENT_DECISION_ATTR, None)


def _runtime_abort_step(self, run_id: int, step_id: int) -> None:
    assert _ORIGINAL_ABORT_STEP is not None
    _ORIGINAL_ABORT_STEP(self, run_id, step_id)
    run_state = _compat_state(self).run
    if run_state is not None and run_state.pending_step_id == int(step_id):
        run_state.pending_step_id = None
        run_state.pending_active = None
        run_state.pending_sigma = None
        run_state.pending_transition_indices = ()
    setattr(self, _CURRENT_DECISION_ATTR, None)


def _runtime_create_rollback_snapshot(self):
    assert _ORIGINAL_CREATE_ROLLBACK_SNAPSHOT is not None
    snapshot = _ORIGINAL_CREATE_ROLLBACK_SNAPSHOT(self)
    compat = _compat_state(self)
    run_state = compat.run
    if run_state is not None:
        compat.rollback_snapshots[id(snapshot)] = _ExternalRollbackState(
            committed_active=run_state.committed_active,
            committed_sigma=run_state.committed_sigma,
            committed_step_id=run_state.committed_step_id,
        )
    return snapshot


def _runtime_restore_rollback_snapshot(self, snapshot) -> None:
    assert _ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT is not None
    compat = _compat_state(self)
    external = compat.rollback_snapshots.pop(id(snapshot), None)
    _ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT(self, snapshot)
    run_state = compat.run
    if run_state is not None and external is not None:
        run_state.committed_active = external.committed_active
        run_state.committed_sigma = external.committed_sigma
        run_state.committed_step_id = external.committed_step_id
        run_state.pending_step_id = None
        run_state.pending_active = None
        run_state.pending_sigma = None
        run_state.pending_transition_indices = ()
    setattr(self, _CURRENT_DECISION_ATTR, None)


def _runtime_debug_summary(self) -> str:
    assert _ORIGINAL_DEBUG_SUMMARY is not None
    base = _ORIGINAL_DEBUG_SUMMARY(self)
    compat = _compat_state(self)
    run_state = compat.run
    if run_state is None:
        return base
    profile = getattr(self, "model_profile", None)
    runtime_patch_count = int(getattr(profile, "recognized_runtime_patch_count", 0))
    runtime_perturbation = float(getattr(profile, "runtime_patch_perturbation", 0.0))
    runtime_final = float(getattr(profile, "runtime_final_block_perturbation", 0.0))
    kinds = getattr(profile, "runtime_patch_kinds", ())
    return (
        f"{base} external_patch_count={len(run_state.parsed.active_descriptors)} "
        f"external_patch_runtime_profile_count={runtime_patch_count} "
        f"external_patch_kinds={','.join(kinds) if kinds else '-'} "
        f"external_patch_runtime_perturbation={runtime_perturbation:.6f} "
        f"external_patch_final_perturbation={runtime_final:.6f} "
        f"external_patch_transitions={run_state.transitions} "
        f"external_patch_forced_actuals={run_state.forced_actuals} "
        f"external_patch_contract_failures={run_state.contract_failures}"
    )


def _outer_sample_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    assert _ORIGINAL_OUTER_SAMPLE_WRAPPER is not None
    from . import minimax_h3
    from . import sampling

    guider = executor.class_obj
    binding = sampling._binding_from_model_options(getattr(guider, "model_options", None))
    if binding is not None:
        model_options = getattr(guider, "model_options", None) or {}
        inner, _ = minimax_h3.locate_minimax_h3_inner(getattr(guider, "model_patcher", None))
        block_count = len(getattr(inner, "blocks", ())) if inner is not None else 0
        configure_runtime_external_patches(
            binding.runtime,
            model_options,
            block_count=block_count,
        )
    return _ORIGINAL_OUTER_SAMPLE_WRAPPER(
        executor,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
    )


def _log_effective_step(runtime: Any, decision: dict[str, Any]) -> None:
    from . import sampling

    if not runtime.config.debug:
        return
    sampling.LOG.warning(
        "Spectrum H3 step run_id=%s step=%s coordinate=%.6f decision=%s reason=%s history=%s window=%.3f",
        decision["run_id"],
        decision["step_id"],
        decision["coordinate"],
        "actual" if decision["actual"] else "forecast",
        decision["reason"],
        runtime.prediction_history_length,
        runtime.stats.current_window,
    )
    model_decision = runtime.active_model_aware_decision
    if model_decision is None:
        return
    audio_gain = model_decision.audio_correction_telemetry
    video_gain = model_decision.video_correction_telemetry
    sampling.LOG.warning(
        "Spectrum H3 model-aware step=%s trajectory_risk=%.6f model_risk=%.6f "
        "patch_risk=%.6f combined_risk=%.6f confidence=%.6f horizon=%.3f "
        "degree=%s ridge=%.8f audio_blend=%.6f video_blend=%.6f "
        "audio_generic_projection=%.6f audio_raw_generic_gain=%.6f "
        "audio_generic_gain=%.6f audio_applied_gain=%.6f "
        "audio_generic_bound_active=%s "
        "video_generic_projection=%.6f video_raw_generic_gain=%.6f "
        "video_generic_gain=%.6f video_applied_gain=%.6f "
        "video_generic_bound_active=%s model_informed_correction=retired decision=%s",
        decision["step_id"],
        model_decision.trajectory_risk,
        model_decision.model_risk,
        model_decision.patch_risk,
        model_decision.combined_risk,
        model_decision.confidence,
        model_decision.forecast_horizon,
        model_decision.degree,
        model_decision.ridge_lambda,
        model_decision.audio_blend_weight,
        model_decision.video_blend_weight,
        audio_gain.residual_projection,
        audio_gain.raw_generic_gain,
        audio_gain.generic_gain,
        model_decision.audio_correction_gain,
        audio_gain.generic_bound_active,
        video_gain.residual_projection,
        video_gain.raw_generic_gain,
        video_gain.generic_gain,
        model_decision.video_correction_gain,
        video_gain.generic_bound_active,
        "ACTUAL" if decision["actual"] else "FORECAST",
    )


def _predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    """Native predict-noise transaction with post-wrapper effective-mode logging.

    A deterministic external transition can promote the current step only after
    Diff-Aid publishes its per-call normalized sigma. The shared decision dict is
    mutated by that promotion while executor() is active, so every operation after
    the model call consumes the effective mode, including ER-SDE ownership.
    """
    from . import sampling

    guider = executor.class_obj
    binding = sampling._binding_from_model_options(getattr(guider, "model_options", None))
    if (
        binding is None
        or binding.runtime.active_run_id is None
        or not binding.runtime.supported_sampler
    ):
        return executor(x, timestep, model_options or {}, seed)

    if "multigpu_clones" in (model_options or {}):
        if binding.runtime.config.debug:
            sampling.LOG.warning(
                "Spectrum H3 native fallback: multi-GPU parallel model calls are not transactionally supported"
            )
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    decision = runtime.begin_step(timestep)

    def execute_attempt(attempt_decision: dict[str, Any]):
        patched = sampling.copy_model_options_with_step(model_options, runtime, attempt_decision)
        return executor(x, timestep, patched, seed)

    def consume_er_sde_increment(
        result: torch.Tensor,
        attempt_decision: dict[str, Any],
    ) -> torch.Tensor:
        transformer_options = (model_options or {}).get("transformer_options") or {}
        tracker = transformer_options.get(sampling.ER_SDE_TRACKER_KEY)
        if not isinstance(tracker, sampling.ERSDEStochasticTracker):
            return result
        descriptor = None
        try:
            descriptor = runtime.describe_current_er_sde_step(
                int(attempt_decision["run_id"]),
                int(attempt_decision["step_id"]),
            )
            return tracker.consume(result, descriptor)
        except sampling.ERSDETrackingError as exc:
            tracker.clear()
            if descriptor is not None and descriptor.mode == "replay":
                raise sampling.OfflineReplayAbort(
                    f"ER-SDE stochastic-state compensation failed during replay: {exc}"
                ) from exc
            if not bool(attempt_decision["actual"]):
                raise sampling.ForecastRetryActual(
                    f"ER-SDE stochastic-state compensation failed: {exc}"
                ) from exc
            if runtime.config.debug:
                sampling.LOG.warning(
                    "Spectrum H3 ER-SDE tracker discarded on state-aware actual "
                    "step=%s reason=%s",
                    attempt_decision["step_id"],
                    exc,
                )
            return result

    try:
        try:
            result = execute_attempt(decision)
            _log_effective_step(runtime, decision)
            runtime.log_offline_transition(
                "actual_executor_return" if decision["actual"] else "forecast_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
            )
            result = consume_er_sde_increment(result, decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
        except sampling.ForecastRetryActual as retry:
            runtime.prepare_actual_retry(
                decision["run_id"], decision["step_id"], str(retry)
            )
            retry_decision = dict(decision)
            retry_decision["actual"] = True
            retry_decision["reason"] = f"forecast transaction retry: {retry}"
            if runtime.config.debug:
                sampling.LOG.warning(
                    "Spectrum H3 forecast retry run_id=%s step=%s reason=%s",
                    decision["run_id"],
                    decision["step_id"],
                    retry,
                )
            result = execute_attempt(retry_decision)
            _log_effective_step(runtime, retry_decision)
            runtime.log_offline_transition(
                "actual_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
                retry=True,
            )
            result = consume_er_sde_increment(result, retry_decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


def _diffusion_model_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    assert _ORIGINAL_DIFFUSION_MODEL_WRAPPER is not None
    from . import sampling

    options = transformer_options or {}
    runtime = options.get(sampling.RUNTIME_KEY)
    run_id = options.get(sampling.RUN_ID_KEY)
    step_id = options.get(sampling.STEP_ID_KEY)
    if runtime is not None and run_id is not None and step_id is not None:
        effective_actual = observe_external_patch_runtime(runtime, options)
        if effective_actual:
            # transformer_options is invocation-local (Spectrum and Diff-Aid both
            # copy it before mutation). Keep downstream metadata consistent with
            # the effective runtime transaction.
            options[sampling.ACTUAL_KEY] = True
            step = getattr(runtime, "_step", None)
            if step is not None:
                options[sampling.REASON_KEY] = step.reason
    return _ORIGINAL_DIFFUSION_MODEL_WRAPPER(
        executor,
        x,
        timestep,
        context,
        options,
        minimax_payload=minimax_payload,
        **kwargs,
    )


def install_external_patch_compat() -> None:
    global _INSTALLED
    global _ORIGINAL_OUTER_SAMPLE_WRAPPER, _ORIGINAL_PREDICT_NOISE_WRAPPER
    global _ORIGINAL_DIFFUSION_MODEL_WRAPPER, _ORIGINAL_PROFILE_LOOKUP
    global _ORIGINAL_START_RUN, _ORIGINAL_END_RUN, _ORIGINAL_BEGIN_STEP
    global _ORIGINAL_FINALIZE_STEP, _ORIGINAL_ABORT_STEP
    global _ORIGINAL_CREATE_ROLLBACK_SNAPSHOT, _ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT
    global _ORIGINAL_DEBUG_SUMMARY
    if _INSTALLED:
        return

    from . import minimax_h3
    from . import model_aware
    from . import runtime as runtime_module
    from . import sampling

    _ORIGINAL_OUTER_SAMPLE_WRAPPER = sampling.outer_sample_wrapper
    _ORIGINAL_PREDICT_NOISE_WRAPPER = sampling.predict_noise_wrapper
    _ORIGINAL_DIFFUSION_MODEL_WRAPPER = minimax_h3.diffusion_model_wrapper
    _ORIGINAL_PROFILE_LOOKUP = model_aware.get_model_forecastability_profile
    _ORIGINAL_START_RUN = runtime_module.SpectrumH3Runtime.start_run
    _ORIGINAL_END_RUN = runtime_module.SpectrumH3Runtime.end_run
    _ORIGINAL_BEGIN_STEP = runtime_module.SpectrumH3Runtime.begin_step
    _ORIGINAL_FINALIZE_STEP = runtime_module.SpectrumH3Runtime.finalize_step
    _ORIGINAL_ABORT_STEP = runtime_module.SpectrumH3Runtime.abort_step
    _ORIGINAL_CREATE_ROLLBACK_SNAPSHOT = runtime_module.SpectrumH3Runtime.create_rollback_snapshot
    _ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT = runtime_module.SpectrumH3Runtime.restore_rollback_snapshot
    _ORIGINAL_DEBUG_SUMMARY = runtime_module.SpectrumH3Runtime.debug_summary

    sampling.outer_sample_wrapper = _outer_sample_wrapper
    sampling.predict_noise_wrapper = _predict_noise_wrapper
    minimax_h3.diffusion_model_wrapper = _diffusion_model_wrapper
    model_aware.get_model_forecastability_profile = get_model_forecastability_profile_with_external_patches
    sampling.get_model_forecastability_profile = get_model_forecastability_profile_with_external_patches
    runtime_module.SpectrumH3Runtime.start_run = _runtime_start_run
    runtime_module.SpectrumH3Runtime.end_run = _runtime_end_run
    runtime_module.SpectrumH3Runtime.begin_step = _runtime_begin_step
    runtime_module.SpectrumH3Runtime.finalize_step = _runtime_finalize_step
    runtime_module.SpectrumH3Runtime.abort_step = _runtime_abort_step
    runtime_module.SpectrumH3Runtime.create_rollback_snapshot = _runtime_create_rollback_snapshot
    runtime_module.SpectrumH3Runtime.restore_rollback_snapshot = _runtime_restore_rollback_snapshot
    runtime_module.SpectrumH3Runtime.debug_summary = _runtime_debug_summary
    _INSTALLED = True


__all__ = [
    "EXTERNAL_PATCH_ARCHITECTURE",
    "EXTERNAL_PATCH_CONTRACTS_KEY",
    "EXTERNAL_PATCH_KIND",
    "EXTERNAL_PATCH_RUNTIME_KEY",
    "EXTERNAL_PATCH_SCHEMA_VERSION",
    "EXTERNAL_PATCH_SCOPE",
    "ExternalAwareProfile",
    "ExternalPatchContractError",
    "ExternalPatchDescriptor",
    "ParsedExternalPatchContracts",
    "clear_external_profile_cache",
    "configure_runtime_external_patches",
    "get_model_forecastability_profile_with_external_patches",
    "install_external_patch_compat",
    "observe_external_patch_runtime",
    "parse_external_patch_contracts",
]
