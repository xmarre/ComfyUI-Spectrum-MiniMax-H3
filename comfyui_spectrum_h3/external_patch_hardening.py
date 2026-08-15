from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import external_patch_compat as compat


_INSTALLED = False
_ORIGINAL_EXTERNAL_PROFILE_BUILDER = None
_ORIGINAL_CLEAR_MODEL_PROFILE_CACHE = None


def _runtime_entries_without_mutation(
    transformer_options: dict[str, Any],
    parsed: compat.ParsedExternalPatchContracts,
) -> tuple[float, ...]:
    """Validate per-call state without mutating producer-owned dictionaries."""

    raw = transformer_options.get(compat.EXTERNAL_PATCH_RUNTIME_KEY)
    if not parsed.descriptors:
        if raw not in (None, (), []):
            raise compat.ExternalPatchContractError(
                "external patch runtime state exists without a declared static contract"
            )
        return ()
    if not isinstance(raw, (tuple, list)):
        raise compat.ExternalPatchContractError(
            f"{compat.EXTERNAL_PATCH_RUNTIME_KEY} is missing or is not a sequence"
        )

    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for position, value in enumerate(raw):
        if not isinstance(value, dict):
            raise compat.ExternalPatchContractError(
                f"runtime_state[{position}] must be a dictionary"
            )
        schema = value.get("schema_version")
        if (
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema != compat.EXTERNAL_PATCH_SCHEMA_VERSION
        ):
            raise compat.ExternalPatchContractError(
                f"runtime_state[{position}] has unsupported schema_version"
            )
        provider = compat._required_string(value, "provider")
        instance_id = compat._required_string(value, "instance_id")
        if instance_id in by_id:
            raise compat.ExternalPatchContractError(
                f"duplicate runtime state for external patch instance {instance_id!r}"
            )
        by_id[instance_id] = (provider, value)

    expected_ids = {value.instance_id for value in parsed.descriptors}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise compat.ExternalPatchContractError(
            f"external patch runtime/static instance mismatch missing={missing} extra={extra}"
        )

    normalized: list[float] = []
    for descriptor in parsed.descriptors:
        provider, value = by_id[descriptor.instance_id]
        if provider != descriptor.provider:
            raise compat.ExternalPatchContractError(
                f"external patch provider changed for instance {descriptor.instance_id!r}"
            )
        sigma = compat._finite_number(
            value.get("normalized_sigma"),
            name=f"runtime_state[{descriptor.instance_id}].normalized_sigma",
        )
        if not 0.0 <= sigma <= 1.0:
            raise compat.ExternalPatchContractError(
                f"runtime normalized sigma for {descriptor.instance_id!r} is outside [0, 1]"
            )
        normalized.append(sigma)
    return tuple(normalized)


def _profile_builder_with_runtime_key_semantics(*args, **kwargs):
    assert _ORIGINAL_EXTERNAL_PROFILE_BUILDER is not None
    profile = _ORIGINAL_EXTERNAL_PROFILE_BUILDER(*args, **kwargs)
    # active_patch_keys is the existing parameter-patch key count. Runtime
    # activation contracts have their own explicit counter and no parameter keys.
    return replace(profile, active_patch_keys=int(profile.base.active_patch_keys))


def get_model_forecastability_profile_fail_safe(model_patcher: Any) -> Any:
    """Keep malformed declared metadata from aborting model-aware setup.

    Runtime setup parses the same declaration before sampling and then disables
    forecasting for that run. Returning the ordinary base profile here avoids a
    premature exception while preserving the required all-actual fail-safe.
    """

    try:
        return compat.get_model_forecastability_profile_with_external_patches(
            model_patcher
        )
    except compat.ExternalPatchContractError:
        assert compat._ORIGINAL_PROFILE_LOOKUP is not None
        return compat._ORIGINAL_PROFILE_LOOKUP(model_patcher)


def _clear_all_model_profile_caches() -> None:
    assert _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE is not None
    _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE()
    compat.clear_external_profile_cache()


def install_external_patch_hardening() -> None:
    global _INSTALLED
    global _ORIGINAL_EXTERNAL_PROFILE_BUILDER, _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE
    if _INSTALLED:
        return

    from . import model_aware
    from . import sampling

    _ORIGINAL_EXTERNAL_PROFILE_BUILDER = compat._external_profile_from_base
    _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE = model_aware.clear_model_profile_cache

    compat._runtime_entries = _runtime_entries_without_mutation
    compat._external_profile_from_base = _profile_builder_with_runtime_key_semantics
    model_aware.get_model_forecastability_profile = (
        get_model_forecastability_profile_fail_safe
    )
    sampling.get_model_forecastability_profile = (
        get_model_forecastability_profile_fail_safe
    )
    model_aware.clear_model_profile_cache = _clear_all_model_profile_caches
    _INSTALLED = True


__all__ = [
    "get_model_forecastability_profile_fail_safe",
    "install_external_patch_hardening",
]
