from __future__ import annotations

import hashlib
import math
from typing import Any

from . import external_patch_compat as compat


VISUAL_PATCH_PROFILES_KEY = "spectrum_h3_visual_reference_patch_profiles"
VISUAL_PATCH_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
VISUAL_PATCH_SCHEMA_VERSION = 2
VISUAL_PATCH_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
VISUAL_PATCH_PROVIDER = "comfyui-flux2-untwisting-rope"
VISUAL_PATCH_KIND = "visual_reference_attention_modulation"
VISUAL_PATCH_ARCHITECTURE = "minimax_h3"
VISUAL_PATCH_SCOPES = frozenset(
    {"image_only", "image_and_video", "all_visual_including_continuum"}
)

_INSTALLED = False
_ORIGINAL_PARSE = None
_ORIGINAL_RUNTIME_ENTRIES = None


def _required_bool(mapping: dict[str, Any], name: str, *, position: int) -> bool:
    value = mapping.get(name)
    if not isinstance(value, bool):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].{name} must be a boolean"
        )
    return value


def _finite(mapping: dict[str, Any], name: str, *, position: int) -> float:
    return compat._finite_number(
        mapping.get(name),
        name=f"visual_profile[{position}].{name}",
    )


def _parse_visual_profile(
    raw: Any,
    *,
    block_count: int,
    position: int,
) -> tuple[compat.ExternalPatchDescriptor, tuple[Any, ...]]:
    if not isinstance(raw, dict):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] must be a dictionary"
        )

    schema = raw.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].schema_version must be an integer"
        )
    if schema not in VISUAL_PATCH_SUPPORTED_SCHEMA_VERSIONS:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] uses unsupported schema_version={schema}"
        )

    provider = compat._required_string(raw, "provider")
    kind = compat._required_string(raw, "kind")
    architecture = compat._required_string(raw, "architecture")
    instance_id = compat._required_string(raw, "instance_id")
    scope = compat._required_string(raw, "scope")
    if provider != VISUAL_PATCH_PROVIDER:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] uses unsupported provider={provider!r}"
        )
    if kind != VISUAL_PATCH_KIND:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] uses unsupported kind={kind!r}"
        )
    if architecture != VISUAL_PATCH_ARCHITECTURE:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] architecture={architecture!r} is not MiniMax H3"
        )
    if scope not in VISUAL_PATCH_SCOPES:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] uses unsupported scope={scope!r}"
        )

    declared_count = raw.get("model_block_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count <= 0:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].model_block_count must be a positive integer"
        )
    if declared_count != int(block_count):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] declares {declared_count} blocks but detected H3 has {block_count}"
        )

    raw_indices = raw.get("block_indices_0based")
    if not isinstance(raw_indices, (tuple, list)) or not raw_indices:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].block_indices_0based must be a non-empty sequence"
        )
    indices: list[int] = []
    for index in raw_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise compat.ExternalPatchContractError(
                f"visual_profile[{position}] block indices must be integers"
            )
        if index < 0 or index >= block_count:
            raise compat.ExternalPatchContractError(
                f"visual_profile[{position}] block index {index} is outside 0..{block_count - 1}"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] contains duplicate block indices"
        )

    strength = _finite(raw, "strength", position=position)
    if strength < 0.0:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].strength must be >= 0"
        )
    progress_start = _finite(raw, "progress_start", position=position)
    progress_end = _finite(raw, "progress_end", position=position)
    if not 0.0 <= progress_start <= progress_end <= 1.0:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}] requires 0 <= progress_start <= progress_end <= 1"
        )
    hard_start = _required_bool(raw, "hard_start", position=position)
    hard_end = _required_bool(raw, "hard_end", position=position)

    high_start = _finite(raw, "high_scale_start", position=position)
    high_end = _finite(raw, "high_scale_end", position=position)
    low_start = _finite(raw, "low_scale_start", position=position)
    low_end = _finite(raw, "low_scale_end", position=position)
    beta = _finite(raw, "beta", position=position)
    if beta <= 0.0:
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].beta must be > 0"
        )
    scale_temporal_axis = _required_bool(
        raw,
        "scale_temporal_axis",
        position=position,
    )
    capability_name = "terminal_pece_exact_corrector_safe"
    if schema == 1:
        if capability_name in raw:
            raise compat.ExternalPatchContractError(
                f"visual_profile[{position}].{capability_name} requires schema_version=2"
            )
        terminal_pece_safe = False
    else:
        terminal_pece_safe = _required_bool(
            raw,
            capability_name,
            position=position,
        )

    expected_strength = max(
        abs(high_start - 1.0),
        abs(high_end - 1.0),
        abs(low_start - 1.0),
        abs(low_end - 1.0),
    )
    if not math.isclose(strength, expected_strength, rel_tol=1e-9, abs_tol=1e-9):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].strength does not match the declared scale endpoints"
        )
    if terminal_pece_safe and not (
        progress_start == 0.0
        and 0.90 <= progress_end < 1.0
        and not hard_start
        and hard_end
        and strength <= 0.05 + 1e-9
        and scope in {"image_only", "image_and_video"}
        and not scale_temporal_axis
    ):
        raise compat.ExternalPatchContractError(
            f"visual_profile[{position}].{capability_name} is outside the reviewed "
            "weak spatial-only late hard-end envelope"
        )

    # Both the producer's schedule_fraction() and Spectrum's active_at() are
    # inclusive at their start/end coordinates. Preserve those endpoint semantics
    # exactly under sigma = 1 - progress. Only producer-declared hard boundaries
    # are retained; soft boundaries are widened to the edge so they cannot
    # spuriously force an actual step.
    guard_progress_start = progress_start if hard_start else 0.0
    guard_progress_end = progress_end if hard_end else 1.0
    sigma_start = 1.0 - guard_progress_end
    sigma_end = 1.0 - guard_progress_start

    descriptor = compat.ExternalPatchDescriptor(
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
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=1.0,
        cond_only=False,
        scope=scope,
        terminal_pece_exact_corrector_safe=terminal_pece_safe,
    )
    canonical = descriptor.canonical + (
        f"visual_reference_profile_v{schema}",
        progress_start,
        progress_end,
        hard_start,
        hard_end,
        high_start,
        high_end,
        low_start,
        low_end,
        beta,
        scale_temporal_axis,
        terminal_pece_safe,
    )
    return descriptor, canonical


def parse_external_patch_contracts_with_visual_references(
    model_options: dict[str, Any] | None,
    *,
    block_count: int,
) -> compat.ParsedExternalPatchContracts:
    assert _ORIGINAL_PARSE is not None
    base = _ORIGINAL_PARSE(model_options, block_count=block_count)
    options = model_options or {}
    raw = options.get(VISUAL_PATCH_PROFILES_KEY)
    if raw is None:
        return base
    if not isinstance(raw, (tuple, list)):
        raise compat.ExternalPatchContractError(
            f"{VISUAL_PATCH_PROFILES_KEY} must be a sequence of descriptors"
        )

    parsed_visual = tuple(
        _parse_visual_profile(value, block_count=int(block_count), position=index)
        for index, value in enumerate(raw)
    )
    if not parsed_visual:
        return base

    descriptors = tuple(base.descriptors) + tuple(value[0] for value in parsed_visual)
    instance_ids = [value.instance_id for value in descriptors]
    if len(set(instance_ids)) != len(instance_ids):
        raise compat.ExternalPatchContractError(
            "external patch instance_id values must be unique across all profile kinds"
        )

    canonical = tuple(base.canonical) + tuple(value[1] for value in parsed_visual)
    fingerprint = hashlib.sha256(
        repr(canonical).encode("utf-8", "backslashreplace")
    ).hexdigest()
    return compat.ParsedExternalPatchContracts(descriptors, canonical, fingerprint)


def _visual_runtime_entries(
    transformer_options: dict[str, Any],
    descriptors: tuple[compat.ExternalPatchDescriptor, ...],
) -> list[dict[str, Any]]:
    if not descriptors:
        raw = transformer_options.get(VISUAL_PATCH_RUNTIME_KEY)
        if raw not in (None, (), []):
            raise compat.ExternalPatchContractError(
                "visual-reference runtime state exists without a declared static profile"
            )
        return []

    raw = transformer_options.get(VISUAL_PATCH_RUNTIME_KEY)
    if not isinstance(raw, (tuple, list)):
        raise compat.ExternalPatchContractError(
            f"{VISUAL_PATCH_RUNTIME_KEY} is missing or is not a sequence"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(raw):
        if not isinstance(value, dict):
            raise compat.ExternalPatchContractError(
                f"visual_runtime[{position}] must be a dictionary"
            )
        schema = value.get("schema_version")
        if (
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema not in VISUAL_PATCH_SUPPORTED_SCHEMA_VERSIONS
        ):
            raise compat.ExternalPatchContractError(
                f"visual_runtime[{position}] has unsupported schema_version"
            )
        provider = compat._required_string(value, "provider")
        instance_id = compat._required_string(value, "instance_id")
        if provider != VISUAL_PATCH_PROVIDER:
            raise compat.ExternalPatchContractError(
                f"visual_runtime[{position}] uses unsupported provider={provider!r}"
            )
        if instance_id in by_id:
            raise compat.ExternalPatchContractError(
                f"duplicate visual runtime state for external patch instance {instance_id!r}"
            )
        progress = compat._finite_number(
            value.get("schedule_progress"),
            name=f"visual_runtime[{instance_id}].schedule_progress",
        )
        if not 0.0 <= progress <= 1.0:
            raise compat.ExternalPatchContractError(
                f"visual runtime progress for {instance_id!r} is outside [0, 1]"
            )
        active = value.get("active")
        if not isinstance(active, bool):
            raise compat.ExternalPatchContractError(
                f"visual_runtime[{instance_id}].active must be a boolean"
            )
        # `active` is validated as part of the producer contract but is not the
        # temporal guard input. The producer folds reference selection/mapping into
        # cfg.enabled, while Spectrum's transaction state must follow the declared
        # hard schedule boundaries. Derive that state only from exact progress.
        by_id[instance_id] = {
            "schema_version": compat.EXTERNAL_PATCH_SCHEMA_VERSION,
            "provider": provider,
            "instance_id": instance_id,
            "normalized_sigma": 1.0 - progress,
            "provider_schema_version": schema,
        }

    expected = {value.instance_id for value in descriptors}
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise compat.ExternalPatchContractError(
            f"visual patch runtime/static instance mismatch missing={missing} extra={extra}"
        )
    ordered: list[dict[str, Any]] = []
    for descriptor in descriptors:
        entry = by_id[descriptor.instance_id]
        if entry.pop("provider_schema_version") != descriptor.schema_version:
            raise compat.ExternalPatchContractError(
                f"visual runtime/profile schema changed for instance {descriptor.instance_id!r}"
            )
        ordered.append(entry)
    return ordered


def runtime_entries_with_visual_references(
    transformer_options: dict[str, Any],
    parsed: compat.ParsedExternalPatchContracts,
) -> tuple[float, ...]:
    assert _ORIGINAL_RUNTIME_ENTRIES is not None
    visual_descriptors = tuple(
        value for value in parsed.descriptors if value.kind == VISUAL_PATCH_KIND
    )
    if not visual_descriptors:
        return _ORIGINAL_RUNTIME_ENTRIES(transformer_options, parsed)

    raw_legacy = transformer_options.get(compat.EXTERNAL_PATCH_RUNTIME_KEY)
    if raw_legacy is None:
        legacy_entries: list[Any] = []
    elif isinstance(raw_legacy, (tuple, list)):
        legacy_entries = list(raw_legacy)
    else:
        raise compat.ExternalPatchContractError(
            f"{compat.EXTERNAL_PATCH_RUNTIME_KEY} is not a sequence"
        )

    synthetic_visual = _visual_runtime_entries(
        transformer_options,
        visual_descriptors,
    )
    combined = dict(transformer_options)
    combined[compat.EXTERNAL_PATCH_RUNTIME_KEY] = tuple(legacy_entries + synthetic_visual)
    return _ORIGINAL_RUNTIME_ENTRIES(combined, parsed)


def install_visual_reference_patch_compat() -> None:
    global _INSTALLED, _ORIGINAL_PARSE, _ORIGINAL_RUNTIME_ENTRIES
    if _INSTALLED:
        return
    if not compat._INSTALLED:
        raise RuntimeError(
            "visual-reference external patch support requires install_external_patch_compat() first"
        )

    _ORIGINAL_PARSE = compat.parse_external_patch_contracts
    _ORIGINAL_RUNTIME_ENTRIES = compat._runtime_entries
    compat.parse_external_patch_contracts = parse_external_patch_contracts_with_visual_references
    compat._runtime_entries = runtime_entries_with_visual_references
    _INSTALLED = True


__all__ = [
    "VISUAL_PATCH_ARCHITECTURE",
    "VISUAL_PATCH_KIND",
    "VISUAL_PATCH_PROFILES_KEY",
    "VISUAL_PATCH_PROVIDER",
    "VISUAL_PATCH_RUNTIME_KEY",
    "VISUAL_PATCH_SCHEMA_VERSION",
    "VISUAL_PATCH_SUPPORTED_SCHEMA_VERSIONS",
    "install_visual_reference_patch_compat",
    "parse_external_patch_contracts_with_visual_references",
    "runtime_entries_with_visual_references",
]
