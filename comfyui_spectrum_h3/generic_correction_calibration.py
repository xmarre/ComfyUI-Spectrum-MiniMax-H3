from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .generic_correction_core import ScalarMoments

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SOURCE_SCHEMA_REVISION = "generic-correction-v1"
LOG_PREFIX = "SPECTRUM_GENERIC_CORRECTION_CALIBRATION_JSON="
PACKAGE_NAME = "comfyui-spectrum-minimax-h3"
FALLBACK_PACKAGE_VERSION = "0.2.8"
MAX_SERIALIZED_BYTES = 512 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def package_version() -> str:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return FALLBACK_PACKAGE_VERSION


def source_revision() -> tuple[str | None, str]:
    value = os.environ.get("SPECTRUM_H3_SOURCE_REVISION")
    if value:
        return value, "SPECTRUM_H3_SOURCE_REVISION"
    return None, "unavailable_without_external_annotation"


@dataclass(slots=True)
class GenericCalibrationState:
    enabled: bool
    run_id: int
    sampler_name: str
    total_steps: int
    schedule: tuple[float, ...]
    config_snapshot: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    topology: tuple[Any, ...] | None = None
    failures: int = 0
    emitted: bool = False


def create_state(
    runtime: Any,
    sigmas: torch.Tensor,
    *,
    enabled: bool,
) -> GenericCalibrationState:
    schedule_tensor = (
        torch.as_tensor(sigmas)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1)
    )
    schedule = tuple(float(value) for value in schedule_tensor.tolist())
    if not all(math.isfinite(value) for value in schedule):
        schedule = ()
    run = runtime._run
    if run is None:
        raise RuntimeError("generic calibration requires an active run")
    return GenericCalibrationState(
        enabled=bool(enabled),
        run_id=int(run.run_id),
        sampler_name=str(run.sampler_name),
        total_steps=int(run.total_steps),
        schedule=schedule,
        config_snapshot=asdict(runtime.config),
    )


def exact_segment_moments(
    forecaster: Any,
    actual_feature: torch.Tensor,
    weighted_segments: Sequence[tuple[str, int, int, torch.Tensor]],
) -> dict[str, ScalarMoments]:
    """Reduce exact full-stream quadratic moments with bounded chunk workspace."""
    if forecaster.feature_shape is None or tuple(actual_feature.shape) != tuple(
        forecaster.feature_shape
    ):
        raise ValueError("exact moments require the current forecaster feature shape")
    if actual_feature.ndim < 2 or forecaster.history_length < 2:
        raise ValueError("exact moments require two actual anchors")
    feature_shape = tuple(int(value) for value in actual_feature.shape)
    branch_count = feature_shape[0]
    feature_rows = feature_shape[1] if len(feature_shape) >= 3 else 1
    row_numel = (
        math.prod(feature_shape[2:])
        if len(feature_shape) >= 3
        else math.prod(feature_shape[1:])
    )
    tail_numel = math.prod(feature_shape[1:])
    actual_flat = actual_feature.detach().reshape(-1)
    target_device = actual_feature.device
    history = forecaster._history
    normalized: list[tuple[str, int, int, tuple[float, ...]]] = []
    for key, start, end, weights in weighted_segments:
        begin = int(start)
        stop = int(end)
        if begin < 0 or stop <= begin or stop > feature_rows:
            raise ValueError("exact moment segment is outside the feature rows")
        if weights.ndim != 1 or int(weights.numel()) != len(history):
            raise ValueError("exact moment weights do not match retained history")
        scalars = tuple(float(value) for value in weights.tolist())
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("exact moment prediction weights are nonfinite")
        normalized.append((str(key), begin, stop, scalars))
    if len({key for key, *_ in normalized}) != len(normalized):
        raise ValueError("exact moment segment keys must be unique")

    accumulators = {
        key: torch.zeros(5, device=target_device, dtype=torch.float32)
        for key, *_ in normalized
    }
    counts = {key: 0 for key, *_ in normalized}
    for source_branch in range(branch_count):
        source_base = source_branch * tail_numel
        for key, start, end, weight_scalars in normalized:
            segment_start = start * row_numel
            segment_numel = (end - start) * row_numel
            chunk_elements = min(
                int(forecaster._chunk_elements(target_device)),
                segment_numel,
            )
            for offset in range(0, segment_numel, chunk_elements):
                length = min(chunk_elements, segment_numel - offset)
                absolute = source_base + segment_start + offset
                predicted = torch.zeros(
                    length,
                    device=target_device,
                    dtype=torch.float32,
                )
                for scalar, entry in zip(weight_scalars, history, strict=True):
                    if scalar == 0.0:
                        continue
                    source = entry.feature_flat.narrow(0, absolute, length)
                    predicted.add_(
                        source.to(
                            device=target_device,
                            dtype=torch.float32,
                            non_blocking=False,
                        ),
                        alpha=scalar,
                    )
                actual = actual_flat.narrow(0, absolute, length).to(dtype=torch.float32)
                latest = (
                    history[-1]
                    .feature_flat.narrow(0, absolute, length)
                    .to(
                        device=target_device,
                        dtype=torch.float32,
                        non_blocking=False,
                    )
                )
                previous = (
                    history[-2]
                    .feature_flat.narrow(0, absolute, length)
                    .to(
                        device=target_device,
                        dtype=torch.float32,
                        non_blocking=False,
                    )
                )
                residual = actual - predicted
                direction = latest - previous
                hold_error = actual - latest
                terms = torch.stack(
                    (
                        residual.square().sum(dtype=torch.float32),
                        (residual * direction).sum(dtype=torch.float32),
                        direction.square().sum(dtype=torch.float32),
                        hold_error.square().sum(dtype=torch.float32),
                        actual.square().sum(dtype=torch.float32),
                    )
                )
                accumulators[key].add_(terms)
                counts[key] += length

    ordered_keys = [key for key, *_ in normalized]
    transferred = torch.cat([accumulators[key] for key in ordered_keys]).to(
        device="cpu",
        dtype=torch.float64,
    )
    resolved = transferred.tolist()
    result: dict[str, ScalarMoments] = {}
    offset = 0
    numeric_epsilon = float(torch.finfo(torch.float32).eps)
    for key in ordered_keys:
        count = counts[key]
        if count <= 0:
            raise ValueError("exact moment segment contains no elements")
        residual_sq, cross, direction_sq, hold_sq, actual_sq = (
            float(value) / count for value in resolved[offset : offset + 5]
        )
        offset += 5
        ratio_epsilon = max(math.sqrt(max(0.0, actual_sq)) * 1.0e-6, numeric_epsilon)
        denominator = max(math.sqrt(max(0.0, hold_sq)), ratio_epsilon)
        result[key] = ScalarMoments(
            sample_count=count,
            residual_sq_mean=max(0.0, residual_sq),
            residual_dot_direction_mean=cross,
            direction_sq_mean=max(0.0, direction_sq),
            hold_error_sq_mean=max(0.0, hold_sq),
            actual_sq_mean=max(0.0, actual_sq),
            ratio_epsilon=ratio_epsilon,
            ratio_denominator_rms=denominator,
        ).validate()
    return result


def record_row(
    state: GenericCalibrationState | None,
    row: dict[str, Any],
    *,
    topology: tuple[Any, ...],
) -> None:
    if state is None or not state.enabled:
        return
    if state.topology is None:
        state.topology = topology
    elif state.topology != topology:
        state.failures += 1
        return
    canonical_json(row)
    state.rows.append(row)


def build_block(runtime: Any, state: GenericCalibrationState) -> dict[str, Any]:
    revision, revision_source = source_revision()
    config_hash = sha256_json(state.config_snapshot)
    schedule_fingerprint = sha256_json(state.schedule)
    topology_fingerprint = sha256_json(repr(state.topology))
    seed = getattr(runtime, "_spectrum_h3_observed_seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = None
    rows_without_fingerprint = [
        {key: value for key, value in row.items() if key != "trace_fingerprint"}
        for row in state.rows
    ]
    rows_without_fingerprint.sort(
        key=lambda row: (
            int(row["target_step_id"]),
            str(row["stream"]),
            str(row.get("region_id") or ""),
        ),
    )
    trace_fingerprint = sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "source_schema_revision": SOURCE_SCHEMA_REVISION,
            "package_version": package_version(),
            "source_revision": revision,
            "seed": seed,
            "config_hash": config_hash,
            "schedule_fingerprint": schedule_fingerprint,
            "topology_fingerprint": topology_fingerprint,
            "rows": rows_without_fingerprint,
        }
    )
    rows = [
        {**row, "trace_fingerprint": trace_fingerprint}
        for row in rows_without_fingerprint
    ]
    block = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spectrum_h3_generic_correction_calibration",
        "provenance": {
            "run_id": state.run_id,
            "seed": seed,
            "package_version": package_version(),
            "source_schema_revision": SOURCE_SCHEMA_REVISION,
            "source_revision": revision,
            "source_revision_source": revision_source,
            "config_hash": config_hash,
            "schedule_fingerprint": schedule_fingerprint,
            "topology_fingerprint": topology_fingerprint,
            "trace_fingerprint": trace_fingerprint,
        },
        "metadata": {
            "sampler": state.sampler_name,
            "steps": state.total_steps,
            "row_count": len(rows),
            "failure_count": state.failures,
            "ratio_aggregation": (
                "per-target sqrt(quadratic_mse)/max(hold_rms,actual_rms*1e-6,float32_eps), then arithmetic mean"
            ),
            "sign_convention": (
                "residual=actual-predicted; direction=latest_exact-previous_exact; MSE(g)=A-2*g*B+g^2*C"
            ),
        },
        "config": state.config_snapshot,
        "target_rows": rows,
        "compatible": state.failures == 0 and bool(rows),
    }
    encoded = canonical_json(block).encode("utf-8")
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise ValueError(
            f"generic correction calibration block exceeds {MAX_SERIALIZED_BYTES} bytes"
        )
    return block


def emit_block(
    runtime: Any, state: GenericCalibrationState | None
) -> dict[str, Any] | None:
    if state is None or not state.enabled or state.emitted or not state.rows:
        return None
    block = build_block(runtime, state)
    state.emitted = True
    LOG.warning("%s%s", LOG_PREFIX, canonical_json(block))
    return block


__all__ = [
    "LOG_PREFIX",
    "SCHEMA_VERSION",
    "GenericCalibrationState",
    "build_block",
    "canonical_json",
    "create_state",
    "emit_block",
    "exact_segment_moments",
    "record_row",
    "sha256_json",
]
