from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch

from .forecast import HistoryWeightForecaster
from .model_aware import ModelAwareForecastDecision

DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
OFFLINE_VALIDATION_SAMPLES = 16 * 1024


def _chunk_elements(chunk_bytes: int) -> int:
    if chunk_bytes < 4096:
        raise ValueError("chunk_bytes must be >= 4096")
    return max(1024, int(chunk_bytes) // torch.tensor([], dtype=torch.float32).element_size())


def tensor_all_finite(value: torch.Tensor, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> bool:
    flat = value.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    for offset in range(0, flat.numel(), chunk):
        if not bool(torch.isfinite(flat.narrow(0, offset, min(chunk, flat.numel() - offset))).all().item()):
            return False
    return True


@dataclass(frozen=True, slots=True)
class StreamResidualScore:
    forecast_rms: float
    hold_rms: float
    actual_rms: float
    epsilon: float
    score: float
    chunks: int


def measure_stream_residual(
    actual: torch.Tensor,
    shadow: torch.Tensor,
    hold: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> StreamResidualScore:
    if actual.shape != shadow.shape or actual.shape != hold.shape:
        raise ValueError("actual, shadow, and hold outputs must have identical shapes")
    if not actual.dtype.is_floating_point or not shadow.dtype.is_floating_point or not hold.dtype.is_floating_point:
        raise ValueError("residual measurement requires floating-point outputs")
    count = actual.numel()
    if count == 0:
        raise ValueError("residual measurement cannot reduce an empty output")

    actual_flat = actual.detach().reshape(-1)
    shadow_flat = shadow.detach().reshape(-1)
    hold_flat = hold.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    actual_sq = 0.0
    forecast_sq = 0.0
    hold_sq = 0.0
    chunks = 0
    for offset in range(0, count, chunk):
        length = min(chunk, count - offset)
        actual_chunk = actual_flat.narrow(0, offset, length).to(torch.float32)
        shadow_chunk = shadow_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        hold_chunk = hold_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        actual_sq += float(torch.sum(actual_chunk * actual_chunk, dtype=torch.float32).item())
        forecast_delta = actual_chunk - shadow_chunk
        hold_delta = actual_chunk - hold_chunk
        forecast_sq += float(torch.sum(forecast_delta * forecast_delta, dtype=torch.float32).item())
        hold_sq += float(torch.sum(hold_delta * hold_delta, dtype=torch.float32).item())
        chunks += 1

    actual_rms = math.sqrt(actual_sq / count)
    forecast_rms = math.sqrt(forecast_sq / count)
    hold_rms = math.sqrt(hold_sq / count)
    if not all(math.isfinite(value) for value in (actual_rms, forecast_rms, hold_rms)):
        raise ValueError("residual measurement produced a nonfinite RMS")

    epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
    if forecast_rms <= epsilon and hold_rms <= epsilon:
        score = 0.0
    else:
        score = forecast_rms / max(hold_rms, epsilon)
    if not math.isfinite(score):
        raise ValueError("residual measurement produced a nonfinite score")
    return StreamResidualScore(
        forecast_rms=forecast_rms,
        hold_rms=hold_rms,
        actual_rms=actual_rms,
        epsilon=epsilon,
        score=score,
        chunks=chunks,
    )


@dataclass(frozen=True, slots=True)
class OfflineModelAwareDecision:
    degree: int
    ridge_lambda: float
    audio_blend_weight: float
    video_blend_weight: float
    audio_correction_gain: float
    video_correction_gain: float
    audio_correction_coefficients: tuple[float, ...] = ()
    video_correction_coefficients: tuple[float, ...] = ()
    correction_anchor_ids: tuple[int, ...] = ()

    @classmethod
    def from_runtime(
        cls,
        decision: ModelAwareForecastDecision,
    ) -> OfflineModelAwareDecision:
        return cls(
            degree=int(decision.degree),
            ridge_lambda=float(decision.ridge_lambda),
            audio_blend_weight=float(decision.audio_blend_weight),
            video_blend_weight=float(decision.video_blend_weight),
            audio_correction_gain=float(decision.audio_correction_gain),
            video_correction_gain=float(decision.video_correction_gain),
            audio_correction_coefficients=(
                tuple(decision.audio_subspace_telemetry.applied_coefficients)
                if decision.audio_subspace_telemetry.eligible
                else (
                    (float(decision.audio_correction_gain),)
                    if decision.audio_correction_gain != 0.0
                    else ()
                )
            ),
            video_correction_coefficients=(
                tuple(decision.video_subspace_telemetry.applied_coefficients)
                if decision.video_subspace_telemetry.eligible
                else (
                    (float(decision.video_correction_gain),)
                    if decision.video_correction_gain != 0.0
                    else ()
                )
            ),
            correction_anchor_ids=tuple(decision.correction_anchor_ids),
        )


@dataclass(frozen=True, slots=True)
class OfflineStepRecord:
    step_id: int
    coordinate: float
    actual: bool
    model_aware_decision: OfflineModelAwareDecision | None = None


@dataclass(slots=True)
class OfflineAnchor:
    step_id: int
    coordinate: float
    feature: torch.Tensor


class OfflineFeatureArchive:
    def __init__(
        self,
        *,
        total_steps: int,
        sampler_name: str,
        history_storage: str = "system_ram",
    ) -> None:
        if history_storage not in {"system_ram", "vram"}:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        self.total_steps = int(total_steps)
        self.sampler_name = str(sampler_name)
        self.history_storage = str(history_storage)
        self.steps: list[OfflineStepRecord] = []
        self.anchors: list[OfflineAnchor] = []
        self.labels: tuple[Any, ...] | None = None
        self.topology: tuple[Any, ...] | None = None
        self.feature_shape: tuple[int, ...] | None = None
        self.feature_dtype: torch.dtype | None = None
        self.history_device: torch.device | None = None
        self.valid = True
        self.failure_reason: str | None = None

    @property
    def tensor_bytes(self) -> int:
        return sum(anchor.feature.numel() * anchor.feature.element_size() for anchor in self.anchors)

    @property
    def estimated_tensor_bytes(self) -> int:
        if not self.anchors:
            return 0
        feature = self.anchors[0].feature
        return self.total_steps * feature.numel() * feature.element_size()

    def invalidate(self, reason: str) -> None:
        if self.valid:
            self.valid = False
            self.failure_reason = str(reason)

    def record_step(
        self,
        step_id: int,
        coordinate: float,
        actual: bool,
        *,
        model_aware_decision: ModelAwareForecastDecision | None = None,
    ) -> None:
        if not self.valid:
            return
        expected = len(self.steps)
        if int(step_id) != expected:
            self.invalidate(f"offline step sequence changed: expected {expected}, got {step_id}")
            return
        self.steps.append(
            OfflineStepRecord(
                int(step_id),
                float(coordinate),
                bool(actual),
                (
                    None
                    if model_aware_decision is None
                    else OfflineModelAwareDecision.from_runtime(model_aware_decision)
                ),
            )
        )

    def record_actual(
        self,
        step_id: int,
        coordinate: float,
        feature: torch.Tensor,
        *,
        labels: tuple[Any, ...],
        topology: tuple[Any, ...],
        take_ownership: bool,
    ) -> None:
        if not self.valid:
            return
        shape = tuple(int(value) for value in feature.shape)
        if self.labels is None:
            self.labels = tuple(labels)
            self.topology = tuple(topology)
            self.feature_shape = shape
            self.feature_dtype = feature.dtype
        elif tuple(labels) != self.labels or tuple(topology) != self.topology:
            self.invalidate("offline branch labels or topology changed across actual anchors")
            return
        elif shape != self.feature_shape or feature.dtype != self.feature_dtype:
            self.invalidate("offline actual feature shape or dtype changed")
            return

        detached = feature.detach()
        storage_device = torch.device("cpu") if self.history_storage == "system_ram" else detached.device
        if self.history_device is None:
            self.history_device = storage_device
        elif storage_device != self.history_device:
            self.invalidate("offline actual feature device changed")
            return
        if take_ownership and detached.device == storage_device and detached.is_contiguous():
            archived = detached
        else:
            archived = detached.to(device=storage_device, dtype=feature.dtype, copy=True).contiguous()
        self.anchors.append(OfflineAnchor(int(step_id), float(coordinate), archived))

    def complete(self, *, minimum_anchors: int) -> bool:
        if len(self.steps) != self.total_steps:
            self.invalidate(
                f"offline first pass recorded {len(self.steps)} of {self.total_steps} logical steps"
            )
        actual_ids = [step.step_id for step in self.steps if step.actual]
        anchor_ids = [anchor.step_id for anchor in self.anchors]
        if actual_ids != anchor_ids:
            self.invalidate("offline actual-step schedule does not match the retained anchor archive")
        if len(self.anchors) < int(minimum_anchors):
            self.invalidate(
                f"offline smoothing requires at least {minimum_anchors} actual anchors"
            )
        if any(
            not step.actual
            and not any(anchor.step_id < step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no earlier actual anchor")
        if any(
            not step.actual
            and not any(anchor.step_id > step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no future actual anchor")
        return self.valid

    def release(self) -> None:
        self.steps.clear()
        self.anchors.clear()
        self.labels = None
        self.topology = None
        self.feature_shape = None
        self.feature_dtype = None
        self.history_device = None


class OfflineSmoother:
    def __init__(
        self,
        archive: OfflineFeatureArchive,
        *,
        degree: int,
        ridge_lambda: float,
        blend_weight: float,
        audio_blend_weight: float = 0.0,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        transition_logger: Callable[..., None] | None = None,
    ) -> None:
        if not archive.valid or not archive.anchors or archive.labels is None:
            raise ValueError("offline archive is incomplete")
        self.archive = archive
        self.blend_weight = float(blend_weight)
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        self.audio_blend_weight = float(audio_blend_weight)
        if not 0.0 <= self.audio_blend_weight <= 1.0:
            raise ValueError("audio_blend_weight must be in [0, 1]")
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self._anchor_ids = [anchor.step_id for anchor in archive.anchors]
        self._anchor_by_step = {anchor.step_id: anchor for anchor in archive.anchors}
        self._branch_count = int(archive.anchors[0].feature.shape[0])
        self._forecaster = HistoryWeightForecaster(
            degree=degree,
            ridge_lambda=ridge_lambda,
            max_history=max(len(archive.anchors), degree + 1, 2),
            chunk_bytes=chunk_bytes,
            history_storage=archive.history_storage,
        )
        attach_started = time.perf_counter()
        for anchor in archive.anchors:
            self._forecaster.update(
                anchor.coordinate,
                anchor.feature,
                anchor_id=anchor.step_id,
                take_ownership=True,
            )
        if transition_logger is not None:
            transition_logger(
                "offline_smoother_archive_attached",
                elapsed_s=time.perf_counter() - attach_started,
                anchors=len(archive.anchors),
                shared_bytes=archive.tensor_bytes,
            )
        self._stream_ranges = self._resolve_stream_ranges()
        if (
            self._stream_ranges[0][0] == "packed"
            and not math.isclose(
                self.audio_blend_weight,
                self.blend_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "offline modality-specific blending requires target audio/video row metadata"
            )
        self.configured_stream_blends = {
            name: self.audio_blend_weight if name == "audio" else self.blend_weight
            for name, _, _ in self._stream_ranges
        }
        self.validation_stream_count = len(self._stream_ranges)
        self.validation_stream_max_scores = {
            name: 0.0 for name, _, _ in self._stream_ranges
        }
        self.validation_samples_per_branch = 0
        self.validation_anchor_count = 0
        self.attenuated_prediction_count = 0
        self.local_only_prediction_count = 0
        self.attenuated_prediction_counts = {
            name: 0 for name, _, _ in self._stream_ranges
        }
        self.local_only_prediction_counts = {
            name: 0 for name, _, _ in self._stream_ranges
        }
        self.effective_blend_stream_stats = {
            name: (blend, blend, blend)
            for name, blend in self.configured_stream_blends.items()
        }
        self.effective_blend_min = self.blend_weight
        self.effective_blend_mean = self.blend_weight
        self.effective_blend_max = self.blend_weight
        self._last_prediction_chunk_count = 0
        self.model_aware_offline_correction_seconds = 0.0
        self.model_aware_offline_correction_applications = 0
        validation_started = time.perf_counter()
        if transition_logger is not None:
            transition_logger(
                "offline_smoother_validation_begin",
                anchors=len(archive.anchors),
                streams=len(self._stream_ranges),
            )
        try:
            self._validation_scores = self._build_validation_scores()
        finally:
            self.validation_seconds = time.perf_counter() - validation_started
            if transition_logger is not None:
                transition_logger(
                    "offline_smoother_validation_end",
                    elapsed_s=self.validation_seconds,
                    anchors=len(archive.anchors),
                    samples_per_branch=self.validation_samples_per_branch,
                )
        weights_started = time.perf_counter()
        self._forecast_weights = self._build_forecast_weights()
        if transition_logger is not None:
            transition_logger(
                "offline_smoother_weights_end",
                elapsed_s=time.perf_counter() - weights_started,
                forecast_weight_sets=len(self._forecast_weights),
            )

    @property
    def history_length(self) -> int:
        return self._forecaster.history_length

    @property
    def history_device(self) -> torch.device | None:
        return self._forecaster.history_device

    @property
    def history_tensor_bytes(self) -> int:
        return self._forecaster.history_tensor_bytes

    @property
    def last_prediction_chunk_count(self) -> int:
        return self._last_prediction_chunk_count

    @property
    def model_aware_fit_seconds(self) -> float:
        return self._forecaster.model_aware_fit_seconds

    @property
    def model_aware_correction_seconds(self) -> float:
        return (
            self._forecaster.model_aware_correction_seconds
            + self.model_aware_offline_correction_seconds
        )

    @staticmethod
    def _affine_spectral_weights(weights: torch.Tensor) -> torch.Tensor:
        normalized = weights.detach().to(device="cpu", dtype=torch.float32).clone()
        if normalized.ndim != 1 or normalized.numel() == 0 or not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("offline spectral weights are invalid")
        normalized.add_((1.0 - float(normalized.sum().item())) / normalized.numel())
        if not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("offline affine spectral correction is nonfinite")
        return normalized

    def _resolve_stream_ranges(self) -> list[tuple[str, int, int]]:
        feature_shape = self.archive.feature_shape
        if feature_shape is None or len(feature_shape) != 3:
            raise RuntimeError("offline smoothing requires [branch, rows, width] features")
        topology = {
            str(entry[0]): entry[1]
            for entry in (self.archive.topology or ())
            if isinstance(entry, tuple) and len(entry) == 2
        }
        audio_rows = topology.get("target_audio_rows")
        video_rows = topology.get("target_video_rows")
        if (
            isinstance(audio_rows, int)
            and isinstance(video_rows, int)
            and audio_rows > 0
            and video_rows > 0
            and audio_rows + video_rows == feature_shape[1]
        ):
            return [
                ("audio", 0, audio_rows),
                ("video", audio_rows, audio_rows + video_rows),
            ]
        return [("packed", 0, feature_shape[1])]

    def _sampled_anchors(self, start_row: int, end_row: int) -> torch.Tensor:
        width = self.archive.anchors[0].feature.shape[2]
        tail_numel = (end_row - start_row) * width
        sample_count = min(OFFLINE_VALIDATION_SAMPLES, tail_numel)
        flat_indices = torch.div(
            torch.arange(sample_count, dtype=torch.int64) * tail_numel,
            sample_count,
            rounding_mode="floor",
        )
        feature_rows = torch.div(flat_indices, width, rounding_mode="floor") + start_row
        feature_columns = flat_indices.remainder(width)
        sampled = []
        for anchor in self.archive.anchors:
            rows = feature_rows.to(device=anchor.feature.device)
            columns = feature_columns.to(device=anchor.feature.device)
            sampled.append(
                anchor.feature[:, rows, columns].to(device="cpu", dtype=torch.float32)
            )
        self.validation_samples_per_branch += sample_count
        return torch.stack(sampled, dim=0)

    def _build_validation_scores(self) -> list[list[list[float | None]]]:
        anchor_count = len(self.archive.anchors)
        scores: list[list[list[float | None]]] = [
            [[None] * self._branch_count for _ in range(anchor_count)]
            for _ in self._stream_ranges
        ]
        if anchor_count < self.degree + 2 or anchor_count < 3:
            return scores

        for stream_index, (stream_name, start_row, end_row) in enumerate(self._stream_ranges):
            samples = self._sampled_anchors(start_row, end_row)
            for target_index in range(1, anchor_count - 1):
                retained = [index for index in range(anchor_count) if index != target_index]
                validator = HistoryWeightForecaster(
                    degree=self.degree,
                    ridge_lambda=self.ridge_lambda,
                    max_history=max(len(retained), self.degree + 1, 2),
                    chunk_bytes=DEFAULT_CHUNK_BYTES,
                    history_storage="system_ram",
                )
                for index in retained:
                    validator.update(
                        self.archive.anchors[index].coordinate,
                        samples[index],
                        take_ownership=True,
                    )
                spectral = self._affine_spectral_weights(
                    validator.spectral_weights(self.archive.anchors[target_index].coordinate)
                )
                spectral_prediction = torch.einsum("k,kbs->bs", spectral, samples[retained])

                left = self.archive.anchors[target_index - 1]
                target = self.archive.anchors[target_index]
                right = self.archive.anchors[target_index + 1]
                spacing = right.coordinate - left.coordinate
                if abs(spacing) <= 1e-12:
                    raise RuntimeError("offline validation anchors have duplicate coordinates")
                ratio = (target.coordinate - left.coordinate) / spacing
                local_prediction = torch.lerp(samples[target_index - 1], samples[target_index + 1], ratio)
                target_samples = samples[target_index]

                for branch in range(self._branch_count):
                    actual = target_samples[branch]
                    spectral_rms = float(torch.sqrt(torch.mean((spectral_prediction[branch] - actual) ** 2)).item())
                    local_rms = float(torch.sqrt(torch.mean((local_prediction[branch] - actual) ** 2)).item())
                    actual_rms = float(torch.sqrt(torch.mean(actual * actual)).item())
                    epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
                    if spectral_rms <= epsilon and local_rms <= epsilon:
                        score = 0.0
                    else:
                        score = spectral_rms / max(local_rms, epsilon)
                    if not math.isfinite(score):
                        raise RuntimeError("offline validation score is nonfinite")
                    scores[stream_index][target_index][branch] = score
                    self.validation_stream_max_scores[stream_name] = max(
                        self.validation_stream_max_scores[stream_name], score
                    )
        self.validation_anchor_count = anchor_count - 2
        return scores

    def _validation_score_for_interval(self, position: int, branch: int, stream_index: int) -> float:
        nearby = [
            self._validation_scores[stream_index][index][branch]
            for index in (position - 1, position)
            if self._validation_scores[stream_index][index][branch] is not None
        ]
        return max(nearby, default=1.0)

    def _build_forecast_weights(self) -> dict[tuple[int, int, int], torch.Tensor]:
        weights_by_step: dict[tuple[int, int, int], torch.Tensor] = {}
        effective_blends: list[float] = []
        stream_effective_blends = {
            name: [] for name, _, _ in self._stream_ranges
        }
        for record in self.archive.steps:
            if record.actual:
                continue
            decision = record.model_aware_decision
            spectral_degree = self.degree if decision is None else decision.degree
            spectral_ridge = self.ridge_lambda if decision is None else decision.ridge_lambda
            spectral = self._affine_spectral_weights(
                self._forecaster.model_aware_weights(
                    record.coordinate,
                    1.0,
                    degree=spectral_degree,
                    ridge_lambda=spectral_ridge,
                    correction_gain=0.0,
                )
            )
            position = bisect.bisect_left(self._anchor_ids, record.step_id)
            if position == 0 or position == len(self._anchor_ids):
                raise RuntimeError("offline forecast requires bracketing actual anchors")
            left = self.archive.anchors[position - 1]
            right = self.archive.anchors[position]
            spacing = right.coordinate - left.coordinate
            if abs(spacing) <= 1e-12:
                raise RuntimeError("offline bracketing anchors have duplicate coordinates")
            ratio = (record.coordinate - left.coordinate) / spacing
            local = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
            local[position - 1] = 1.0 - ratio
            local[position] = ratio
            for stream_index, (stream_name, _, _) in enumerate(self._stream_ranges):
                configured_blend = self.configured_stream_blends[stream_name]
                correction_coefficients: tuple[float, ...] = ()
                if decision is not None:
                    if stream_name == "audio":
                        configured_blend = decision.audio_blend_weight
                        correction_coefficients = decision.audio_correction_coefficients
                    elif stream_name == "video":
                        configured_blend = decision.video_blend_weight
                        correction_coefficients = decision.video_correction_coefficients
                    else:
                        configured_blend = decision.video_blend_weight
                        if len(decision.audio_correction_coefficients) == len(
                            decision.video_correction_coefficients
                        ):
                            correction_coefficients = tuple(
                                0.5 * (audio + video)
                                for audio, video in zip(
                                    decision.audio_correction_coefficients,
                                    decision.video_correction_coefficients,
                                    strict=True,
                                )
                            )
                for branch in range(self._branch_count):
                    validation_score = self._validation_score_for_interval(
                        position,
                        branch,
                        stream_index,
                    )
                    effective_blend = configured_blend / max(1.0, validation_score)
                    weights = (
                        effective_blend * spectral + (1.0 - effective_blend) * local
                    )
                    if correction_coefficients and any(
                        value != 0.0 for value in correction_coefficients
                    ):
                        correction_started = time.perf_counter()
                        try:
                            weights = weights.clone()
                            required = len(correction_coefficients) + 1
                            if (
                                decision is not None
                                and len(decision.correction_anchor_ids) == required
                            ):
                                try:
                                    correction_positions = [
                                        self._anchor_ids.index(anchor_id)
                                        for anchor_id in decision.correction_anchor_ids
                                    ]
                                except ValueError as exc:
                                    raise RuntimeError(
                                        "offline correction anchor stencil is missing"
                                    ) from exc
                                if (
                                    correction_positions != sorted(correction_positions)
                                    or len(set(correction_positions)) != required
                                    or correction_positions[-1] >= position
                                ):
                                    raise RuntimeError(
                                        "offline correction anchor stencil violates first-pass chronology"
                                    )
                            elif len(correction_coefficients) == 1:
                                correction_positions = [position - 1, position]
                            else:
                                raise RuntimeError(
                                    "offline K=2 correction is missing explicit causal anchor IDs"
                                )
                            if len(correction_coefficients) == 1:
                                gain = correction_coefficients[0]
                                weights[correction_positions[-2]] -= gain
                                weights[correction_positions[-1]] += gain
                            else:
                                g0, g1 = correction_coefficients
                                weights[correction_positions[-3]] -= g1
                                weights[correction_positions[-2]] += -g0 + g1
                                weights[correction_positions[-1]] += g0
                        finally:
                            self.model_aware_offline_correction_seconds += (
                                time.perf_counter() - correction_started
                            )
                        self.model_aware_offline_correction_applications += 1
                    weights_by_step[(record.step_id, branch, stream_index)] = weights
                    effective_blends.append(effective_blend)
                    stream_effective_blends[stream_name].append(effective_blend)
                    if effective_blend < configured_blend - 1e-7:
                        self.attenuated_prediction_count += 1
                        self.attenuated_prediction_counts[stream_name] += 1
                    if effective_blend <= 1e-7:
                        self.local_only_prediction_count += 1
                        self.local_only_prediction_counts[stream_name] += 1
        if effective_blends:
            self.effective_blend_min = min(effective_blends)
            self.effective_blend_mean = sum(effective_blends) / len(effective_blends)
            self.effective_blend_max = max(effective_blends)
        for stream_name, values in stream_effective_blends.items():
            if values:
                self.effective_blend_stream_stats[stream_name] = (
                    min(values),
                    sum(values) / len(values),
                    max(values),
                )
        return weights_by_step

    def predict(
        self,
        step_id: int,
        *,
        rows: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        record = self.archive.steps[int(step_id)]
        anchor = self._anchor_by_step.get(record.step_id)
        if anchor is not None:
            weights = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
            weights[self._anchor_ids.index(record.step_id)] = 1.0
            result = self._forecaster.predict_with_weights(
                weights,
                rows=rows,
                device=device,
                dtype=dtype,
            )
            self._last_prediction_chunk_count = self._forecaster.last_prediction_chunk_count
            return result

        predictions = []
        chunks = 0
        for row in rows:
            weighted_segments = [
                (
                    start,
                    end,
                    self._forecast_weights[(record.step_id, int(row), stream_index)],
                )
                for stream_index, (_, start, end) in enumerate(self._stream_ranges)
            ]
            prediction = self._forecaster.predict_with_segment_weights(
                weighted_segments,
                rows=(int(row),),
                device=device,
                dtype=dtype,
            )
            predictions.append(prediction)
            chunks += self._forecaster.last_prediction_chunk_count
        self._last_prediction_chunk_count = chunks
        return torch.cat(predictions, dim=0)
