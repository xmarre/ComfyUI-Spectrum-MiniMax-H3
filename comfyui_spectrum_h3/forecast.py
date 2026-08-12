from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .model_aware import AnchorEvidence, AnchorEvidenceTiming, StreamAnchorEvidence

_EXACT_HEAD_SAMPLE_ROWS = 32


@dataclass(slots=True)
class _HistoryEntry:
    coordinate: float
    feature_flat: torch.Tensor


@dataclass(slots=True)
class ForecasterSnapshot:
    history: list[_HistoryEntry]
    evidence_history: list[dict[str, torch.Tensor]]
    exact_head_history: list[dict[str, torch.Tensor]]
    exact_head_row_indices: dict[str, torch.Tensor]
    exact_head_shapes: dict[str, tuple[int, int]]
    evidence_segments: tuple[tuple[str, int, int], ...] | None
    feature_shape: tuple[int, ...] | None
    feature_dtype: torch.dtype | None
    history_device: torch.device | None
    generation: int
    factor_generation: int
    design: torch.Tensor | None
    cholesky: torch.Tensor | None
    factorization_count: int
    jitter_attempts: int


class HistoryWeightForecaster:
    """Chebyshev ridge forecasting without full-feature FP32 coefficients.

    Persistent large tensors are model-dtype history snapshots on the configured
    storage device. Regression work is limited to K x (M + 1) design data and a
    (M + 1)^2 factorization.
    """

    def __init__(
        self,
        degree: int = 4,
        ridge_lambda: float = 0.1,
        max_history: int = 8,
        chunk_bytes: int = 32 * 1024 * 1024,
        history_storage: str = "system_ram",
    ) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = int(max_history)
        self.chunk_bytes = int(chunk_bytes)
        self.history_storage = str(history_storage)
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.max_history < max(2, self.degree + 1):
            raise ValueError("max_history is too small for the requested polynomial degree")
        if self.chunk_bytes < 4096:
            raise ValueError("chunk_bytes must be >= 4096")
        if self.history_storage not in {"system_ram", "vram"}:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        self.reset()

    def reset(self) -> None:
        self._history: list[_HistoryEntry] = []
        self._evidence_history: list[dict[str, torch.Tensor]] = []
        self._exact_head_history: list[dict[str, torch.Tensor]] = []
        self._exact_head_row_indices: dict[str, torch.Tensor] = {}
        self._exact_head_shapes: dict[str, tuple[int, int]] = {}
        self._evidence_segments: tuple[tuple[str, int, int], ...] | None = None
        self._feature_shape: tuple[int, ...] | None = None
        self._feature_dtype: torch.dtype | None = None
        self._history_device: torch.device | None = None
        self._generation = 0
        self._factor_generation = -1
        self._design: torch.Tensor | None = None
        self._cholesky: torch.Tensor | None = None
        self._factorization_count = 0
        self._jitter_attempts = 0
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0
        self.model_aware_fit_seconds = 0.0
        self.model_aware_correction_seconds = 0.0
        self.model_aware_exact_head_projection_seconds = 0.0
        self.model_aware_exact_head_projection_calls = 0
        self.model_aware_exact_head_workspace_bytes = 0

    def snapshot(self) -> ForecasterSnapshot:
        return ForecasterSnapshot(
            history=list(self._history),
            evidence_history=[dict(entry) for entry in self._evidence_history],
            exact_head_history=[dict(entry) for entry in self._exact_head_history],
            exact_head_row_indices=dict(self._exact_head_row_indices),
            exact_head_shapes=dict(self._exact_head_shapes),
            evidence_segments=self._evidence_segments,
            feature_shape=self._feature_shape,
            feature_dtype=self._feature_dtype,
            history_device=self._history_device,
            generation=self._generation,
            factor_generation=self._factor_generation,
            design=self._design,
            cholesky=self._cholesky,
            factorization_count=self._factorization_count,
            jitter_attempts=self._jitter_attempts,
        )

    def restore(self, snapshot: ForecasterSnapshot) -> None:
        if not isinstance(snapshot, ForecasterSnapshot):
            raise TypeError("snapshot must be a ForecasterSnapshot")
        self._history = list(snapshot.history)
        self._evidence_history = [dict(entry) for entry in snapshot.evidence_history]
        self._exact_head_history = [dict(entry) for entry in snapshot.exact_head_history]
        self._exact_head_row_indices = dict(snapshot.exact_head_row_indices)
        self._exact_head_shapes = dict(snapshot.exact_head_shapes)
        self._evidence_segments = snapshot.evidence_segments
        self._feature_shape = snapshot.feature_shape
        self._feature_dtype = snapshot.feature_dtype
        self._history_device = snapshot.history_device
        self._generation = snapshot.generation
        self._factor_generation = snapshot.factor_generation
        self._design = snapshot.design
        self._cholesky = snapshot.cholesky
        self._factorization_count = snapshot.factorization_count
        self._jitter_attempts = snapshot.jitter_attempts
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def feature_shape(self) -> tuple[int, ...] | None:
        return self._feature_shape

    @property
    def feature_dtype(self) -> torch.dtype | None:
        return self._feature_dtype

    @property
    def history_device(self) -> torch.device | None:
        return self._history_device

    @property
    def factorization_count(self) -> int:
        return self._factorization_count

    @property
    def jitter_attempts(self) -> int:
        return self._jitter_attempts

    @property
    def persistent_tensor_bytes(self) -> int:
        tensors = [entry.feature_flat for entry in self._history]
        tensors.extend(
            sample
            for entry in self._evidence_history
            for sample in entry.values()
        )
        tensors.extend(
            sample
            for entry in self._exact_head_history
            for sample in entry.values()
        )
        tensors.extend(self._exact_head_row_indices.values())
        tensors.extend(t for t in (self._design, self._cholesky) if t is not None)
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def history_tensor_bytes(self) -> int:
        return sum(entry.feature_flat.numel() * entry.feature_flat.element_size() for entry in self._history)

    @property
    def evidence_tensor_bytes(self) -> int:
        return self.generic_evidence_tensor_bytes + self.exact_head_evidence_tensor_bytes

    @property
    def generic_evidence_tensor_bytes(self) -> int:
        return sum(
            sample.numel() * sample.element_size()
            for entry in self._evidence_history
            for sample in entry.values()
        )

    @property
    def exact_head_evidence_tensor_bytes(self) -> int:
        return sum(
            sample.numel() * sample.element_size()
            for entry in self._exact_head_history
            for sample in entry.values()
        ) + sum(
            indices.numel() * indices.element_size()
            for indices in self._exact_head_row_indices.values()
        )

    def ready(self, minimum: int | None = None) -> bool:
        required = max(2, self.degree + 1, int(minimum or 0))
        return len(self._history) >= required

    @staticmethod
    def chebyshev_design(coordinates: torch.Tensor, degree: int) -> torch.Tensor:
        x = coordinates.reshape(-1, 1).to(device="cpu", dtype=torch.float32)
        columns = [torch.ones_like(x)]
        if degree >= 1:
            columns.append(x)
        for _ in range(2, degree + 1):
            columns.append(2.0 * x * columns[-1] - columns[-2])
        return torch.cat(columns[: degree + 1], dim=1)

    @staticmethod
    def _complete_row_indices(
        row_count: int,
        branch_count: int,
        device: torch.device,
        *,
        limit: int = _EXACT_HEAD_SAMPLE_ROWS,
    ) -> torch.Tensor:
        rows = int(row_count)
        branches = int(branch_count)
        if rows < 1 or branches < 1:
            raise ValueError("exact head evidence requires non-empty stream rows and branches")
        count = min(rows, max(1, int(limit) // branches))
        positions = torch.arange(count, device=device, dtype=torch.int64)
        return (((2 * positions + 1) * rows) // (2 * count)).clamp_max(rows - 1)

    @staticmethod
    def _project_hidden_rows(
        feature: torch.Tensor,
        start_row: int,
        end_row: int,
        row_indices: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 3:
            raise ValueError("exact head evidence requires [branch, row, hidden] features")
        hidden = int(feature.shape[-1])
        if (
            head_weight.ndim != 2
            or int(head_weight.shape[1]) != hidden
            or head_weight.device != feature.device
            or head_weight.dtype != torch.float32
        ):
            raise ValueError("FinalLayer head operator is incompatible with cached hidden rows")
        if row_indices.ndim != 1 or row_indices.device != feature.device:
            raise ValueError("exact head row indices are incompatible with cached hidden rows")
        selected = feature[:, int(start_row) : int(end_row)].index_select(
            1,
            row_indices,
        )
        projected = torch.matmul(
            selected.to(dtype=torch.float32),
            head_weight.transpose(0, 1),
        )
        return projected.contiguous()

    def update(
        self,
        coordinate: float,
        feature: torch.Tensor,
        *,
        take_ownership: bool = False,
        evidence_segments: Sequence[tuple[str, int, int]] | None = None,
        exact_head_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not torch.is_tensor(feature) or not feature.dtype.is_floating_point:
            raise ValueError("Spectrum history features must be floating-point tensors")
        shape = tuple(int(v) for v in feature.shape)
        if len(shape) < 2:
            raise ValueError("Spectrum history features must have a branch dimension and feature dimensions")
        if self._feature_shape is None:
            self._feature_shape = shape
            self._feature_dtype = feature.dtype
        elif shape != self._feature_shape:
            raise ValueError(f"feature shape changed from {self._feature_shape} to {shape}")
        elif feature.dtype != self._feature_dtype:
            raise ValueError(f"feature dtype changed from {self._feature_dtype} to {feature.dtype}")

        detached = feature.detach()
        sampled_evidence = None
        exact_head_evidence: dict[str, torch.Tensor] = {}
        normalized_segments = None
        clear_evidence = evidence_segments is None and bool(self._evidence_history)
        if evidence_segments is not None:
            normalized_segments = tuple(
                (str(name), int(start), int(end))
                for name, start, end in evidence_segments
            )
            if (
                self._evidence_segments is not None
                and normalized_segments != self._evidence_segments
            ):
                raise ValueError("model-aware evidence segments changed during actual history")
            if self._history and len(self._evidence_history) != len(self._history):
                raise ValueError("model-aware evidence sampling cannot start after history")
            if any(
                sample.device != detached.device
                for entry in self._evidence_history
                for sample in entry.values()
            ):
                raise ValueError("model-aware evidence device changed during actual history")
            sampled_evidence = {
                name: self._sample_segment_device(detached, start, end)
                for name, start, end in normalized_segments
            }
            head_weights = exact_head_weights or {}
            projection_started = time.perf_counter() if head_weights else None
            for name, start, end in normalized_segments:
                head_weight = head_weights.get(name)
                if head_weight is None:
                    continue
                row_count = int(end) - int(start)
                indices = self._exact_head_row_indices.get(name)
                if indices is None:
                    indices = self._complete_row_indices(
                        row_count,
                        int(detached.shape[0]),
                        detached.device,
                    )
                    self._exact_head_row_indices[name] = indices
                shape = (int(head_weight.shape[0]), int(head_weight.shape[1]))
                previous_shape = self._exact_head_shapes.get(name)
                if previous_shape is not None and previous_shape != shape:
                    raise ValueError("FinalLayer head operator shape changed during actual history")
                self._exact_head_shapes[name] = shape
                exact_head_evidence[name] = self._project_hidden_rows(
                    detached,
                    start,
                    end,
                    indices,
                    head_weight,
                )
                self.model_aware_exact_head_workspace_bytes = max(
                    self.model_aware_exact_head_workspace_bytes,
                    int(detached.shape[0])
                    * int(indices.numel())
                    * (int(detached.shape[-1]) + int(head_weight.shape[0]))
                    * torch.tensor([], dtype=torch.float32).element_size(),
                )
                self.model_aware_exact_head_projection_calls += 1
            if projection_started is not None:
                self.model_aware_exact_head_projection_seconds += (
                    time.perf_counter() - projection_started
                )
            if self._history:
                if len(self._exact_head_history) not in {0, len(self._history)}:
                    raise ValueError("exact head evidence history is not aligned")
                if self._exact_head_history and (
                    set(exact_head_evidence) != set(self._exact_head_history[0])
                ):
                    raise ValueError("exact head evidence streams changed during actual history")
                if not self._exact_head_history and exact_head_evidence:
                    raise ValueError("exact head evidence sampling cannot start after history")
        storage_device = torch.device("cpu") if self.history_storage == "system_ram" else detached.device
        if self._history_device is None:
            self._history_device = storage_device
        elif storage_device != self._history_device:
            raise ValueError(f"history device changed from {self._history_device} to {storage_device}")

        if take_ownership and detached.device == storage_device and detached.is_contiguous():
            archived = detached.reshape(-1)
        else:
            archived = (
                detached.to(device=storage_device, dtype=self._feature_dtype, copy=True)
                .contiguous()
                .reshape(-1)
            )
        self._history.append(_HistoryEntry(float(coordinate), archived))
        if sampled_evidence is not None:
            self._evidence_segments = normalized_segments
            self._evidence_history.append(sampled_evidence)
            if exact_head_evidence:
                self._exact_head_history.append(exact_head_evidence)
        elif clear_evidence:
            self._evidence_history.clear()
            self._exact_head_history.clear()
            self._exact_head_row_indices.clear()
            self._exact_head_shapes.clear()
            self._evidence_segments = None
        if len(self._history) > self.max_history:
            self._history.pop(0)
            if self._evidence_history:
                self._evidence_history.pop(0)
            if self._exact_head_history:
                self._exact_head_history.pop(0)
        self._generation += 1
        self._design = None
        self._cholesky = None

    def _ensure_factorization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ready():
            raise RuntimeError("Spectrum forecaster does not have enough actual history")
        if (
            self._factor_generation == self._generation
            and self._design is not None
            and self._cholesky is not None
        ):
            return self._design, self._cholesky

        coords = torch.tensor([entry.coordinate for entry in self._history], dtype=torch.float32)
        design = self.chebyshev_design(coords, self.degree)
        cholesky, jitter_attempts = self._factorize_design(
            design,
            self.ridge_lambda,
            failure_message="Spectrum ridge factorization failed after bounded jitter attempts",
        )

        self._jitter_attempts += jitter_attempts
        self._factorization_count += 1
        self._design = design
        self._cholesky = cholesky
        self._factor_generation = self._generation
        return design, cholesky

    @staticmethod
    def _factorize_design(
        design: torch.Tensor,
        ridge_lambda: float,
        *,
        failure_message: str,
    ) -> tuple[torch.Tensor, int]:
        gram = design.transpose(0, 1) @ design
        eye = torch.eye(gram.shape[0], dtype=torch.float32)
        base = gram + float(ridge_lambda) * eye
        diagonal_scale = max(float(gram.diag().abs().mean().item()), 1.0)
        jitters = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)
        last_error: RuntimeError | None = None
        for attempt, multiplier in enumerate(jitters):
            try:
                cholesky = torch.linalg.cholesky(
                    base + (diagonal_scale * multiplier) * eye
                )
                return cholesky, attempt
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(failure_message) from last_error

    def _spectral_weights(self, coordinate: float) -> torch.Tensor:
        design, cholesky = self._ensure_factorization()
        phi = self.chebyshev_design(torch.tensor([float(coordinate)]), self.degree)
        solved = torch.cholesky_solve(design.transpose(0, 1), cholesky)
        return (phi @ solved).reshape(-1)

    def spectral_weights(self, coordinate: float) -> torch.Tensor:
        return self._spectral_weights(coordinate)

    def _spectral_weights_configured(
        self,
        coordinate: float,
        *,
        degree: int,
        ridge_lambda: float,
    ) -> torch.Tensor:
        resolved_degree = int(degree)
        resolved_ridge = float(ridge_lambda)
        if resolved_degree == self.degree and math.isclose(
            resolved_ridge,
            self.ridge_lambda,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return self._spectral_weights(coordinate)
        if resolved_degree < 1 or resolved_degree + 1 > len(self._history):
            raise ValueError("adaptive degree is invalid for the available actual history")
        if not math.isfinite(resolved_ridge) or resolved_ridge < 0.0:
            raise ValueError("adaptive ridge_lambda must be finite and nonnegative")
        coords = torch.tensor([entry.coordinate for entry in self._history], dtype=torch.float32)
        design = self.chebyshev_design(coords, resolved_degree)
        cholesky, jitter_attempts = self._factorize_design(
            design,
            resolved_ridge,
            failure_message="adaptive Spectrum ridge factorization failed",
        )
        self._jitter_attempts += jitter_attempts
        self._factorization_count += 1
        phi = self.chebyshev_design(torch.tensor([float(coordinate)]), resolved_degree)
        solved = torch.cholesky_solve(design.transpose(0, 1), cholesky)
        return (phi @ solved).reshape(-1)

    def _linear_weights(self, coordinate: float) -> torch.Tensor:
        weights = torch.zeros(len(self._history), dtype=torch.float32)
        if len(self._history) == 1:
            weights[-1] = 1.0
            return weights
        previous = self._history[-2].coordinate
        latest = self._history[-1].coordinate
        spacing = latest - previous
        if abs(spacing) <= 1e-12:
            weights[-1] = 1.0
            return weights
        ratio = (float(coordinate) - latest) / spacing
        weights[-2] = -ratio
        weights[-1] = 1.0 + ratio
        return weights

    def combined_weights(self, coordinate: float, blend_weight: float) -> torch.Tensor:
        blend = float(blend_weight)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if blend <= 1e-12:
            return self._linear_weights(coordinate)
        spectral = self._spectral_weights(coordinate)
        if blend >= 1.0 - 1e-12:
            return spectral
        return blend * spectral + (1.0 - blend) * self._linear_weights(coordinate)

    def model_aware_weights(
        self,
        coordinate: float,
        blend_weight: float,
        *,
        degree: int,
        ridge_lambda: float,
        correction_gain: float = 0.0,
    ) -> torch.Tensor:
        fit_started = time.perf_counter()
        blend = float(blend_weight)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if blend <= 1e-12:
            weights = self._linear_weights(coordinate)
        else:
            spectral = self._spectral_weights_configured(
                coordinate,
                degree=int(degree),
                ridge_lambda=float(ridge_lambda),
            )
            weights = spectral if blend >= 1.0 - 1e-12 else blend * spectral + (1.0 - blend) * self._linear_weights(coordinate)
        self.model_aware_fit_seconds += time.perf_counter() - fit_started
        gain = float(correction_gain)
        if not math.isfinite(gain) or not -0.25 <= gain <= 0.25:
            raise ValueError("model-aware correction gain must be finite and in [-0.25, 0.25]")
        if gain != 0.0:
            correction_started = time.perf_counter()
            if len(self._history) < 2:
                raise RuntimeError("model-aware correction requires two actual history entries")
            weights = weights.clone()
            weights[-2] -= gain
            weights[-1] += gain
            self.model_aware_correction_seconds += time.perf_counter() - correction_started
        return weights

    def fit_condition(self, *, degree: int | None = None) -> float:
        resolved_degree = self.degree if degree is None else int(degree)
        if resolved_degree < 1 or resolved_degree + 1 > len(self._history):
            return float("inf")
        coords = torch.tensor([entry.coordinate for entry in self._history], dtype=torch.float32)
        design = self.chebyshev_design(coords, resolved_degree)
        try:
            condition = float(torch.linalg.cond(design).item())
        except RuntimeError:
            return float("inf")
        return condition if math.isfinite(condition) else float("inf")

    @staticmethod
    def _sample_segment_device(
        feature: torch.Tensor,
        start_row: int,
        end_row: int,
        *,
        limit: int = 4096,
    ) -> torch.Tensor:
        selected = []
        per_branch = max(1, int(limit) // max(1, int(feature.shape[0])))
        for branch in range(int(feature.shape[0])):
            flat = feature[branch, start_row:end_row].detach().reshape(-1)
            if flat.numel() == 0:
                continue
            stride = max(1, flat.numel() // per_branch)
            selected.append(flat[::stride][:per_branch].to(torch.float32))
        if not selected:
            raise ValueError("cannot sample an empty feature segment")
        return torch.cat(selected).contiguous()

    @staticmethod
    def _sample_channel_sensitivity(
        feature_shape: tuple[int, ...],
        start_row: int,
        end_row: int,
        sensitivity: torch.Tensor,
        *,
        limit: int = 4096,
    ) -> torch.Tensor:
        hidden = int(feature_shape[-1])
        if sensitivity.ndim != 1 or int(sensitivity.shape[0]) != hidden:
            raise ValueError("FinalLayer head sensitivity does not match the cached hidden width")
        branch_count = int(feature_shape[0])
        flat_elements = (int(end_row) - int(start_row)) * hidden
        per_branch = max(1, int(limit) // max(1, branch_count))
        stride = max(1, flat_elements // per_branch)
        positions = torch.arange(
            0,
            flat_elements,
            stride,
            device=sensitivity.device,
        )[:per_branch]
        sampled = sensitivity.index_select(0, positions.remainder(hidden))
        return sampled.repeat(branch_count).contiguous()

    def sampled_anchor_evidence(
        self,
        coordinate: float,
        actual_feature: torch.Tensor,
        stream_parameters: Sequence[
            tuple[str, int, int, float, float, float, float, float]
        ],
        *,
        degree: int,
        ridge_lambda: float,
        stream_diagonals: dict[str, torch.Tensor] | None = None,
        exact_head_weights: dict[str, torch.Tensor] | None = None,
    ) -> AnchorEvidence | None:
        if len(self._history) < 2 or self._feature_shape is None:
            return None
        if tuple(actual_feature.shape) != self._feature_shape:
            raise ValueError("actual feature shape changed during model-aware evidence sampling")
        if len(self._evidence_history) != len(self._history):
            raise RuntimeError("device-local model-aware evidence history is not aligned")
        stream_evidence: dict[str, StreamAnchorEvidence] = {}
        weight_fit_seconds = 0.0
        sample_index_seconds = 0.0
        scalar_transfer_seconds = 0.0
        reduction_seconds = 0.0
        exact_head_projection_seconds = 0.0
        for (
            name,
            start,
            end,
            blend,
            model_gain,
            generic_gain,
            diagonal_candidate_gain,
            model_candidate_gain,
        ) in stream_parameters:
            fit_started = time.perf_counter()
            raw_weights = self.model_aware_weights(
                coordinate,
                blend,
                degree=degree,
                ridge_lambda=ridge_lambda,
                correction_gain=0.0,
            )
            weight_fit_seconds += time.perf_counter() - fit_started
            history_samples = [entry[str(name)] for entry in self._evidence_history]
            selection_started = time.perf_counter()
            actual = self._sample_segment_device(
                actual_feature,
                int(start),
                int(end),
            )
            if any(sample.device != actual.device for sample in history_samples):
                raise RuntimeError("model-aware evidence device changed during actual history")
            diagonal = None if stream_diagonals is None else stream_diagonals.get(str(name))
            if diagonal is not None and diagonal.device != actual.device:
                raise ValueError("FinalLayer head sensitivity is on the wrong evidence device")
            sampled_diagonal = (
                None
                if diagonal is None
                else self._sample_channel_sensitivity(
                    self._feature_shape,
                    int(start),
                    int(end),
                    diagonal,
                )
            )
            head_weight = (
                None
                if exact_head_weights is None
                else exact_head_weights.get(str(name))
            )
            actual_head = None
            history_head: list[torch.Tensor] = []
            if head_weight is not None:
                if len(self._exact_head_history) != len(self._history):
                    raise RuntimeError("exact head evidence history is not aligned")
                indices = self._exact_head_row_indices.get(str(name))
                if indices is None:
                    raise RuntimeError("exact head evidence row indices are missing")
                history_head = [
                    entry[str(name)] for entry in self._exact_head_history
                ]
                if any(sample.device != actual.device for sample in history_head):
                    raise RuntimeError("exact head evidence device changed during actual history")
                head_projection_started = time.perf_counter()
                actual_head = self._project_hidden_rows(
                    actual_feature,
                    int(start),
                    int(end),
                    indices,
                    head_weight,
                )
                self.model_aware_exact_head_workspace_bytes = max(
                    self.model_aware_exact_head_workspace_bytes,
                    int(actual_feature.shape[0])
                    * int(indices.numel())
                    * (int(actual_feature.shape[-1]) + int(head_weight.shape[0]))
                    * torch.tensor([], dtype=torch.float32).element_size(),
                )
                head_projection_elapsed = time.perf_counter() - head_projection_started
                exact_head_projection_seconds += head_projection_elapsed
                self.model_aware_exact_head_projection_seconds += head_projection_elapsed
                self.model_aware_exact_head_projection_calls += 1
            sample_index_seconds += time.perf_counter() - selection_started
            reduction_started = time.perf_counter()
            predicted = torch.zeros_like(actual)
            for weight, sample in zip(raw_weights.tolist(), history_samples, strict=True):
                if weight != 0.0:
                    predicted.add_(sample, alpha=float(weight))
            latest = history_samples[-1]
            previous = history_samples[-2]
            delta = latest - previous
            residual = actual - predicted
            epsilon = torch.sqrt(torch.mean(actual.square())).mul(1e-6).clamp_min(
                torch.finfo(torch.float32).eps
            )
            hold_error = actual - latest
            hold_rms = torch.sqrt(torch.mean(hold_error.square())).clamp_min(epsilon)
            forecast_rms = torch.sqrt(torch.mean(residual.square()))
            dot_epsilon = epsilon.square() * max(1, int(delta.numel()))
            projection = torch.dot(residual, delta) / torch.dot(delta, delta).clamp_min(
                dot_epsilon
            )
            if sampled_diagonal is None:
                diagonal_projection = projection
            else:
                weighted_delta = sampled_diagonal * delta
                diagonal_projection = torch.dot(residual, weighted_delta) / torch.dot(
                    delta,
                    weighted_delta,
                ).clamp_min(dot_epsilon)
            predicted_head = None
            latest_head = previous_head = delta_head = residual_head = None
            if actual_head is None:
                model_projection = projection
            else:
                predicted_head = torch.zeros_like(actual_head)
                for weight, sample in zip(
                    raw_weights.tolist(),
                    history_head,
                    strict=True,
                ):
                    if weight != 0.0:
                        predicted_head.add_(sample, alpha=float(weight))
                latest_head = history_head[-1]
                previous_head = history_head[-2]
                delta_head = latest_head - previous_head
                residual_head = actual_head - predicted_head
                flattened_delta_head = delta_head.reshape(-1)
                flattened_residual_head = residual_head.reshape(-1)
                head_epsilon = (
                    torch.sqrt(torch.mean(actual_head.square()))
                    .mul(1e-6)
                    .clamp_min(torch.finfo(torch.float32).eps)
                )
                head_dot_epsilon = head_epsilon.square() * max(
                    1,
                    int(flattened_delta_head.numel()),
                )
                model_projection = torch.dot(
                    flattened_residual_head,
                    flattened_delta_head,
                ) / torch.dot(
                    flattened_delta_head,
                    flattened_delta_head,
                ).clamp_min(head_dot_epsilon)
            forecast_ratio = forecast_rms / hold_rms
            model_predicted = predicted + float(model_gain) * delta
            generic_predicted = predicted + float(generic_gain) * delta
            diagonal_candidate_predicted = (
                predicted + float(diagonal_candidate_gain) * delta
            )
            model_candidate_predicted = predicted + float(model_candidate_gain) * delta
            model_error = actual - model_predicted
            generic_error = actual - generic_predicted
            diagonal_candidate_error = actual - diagonal_candidate_predicted
            model_candidate_error = actual - model_candidate_predicted
            model_ratio = torch.sqrt(torch.mean(model_error.square())) / hold_rms
            generic_ratio = torch.sqrt(torch.mean(generic_error.square())) / hold_rms
            diagonal_candidate_ratio = (
                torch.sqrt(torch.mean(diagonal_candidate_error.square())) / hold_rms
            )
            model_candidate_ratio = (
                torch.sqrt(torch.mean(model_candidate_error.square())) / hold_rms
            )
            if actual_head is None:
                model_head_ratio = model_ratio
                generic_head_ratio = generic_ratio
                diagonal_candidate_head_ratio = diagonal_candidate_ratio
                model_candidate_head_ratio = model_candidate_ratio
            else:
                assert predicted_head is not None
                assert latest_head is not None
                assert delta_head is not None
                head_hold_error = actual_head - latest_head
                head_hold_rms = torch.sqrt(
                    torch.mean(head_hold_error.square())
                ).clamp_min(head_epsilon)
                model_head_ratio = torch.sqrt(
                    torch.mean(
                        (
                            actual_head
                            - (predicted_head + float(model_gain) * delta_head)
                        ).square()
                    )
                ) / head_hold_rms
                generic_head_ratio = torch.sqrt(
                    torch.mean(
                        (
                            actual_head
                            - (predicted_head + float(generic_gain) * delta_head)
                        ).square()
                    )
                ) / head_hold_rms
                diagonal_candidate_head_ratio = torch.sqrt(
                    torch.mean(
                        (
                            actual_head
                            - (
                                predicted_head
                                + float(diagonal_candidate_gain) * delta_head
                            )
                        ).square()
                    )
                ) / head_hold_rms
                model_candidate_head_ratio = torch.sqrt(
                    torch.mean(
                        (
                            actual_head
                            - (
                                predicted_head
                                + float(model_candidate_gain) * delta_head
                            )
                        ).square()
                    )
                ) / head_hold_rms
            if len(history_samples) >= 3:
                curvature = latest - 2.0 * previous + history_samples[-3]
                curvature_ratio = torch.sqrt(torch.mean(curvature.square())) / torch.sqrt(
                    torch.mean(delta.square())
                ).clamp_min(epsilon)
            else:
                curvature_ratio = torch.zeros((), dtype=torch.float32, device=actual.device)
            scalar_values = torch.stack(
                (
                    forecast_ratio,
                    curvature_ratio,
                    projection,
                    diagonal_projection,
                    model_projection,
                    model_ratio,
                    generic_ratio,
                    diagonal_candidate_ratio,
                    model_candidate_ratio,
                    model_head_ratio,
                    generic_head_ratio,
                    diagonal_candidate_head_ratio,
                    model_candidate_head_ratio,
                )
            )
            reduction_seconds += time.perf_counter() - reduction_started
            transfer_started = time.perf_counter()
            (
                forecast_ratio_value,
                curvature_ratio_value,
                projection_value,
                diagonal_projection_value,
                model_projection_value,
                model_ratio_value,
                generic_ratio_value,
                diagonal_candidate_ratio_value,
                model_candidate_ratio_value,
                model_head_ratio_value,
                generic_head_ratio_value,
                diagonal_candidate_head_ratio_value,
                model_candidate_head_ratio_value,
            ) = scalar_values.detach().to(device="cpu").tolist()
            scalar_transfer_seconds += time.perf_counter() - transfer_started
            stream_evidence[str(name)] = StreamAnchorEvidence(
                forecast_ratio=forecast_ratio_value,
                curvature_ratio=curvature_ratio_value,
                residual_projection=projection_value,
                model_corrected_ratio=model_ratio_value,
                generic_corrected_ratio=generic_ratio_value,
                diagonal_projection=diagonal_projection_value,
                model_projection=model_projection_value,
                diagonal_candidate_ratio=diagonal_candidate_ratio_value,
                model_candidate_ratio=model_candidate_ratio_value,
                model_corrected_head_ratio=model_head_ratio_value,
                generic_corrected_head_ratio=generic_head_ratio_value,
                diagonal_candidate_head_ratio=diagonal_candidate_head_ratio_value,
                model_candidate_head_ratio=model_candidate_head_ratio_value,
            )
        if "packed" in stream_evidence:
            audio = video = stream_evidence["packed"]
        else:
            audio = stream_evidence.get("audio", StreamAnchorEvidence())
            video = stream_evidence.get("video", StreamAnchorEvidence())
        values = [
            value
            for evidence in stream_evidence.values()
            for value in (
                evidence.forecast_ratio,
                evidence.curvature_ratio,
                evidence.residual_projection,
                evidence.diagonal_projection,
                evidence.model_projection,
                evidence.model_corrected_ratio,
                evidence.generic_corrected_ratio,
                evidence.diagonal_candidate_ratio,
                evidence.model_candidate_ratio,
                evidence.model_corrected_head_ratio,
                evidence.generic_corrected_head_ratio,
                evidence.diagonal_candidate_head_ratio,
                evidence.model_candidate_head_ratio,
            )
        ]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("model-aware anchor evidence is nonfinite")
        fit_condition_started = time.perf_counter()
        fit_condition = self.fit_condition(degree=degree)
        fit_condition_seconds = time.perf_counter() - fit_condition_started
        return AnchorEvidence(
            forecast_ratio=max(audio.forecast_ratio, video.forecast_ratio),
            curvature_ratio=max(audio.curvature_ratio, video.curvature_ratio),
            fit_condition=fit_condition,
            audio_projection=audio.residual_projection,
            video_projection=video.residual_projection,
            model_corrected_ratio=max(
                audio.model_corrected_ratio,
                video.model_corrected_ratio,
            ),
            generic_corrected_ratio=max(
                audio.generic_corrected_ratio,
                video.generic_corrected_ratio,
            ),
            audio=audio,
            video=video,
            timing=AnchorEvidenceTiming(
                weight_fit_seconds=weight_fit_seconds,
                sample_index_seconds=sample_index_seconds,
                device_transfer_seconds=(
                    scalar_transfer_seconds
                ),
                sensitivity_transfer_seconds=0.0,
                scalar_transfer_seconds=scalar_transfer_seconds,
                reduction_seconds=reduction_seconds,
                exact_head_projection_seconds=exact_head_projection_seconds,
                fit_condition_seconds=fit_condition_seconds,
            ),
        )

    def _chunk_elements(self, device: torch.device) -> int:
        target_bytes = self.chunk_bytes
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                target_bytes = min(target_bytes, max(4 * 1024 * 1024, int(free_bytes) // 32))
            except (RuntimeError, TypeError):
                pass
        return max(1024, target_bytes // torch.tensor([], dtype=torch.float32).element_size())

    def _normalize_rows(self, rows: Sequence[int] | None) -> tuple[int, ...]:
        if self._feature_shape is None:
            raise RuntimeError("Spectrum forecaster has no feature shape")
        branch_count = self._feature_shape[0]
        resolved = tuple(range(branch_count)) if rows is None else tuple(int(v) for v in rows)
        if not resolved:
            raise ValueError("row selection cannot be empty")
        if any(row < 0 or row >= branch_count for row in resolved):
            raise ValueError(f"row selection {resolved} is outside branch count {branch_count}")
        return resolved

    def predict(
        self,
        coordinate: float,
        blend_weight: float,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        weights = self.combined_weights(coordinate, blend_weight)
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_segments(
        self,
        coordinate: float,
        segment_blends: Sequence[tuple[int, int, float]],
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        weighted_segments = [
            (start, end, self.combined_weights(coordinate, blend_weight))
            for start, end, blend_weight in segment_blends
        ]
        return self._predict_with_segment_weights(
            weighted_segments,
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def predict_one_point_hold(
        self,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if len(self._history) != 1:
            raise RuntimeError("one-point hold requires exactly one actual history entry")
        weights = torch.ones(1, dtype=torch.float32)
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_latest_hold(
        self,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self._history:
            raise RuntimeError("latest hold requires at least one actual history entry")
        weights = torch.zeros(len(self._history), dtype=torch.float32)
        weights[-1] = 1.0
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_with_weights(
        self,
        weights: torch.Tensor,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_with_segment_weights(
        self,
        weighted_segments: Sequence[tuple[int, int, torch.Tensor]],
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._predict_with_segment_weights(
            weighted_segments,
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def _predict_with_weights(
        self,
        weights: torch.Tensor,
        *,
        rows: Sequence[int] | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        if len(self._feature_shape) < 2:
            raise RuntimeError("Spectrum forecaster feature shape is invalid")
        feature_rows = self._feature_shape[1] if len(self._feature_shape) >= 3 else 1
        return self._predict_with_segment_weights(
            ((0, feature_rows, weights),),
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def _predict_with_segment_weights(
        self,
        weighted_segments: Sequence[tuple[int, int, torch.Tensor]],
        *,
        rows: Sequence[int] | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        feature_rows = self._feature_shape[1] if len(self._feature_shape) >= 3 else 1
        normalized_segments = []
        expected_start = 0
        for start, end, weights in weighted_segments:
            start = int(start)
            end = int(end)
            if start != expected_start or end <= start or end > feature_rows:
                raise ValueError("prediction segments must cover feature rows contiguously")
            if weights.ndim != 1 or weights.numel() != len(self._history):
                raise ValueError("prediction weights must match actual history length")
            normalized_segments.append((start, end, tuple(float(weight) for weight in weights.tolist())))
            expected_start = end
        if expected_start != feature_rows:
            raise ValueError("prediction segments must cover every feature row")

        resolved_rows = self._normalize_rows(rows)
        target_device = torch.device(device or "cpu")
        target_dtype = dtype or self._feature_dtype
        if not target_dtype.is_floating_point:
            raise ValueError("prediction dtype must be floating point")

        tail_shape = self._feature_shape[1:]
        tail_numel = 1
        for size in tail_shape:
            tail_numel *= size
        if len(self._feature_shape) >= 3:
            row_numel = 1
            for size in tail_shape[1:]:
                row_numel *= size
        else:
            row_numel = tail_numel
        result = torch.empty((len(resolved_rows), *tail_shape), device=target_device, dtype=target_dtype)
        result_flat = result.reshape(-1)
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0

        for target_row, source_row in enumerate(resolved_rows):
            source_base = source_row * tail_numel
            target_base = target_row * tail_numel
            for start, end, weight_scalars in normalized_segments:
                segment_start = start * row_numel
                segment_numel = (end - start) * row_numel
                chunk_elements = min(self._chunk_elements(target_device), segment_numel)
                for offset in range(0, segment_numel, chunk_elements):
                    length = min(chunk_elements, segment_numel - offset)
                    accumulator = torch.zeros(length, device=target_device, dtype=torch.float32)
                    for scalar, entry in zip(weight_scalars, self._history, strict=True):
                        if scalar == 0.0:
                            continue
                        source = entry.feature_flat.narrow(
                            0,
                            source_base + segment_start + offset,
                            length,
                        )
                        source_fp32 = source.to(device=target_device, dtype=torch.float32, non_blocking=False)
                        accumulator.add_(source_fp32, alpha=scalar)
                    result_flat.narrow(
                        0,
                        target_base + segment_start + offset,
                        length,
                    ).copy_(accumulator.to(target_dtype))
                    self.last_prediction_chunk_count += 1
                    self.last_prediction_max_fp32_elements = max(
                        self.last_prediction_max_fp32_elements, accumulator.numel()
                    )
        return result
