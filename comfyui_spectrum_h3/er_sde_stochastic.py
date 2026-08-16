from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import logging
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

LOG = logging.getLogger(__name__)


class ERSDETrackingError(RuntimeError):
    """The reviewed ER-SDE stochastic-state contract could not be maintained."""


class ERSDEDenseOutputError(ERSDETrackingError):
    """The causal ER-SDE solver-space dense-output contract could not be maintained."""


@dataclass(frozen=True, slots=True)
class ERSDEStepDescriptor:
    run_id: int
    step_id: int
    mode: str
    replay_source_actual: bool | None
    requires_compensation: bool


@dataclass(slots=True)
class _PendingIncrement:
    target_step_id: int
    value: torch.Tensor


@dataclass(slots=True)
class _DenoisedAnchor:
    step_id: int
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class ERSDEDenseOutputPrediction:
    value: torch.Tensor
    mode: str
    anchor_steps: tuple[int, ...]
    alpha: float | torch.Tensor


def function_node_ast_digest(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Return the normalized digest shared by runtime and source-contract tests."""
    normalized = copy.deepcopy(function_node)
    normalized.decorator_list = []
    # Python 3.13 changed ast.dump() to omit empty fields by default. The reviewed
    # source digests were generated from the full-field representation used by
    # Python <=3.12. Force that representation when the runtime supports the new
    # show_empty switch so identical native source keeps identical provenance.
    try:
        dumped = ast.dump(normalized, include_attributes=False, show_empty=True)
    except TypeError:
        dumped = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def function_ast_digest(
    function: Callable[..., Any],
    *,
    unwrap: bool = True,
) -> str | None:
    """Return a stable semantic-source digest, or None when source is unavailable."""
    try:
        inspected = inspect.unwrap(function) if unwrap else function
        source = inspect.getsource(inspected)
        module = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(module.body) != 1 or not isinstance(
        module.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return None
    return function_node_ast_digest(module.body[0])


def native_default_er_sde_noise_scaler(value: torch.Tensor) -> torch.Tensor:
    """Reviewed native sample_er_sde default, kept byte-for-byte mathematically equal."""
    return value * ((value ** 0.3).exp() + 10.0)


class ERSDEStochasticTracker:
    """Track native ER-SDE stochastic increments and bounded solver-space anchors."""

    def __init__(
        self,
        *,
        noise_sampler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise_scaler: Callable[[torch.Tensor], torch.Tensor],
        effective_s_noise: float,
        max_stage: int,
        debug: bool,
        run_id: int,
    ) -> None:
        if not math.isfinite(float(effective_s_noise)) or effective_s_noise <= 0.0:
            raise ValueError("ER-SDE stochastic tracking requires positive finite s_noise")
        self._base_noise_sampler = noise_sampler
        self._base_noise_scaler = noise_scaler
        self.effective_s_noise = float(effective_s_noise)
        self.max_stage = int(max_stage)
        self.debug = bool(debug)
        self.run_id = int(run_id)
        self._scaler_prefix: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._scaler_calls = 0
        self._noise_calls = 0
        self._pending: _PendingIncrement | None = None
        self._solver_coordinates: dict[int, torch.Tensor] = {}
        self._denoised_anchors: list[_DenoisedAnchor] = []

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_step_id(self) -> int | None:
        return None if self._pending is None else self._pending.target_step_id

    @property
    def noise_calls(self) -> int:
        return self._noise_calls

    @property
    def denoised_anchor_steps(self) -> tuple[int, ...]:
        return tuple(anchor.step_id for anchor in self._denoised_anchors)

    def noise_scaler(self, value: torch.Tensor) -> torch.Tensor:
        result = self._base_noise_scaler(value)
        if self._scaler_calls < 2:
            if not torch.is_tensor(value) or not torch.is_tensor(result):
                raise ERSDETrackingError(
                    "reviewed ER-SDE noise_scaler did not preserve tensor semantics"
                )
            self._scaler_prefix.append((value, result))
        self._scaler_calls += 1
        return result

    @staticmethod
    def _coordinate_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.numel() != 1:
            raise ERSDETrackingError(f"ER-SDE {label} is not a scalar tensor")
        return value.detach().clone()

    def _record_solver_coordinate(
        self,
        step_id: int,
        coordinate: torch.Tensor,
    ) -> None:
        existing = self._solver_coordinates.get(int(step_id))
        if existing is None:
            self._solver_coordinates[int(step_id)] = coordinate
            return
        if self.debug and not bool(
            torch.isclose(existing, coordinate, rtol=1e-6, atol=1e-9).item()
        ):
            raise ERSDETrackingError(
                f"ER-SDE solver coordinate changed for step {step_id}"
            )

    def noise_sampler(
        self,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        noise = self._base_noise_sampler(sigma, sigma_next)
        if not torch.is_tensor(noise):
            raise ERSDETrackingError("ER-SDE noise_sampler did not return a tensor")
        if self._pending is not None:
            raise ERSDETrackingError(
                "ER-SDE produced a second stochastic increment before the first was consumed"
            )
        if len(self._scaler_prefix) != 2:
            raise ERSDETrackingError(
                "ER-SDE noise_scaler call ordering no longer matches the reviewed solver"
            )

        # Native sample_er_sde evaluates r's numerator first:
        # noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s).
        er_lambda_t, scaled_t = self._scaler_prefix[0]
        er_lambda_s, scaled_s = self._scaler_prefix[1]
        source_step_id = self._noise_calls
        target_step_id = source_step_id + 1
        self._record_solver_coordinate(
            source_step_id,
            self._coordinate_tensor(er_lambda_s, label="er_lambda_s"),
        )
        self._record_solver_coordinate(
            target_step_id,
            self._coordinate_tensor(er_lambda_t, label="er_lambda_t"),
        )

        r = scaled_t / scaled_s
        alpha_t = sigma_next / er_lambda_t
        stochastic_scale = (
            er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2
        ).sqrt().nan_to_num(nan=0.0)

        # Collapse scalar terms before the only latent-sized multiply. The retained
        # tensor owns only the packed increment needed by the following model call.
        coefficient = alpha_t * self.effective_s_noise * stochastic_scale
        increment = noise * coefficient
        if increment.shape != noise.shape:
            raise ERSDETrackingError(
                "ER-SDE stochastic increment shape changed through scalar scaling"
            )
        self._noise_calls += 1
        self._pending = _PendingIncrement(target_step_id, increment)
        self._scaler_prefix.clear()
        self._scaler_calls = 0
        return noise

    def _validate_descriptor(self, descriptor: ERSDEStepDescriptor) -> None:
        if descriptor.run_id != self.run_id:
            self.clear()
            raise ERSDETrackingError(
                f"stale ER-SDE tracker belongs to run {self.run_id}, current run is "
                f"{descriptor.run_id}"
            )

    def _take_pending(
        self,
        denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
    ) -> torch.Tensor | None:
        pending = self._pending
        if pending is None:
            if descriptor.step_id == 0:
                return None
            if descriptor.requires_compensation:
                raise ERSDETrackingError(
                    f"forecast step {descriptor.step_id} has no preceding ER-SDE increment"
                )
            return None

        # Transfer ownership immediately. Every exit below leaves no stale tensor.
        self._pending = None
        if pending.target_step_id != descriptor.step_id:
            raise ERSDETrackingError(
                "stale ER-SDE increment targets step "
                f"{pending.target_step_id}, current step is {descriptor.step_id}"
            )
        increment = pending.value
        if (
            increment.shape != denoised.shape
            or increment.device != denoised.device
            or increment.dtype != denoised.dtype
        ):
            raise ERSDETrackingError(
                "ER-SDE increment does not exactly match the packed denoised tensor "
                f"(q={tuple(increment.shape)}/{increment.dtype}/{increment.device}, "
                f"denoised={tuple(denoised.shape)}/{denoised.dtype}/{denoised.device})"
            )
        return increment

    def _causal_dense_output_ready(self, descriptor: ERSDEStepDescriptor) -> bool:
        """Enable dense output only once exact actual-anchor cadence proves it safe."""
        if descriptor.mode != "forecast" or not self._denoised_anchors:
            return False
        latest = self._denoised_anchors[-1]
        if descriptor.step_id != latest.step_id + 1:
            return False
        if len(self._denoised_anchors) == 1:
            # The reviewed one-point bootstrap is exactly actual step 0 -> forecast step 1.
            return latest.step_id == 0 and descriptor.step_id == 1
        previous = self._denoised_anchors[-2]
        # After any causal forecast, ER-SDE forces an actual refresh. A gap of at
        # least two between exact anchors proves that this history has crossed a
        # forecast interval rather than being only warmup/all-actual history.
        return latest.step_id - previous.step_id >= 2

    def consume(
        self,
        denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
    ) -> torch.Tensor:
        """Return the sampler-space denoised value with ER-SDE-specific ownership.

        Causal forecasts use bounded dense output from exact actual denoised anchors
        whenever the reviewed cadence proves that history is available. Replay keeps
        the original exact-q compensation contract.
        """
        if not torch.is_tensor(denoised):
            raise ERSDETrackingError("ER-SDE model result is not a packed tensor")
        self._validate_descriptor(descriptor)

        dense_prediction = None
        if self._causal_dense_output_ready(descriptor):
            try:
                dense_prediction = self.predict_causal_denoised(denoised, descriptor)
            except ERSDEDenseOutputError as exc:
                # Fail closed to the already-reviewed direct-q correction. The dense
                # path is an ER-SDE forecast refinement, not a prerequisite for the
                # existing stochastic ownership contract.
                if self.debug:
                    LOG.warning(
                        "Spectrum H3 ER-SDE dense output unavailable step=%s reason=%s; "
                        "using exact-q forecast compensation",
                        descriptor.step_id,
                        exc,
                    )

        increment = self._take_pending(denoised, descriptor)
        if increment is None:
            reason = "first_step" if descriptor.step_id == 0 else "no_pending_increment"
            self._debug_log(denoised, descriptor, None, False, reason)
            if descriptor.mode == "actual":
                self.observe_actual_denoised(denoised, descriptor)
            return denoised

        if dense_prediction is not None:
            dense = dense_prediction.value
            if (
                dense.shape != denoised.shape
                or dense.device != denoised.device
                or dense.dtype != denoised.dtype
            ):
                raise ERSDEDenseOutputError(
                    "ER-SDE dense-output prediction does not match the current packed result"
                )
            self._debug_dense_log(
                denoised, dense, descriptor, increment, dense_prediction
            )
            return dense

        if not descriptor.requires_compensation:
            self._debug_log(denoised, descriptor, increment, False, "state_aware_step")
            if descriptor.mode == "actual":
                self.observe_actual_denoised(denoised, descriptor)
            return denoised

        corrected = denoised - increment
        self._debug_log(corrected, descriptor, increment, True, "forecast_state_mismatch")
        return corrected

    def observe_actual_denoised(
        self,
        denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
    ) -> None:
        """Retain at most two causal actual solver-space denoised anchors."""
        if descriptor.mode != "actual":
            return
        if not torch.is_tensor(denoised):
            raise ERSDEDenseOutputError("ER-SDE actual denoised anchor is not a tensor")
        if descriptor.run_id != self.run_id:
            raise ERSDEDenseOutputError(
                f"dense-output anchor belongs to stale run {descriptor.run_id}"
            )
        if self._denoised_anchors and descriptor.step_id <= self._denoised_anchors[-1].step_id:
            raise ERSDEDenseOutputError(
                "ER-SDE actual denoised anchors are not strictly increasing in step order"
            )
        if self._denoised_anchors:
            previous = self._denoised_anchors[-1].value
            if (
                previous.shape != denoised.shape
                or previous.device != denoised.device
                or previous.dtype != denoised.dtype
            ):
                raise ERSDEDenseOutputError(
                    "ER-SDE actual denoised anchor shape/device/dtype changed within the run"
                )
        retained = denoised.detach().clone(memory_format=torch.contiguous_format)
        self._denoised_anchors.append(
            _DenoisedAnchor(step_id=int(descriptor.step_id), value=retained)
        )
        if len(self._denoised_anchors) > 2:
            del self._denoised_anchors[:-2]
        if self.debug:
            LOG.warning(
                "Spectrum H3 ER-SDE dense anchor step=%s anchors=%s retained_bytes=%s",
                descriptor.step_id,
                self.denoised_anchor_steps,
                sum(a.value.numel() * a.value.element_size() for a in self._denoised_anchors),
            )

    def predict_causal_denoised(
        self,
        raw_denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
    ) -> ERSDEDenseOutputPrediction | None:
        """Predict ER-SDE's denoised variable directly from clean actual anchors."""
        if descriptor.mode != "forecast":
            return None
        if descriptor.run_id != self.run_id:
            raise ERSDEDenseOutputError(
                f"dense-output forecast belongs to stale run {descriptor.run_id}"
            )
        if not torch.is_tensor(raw_denoised):
            raise ERSDEDenseOutputError("ER-SDE forecast result is not a packed tensor")
        if not self._denoised_anchors:
            return None

        latest = self._denoised_anchors[-1]
        if latest.step_id >= descriptor.step_id:
            raise ERSDEDenseOutputError(
                "ER-SDE dense-output anchor is not strictly earlier than the forecast"
            )
        if descriptor.step_id != latest.step_id + 1:
            raise ERSDEDenseOutputError(
                "ER-SDE dense output requires the latest actual anchor immediately before "
                f"the forecast (anchor={latest.step_id}, forecast={descriptor.step_id})"
            )
        if (
            latest.value.shape != raw_denoised.shape
            or latest.value.device != raw_denoised.device
            or latest.value.dtype != raw_denoised.dtype
        ):
            raise ERSDEDenseOutputError(
                "ER-SDE dense-output anchor does not match the current packed denoised tensor"
            )

        latest_coordinate = self._solver_coordinates.get(latest.step_id)
        target_coordinate = self._solver_coordinates.get(descriptor.step_id)
        if latest_coordinate is None or target_coordinate is None:
            return None

        if len(self._denoised_anchors) == 1:
            predicted = latest.value.clone(memory_format=torch.contiguous_format)
            return ERSDEDenseOutputPrediction(
                value=predicted,
                mode="latest_actual_hold",
                anchor_steps=(latest.step_id,),
                alpha=0.0,
            )

        previous = self._denoised_anchors[-2]
        if (
            previous.value.shape != latest.value.shape
            or previous.value.device != latest.value.device
            or previous.value.dtype != latest.value.dtype
        ):
            raise ERSDEDenseOutputError(
                "ER-SDE dense-output actual anchor tensors are incompatible"
            )
        previous_coordinate = self._solver_coordinates.get(previous.step_id)
        if previous_coordinate is None:
            predicted = latest.value.clone(memory_format=torch.contiguous_format)
            return ERSDEDenseOutputPrediction(
                value=predicted,
                mode="latest_actual_hold_missing_coordinate",
                anchor_steps=(latest.step_id,),
                alpha=0.0,
            )
        denominator = latest_coordinate - previous_coordinate
        advance = target_coordinate - latest_coordinate
        alpha = advance / denominator
        scale = torch.maximum(
            torch.maximum(previous_coordinate.abs(), latest_coordinate.abs()),
            torch.ones_like(latest_coordinate),
        )
        # Build the trust guard entirely on-device. Invalid or overlong
        # extrapolation becomes weight zero, i.e. a latest-actual hold, without a
        # CUDA synchronization in the non-debug hot path.
        valid = (
            torch.isfinite(alpha)
            & torch.isfinite(denominator)
            & (denominator.abs() > 1e-12 * scale)
            & (denominator * advance >= 0.0)
            & (alpha >= -1e-6)
            & (alpha <= 1.0 + 1e-6)
        )
        bounded_alpha = alpha.clamp(0.0, 1.0)
        weight = torch.where(valid, bounded_alpha, torch.zeros_like(bounded_alpha))
        weight = weight.to(device=latest.value.device, dtype=latest.value.dtype)
        predicted = torch.lerp(latest.value, previous.value, -weight)
        mode = "lambda_bounded_extrapolation"
        if self.debug and not bool(valid.item()):
            mode = "latest_actual_hold_extrapolation_guard"
        return ERSDEDenseOutputPrediction(
            value=predicted,
            mode=mode,
            anchor_steps=(previous.step_id, latest.step_id),
            alpha=weight,
        )

    def clear(self) -> None:
        self._pending = None
        self._scaler_prefix.clear()
        self._scaler_calls = 0
        self._solver_coordinates.clear()
        self._denoised_anchors.clear()

    @staticmethod
    def _rms(value: torch.Tensor) -> float:
        if value.numel() == 0:
            return 0.0
        norm = torch.linalg.vector_norm(value.detach(), dtype=torch.float32)
        return float((norm / math.sqrt(value.numel())).item())

    def _debug_log(
        self,
        denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
        increment: torch.Tensor | None,
        applied: bool,
        reason: str,
    ) -> None:
        if not self.debug:
            return
        denoised_rms = self._rms(denoised)
        q_rms = 0.0 if increment is None else self._rms(increment)
        ratio = q_rms / denoised_rms if denoised_rms > 0.0 else float("inf")
        LOG.warning(
            "Spectrum H3 ER-SDE compensation step=%s step_type=%s "
            "replay_source_actual=%s q_pending=%s q_rms=%.8f denoised_rms=%.8f "
            "correction_rms=%.8f correction_ratio=%.8f applied=%s reason=%s "
            "max_stage=%s stochastic=true",
            descriptor.step_id,
            descriptor.mode,
            descriptor.replay_source_actual,
            increment is not None,
            q_rms,
            denoised_rms,
            q_rms if applied else 0.0,
            ratio if applied else 0.0,
            applied,
            reason,
            self.max_stage,
        )

    def _debug_dense_log(
        self,
        raw_denoised: torch.Tensor,
        dense: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
        increment: torch.Tensor,
        prediction: ERSDEDenseOutputPrediction,
    ) -> None:
        if not self.debug:
            return
        raw_rms = self._rms(raw_denoised)
        dense_rms = self._rms(dense)
        q_rms = self._rms(increment)
        alpha = (
            float(prediction.alpha.detach().item())
            if torch.is_tensor(prediction.alpha)
            else float(prediction.alpha)
        )
        LOG.warning(
            "Spectrum H3 ER-SDE dense output step=%s mode=%s anchor_steps=%s "
            "alpha=%.8f q_pending=true q_rms=%.8f raw_denoised_rms=%.8f "
            "dense_denoised_rms=%.8f q_applied=false reason=solver_space_dense_output "
            "max_stage=%s stochastic=true",
            descriptor.step_id,
            prediction.mode,
            prediction.anchor_steps,
            alpha,
            q_rms,
            raw_rms,
            dense_rms,
            self.max_stage,
        )


__all__ = [
    "ERSDEDenseOutputError",
    "ERSDEDenseOutputPrediction",
    "ERSDEStepDescriptor",
    "ERSDEStochasticTracker",
    "ERSDETrackingError",
    "function_ast_digest",
    "function_node_ast_digest",
    "native_default_er_sde_noise_scaler",
]
