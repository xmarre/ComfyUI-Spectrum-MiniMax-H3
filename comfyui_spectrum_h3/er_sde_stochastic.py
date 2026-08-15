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


def function_node_ast_digest(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Return the normalized digest shared by runtime and source-contract tests."""
    normalized = copy.deepcopy(function_node)
    normalized.decorator_list = []
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


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
    """Own exactly one ER-SDE increment until the following model result consumes it."""

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

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_step_id(self) -> int | None:
        return None if self._pending is None else self._pending.target_step_id

    @property
    def noise_calls(self) -> int:
        return self._noise_calls

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
        self._pending = _PendingIncrement(self._noise_calls, increment)
        self._scaler_prefix.clear()
        self._scaler_calls = 0
        return noise

    def consume(
        self,
        denoised: torch.Tensor,
        descriptor: ERSDEStepDescriptor,
    ) -> torch.Tensor:
        if not torch.is_tensor(denoised):
            raise ERSDETrackingError("ER-SDE model result is not a packed tensor")
        if descriptor.run_id != self.run_id:
            self.clear()
            raise ERSDETrackingError(
                f"stale ER-SDE tracker belongs to run {self.run_id}, current run is "
                f"{descriptor.run_id}"
            )
        pending = self._pending
        if pending is None:
            if descriptor.step_id == 0:
                self._debug_log(denoised, descriptor, None, False, "first_step")
                return denoised
            if descriptor.requires_compensation:
                raise ERSDETrackingError(
                    f"forecast step {descriptor.step_id} has no preceding ER-SDE increment"
                )
            self._debug_log(denoised, descriptor, None, False, "no_pending_increment")
            return denoised

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
        if not descriptor.requires_compensation:
            self._debug_log(denoised, descriptor, increment, False, "state_aware_step")
            return denoised

        corrected = denoised - increment
        self._debug_log(corrected, descriptor, increment, True, "forecast_state_mismatch")
        return corrected

    def clear(self) -> None:
        self._pending = None
        self._scaler_prefix.clear()
        self._scaler_calls = 0

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


__all__ = [
    "ERSDEStepDescriptor",
    "ERSDEStochasticTracker",
    "ERSDETrackingError",
    "function_ast_digest",
    "function_node_ast_digest",
    "native_default_er_sde_noise_scaler",
]
