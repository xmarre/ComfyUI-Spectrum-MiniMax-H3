from __future__ import annotations

import contextlib
import inspect
import logging
import math
import threading
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

LOG = logging.getLogger(__name__)

RES4LYF_STOCHASTIC_BRIDGE_KEY = "spectrum_h3_res4lyf_stochastic_bridge"

# RES4LYF draws outer-step noise from ``noise_sampler`` and internal-substep
# noise from ``noise_sampler2``. Each reviewed swap consumes exactly one draw
# from its own channel; every other draw is an untracked stochastic path.
_SWAP_CHANNELS = {"step": "step", "substep": "substep"}


class RES4LYFStochasticError(RuntimeError):
    """The reviewed RES4LYF stochastic-state contract could not be maintained."""


@dataclass(frozen=True, slots=True)
class RES4LYFStepDescriptor:
    run_id: int
    step_id: int
    mode: str


@dataclass(frozen=True, slots=True)
class _Transition:
    kind: str
    target_sigma: float
    noised: torch.Tensor
    increment_rms: float | None


@dataclass(frozen=True, slots=True)
class _Anchor:
    step_id: int
    coordinate: float
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class RES4LYFDenseOutputPrediction:
    value: torch.Tensor
    mode: str
    anchor_steps: tuple[int, ...]
    alpha: float


class _CountingNoiseGenerator:
    """Delegate a RES4LYF noise generator while attributing every draw."""

    __slots__ = ("_base", "_bridge", "_channel")

    def __init__(self, base: Any, bridge: RES4LYFStochasticBridge, channel: str) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_bridge", bridge)
        object.__setattr__(self, "_channel", channel)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._bridge.record_noise_draw(self._channel)
        return self._base(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._base, name, value)


def _scalar_sigma(value: Any) -> float:
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if flat.numel() == 0:
            raise RES4LYFStochasticError("RES4LYF model-call sigma is empty")
        if flat.numel() > 1 and not bool((flat == flat[0]).all().item()):
            raise RES4LYFStochasticError(
                "RES4LYF model-call sigma differs across the batch"
            )
        result = float(flat[0].item())
    else:
        result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RES4LYFStochasticError(
            f"RES4LYF model-call sigma is nonfinite or nonpositive: {result}"
        )
    return result


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    norm = torch.linalg.vector_norm(value.detach(), dtype=torch.float32)
    return float((norm / math.sqrt(value.numel())).item())


class RES4LYFStochasticBridge:
    """Own RES4LYF's stochastic transitions and causal solver-space dense output.

    RES4LYF's SDE wrappers replace the deterministic RK landing state with a
    freshly noised state before the next H3 call: once after every outer step
    and, for the fixed-stage methods, once after every internal stage. An exact
    H3 evaluation responds to that fresh noise; a skipped transformer cannot.
    Every exact actual anchor in Spectrum's feature history also saw a different
    noise realization, so a hidden-feature forecast cannot be made consistent
    with the current noise by compensating only the latest increment.

    Forecast results are therefore replaced, before RES4LYF consumes them, by a
    bounded causal dense output in denoised space built only from exact H3
    anchors. The bridge also proves at runtime that every stochastic draw went
    through a reviewed swap and that each noised state is consumed exactly once,
    bit-for-bit, by the immediately following model call.
    """

    def __init__(
        self,
        *,
        run_id: int,
        coordinate_fn: Callable[[float], float],
        debug: bool,
        hold_only: bool = False,
    ) -> None:
        self.run_id = int(run_id)
        self.debug = bool(debug)
        # H3 Continuum continuation chunks keep a zero-order hold for every
        # forecast, matching the reviewed SA-Solver Continuum policy.
        self.hold_only = bool(hold_only)
        self._coordinate_fn = coordinate_fn
        self._bound_noise_sampler: Any = None
        self._invalid_reason: str | None = None
        self._active_swap: str | None = None
        self._active_draws = 0
        self._pending: _Transition | None = None
        self._associated_step_id: int | None = None
        self._associated_kind: str | None = None
        self._associated_input: Any = None
        # Owned copy of the validated input contents. Identity alone cannot
        # detect an in-place edit, and inference tensors have no version counter.
        self._associated_snapshot: torch.Tensor | None = None
        self._associated_sigma: float | None = None
        # Noise swaps observed since the last associated model call. Every SDE
        # model call after the first must be preceded by at least one swap.
        self._swaps_since_call = 0
        self._integrity_check: Callable[[], str | None] | None = None
        self._closed = False
        self._fail_closed_applied = False
        self._anchors: list[_Anchor] = []
        self.transitions_recorded = 0
        self.transitions_consumed = 0

    @property
    def invalid_reason(self) -> str | None:
        return self._invalid_reason

    @property
    def bound(self) -> bool:
        return self._bound_noise_sampler is not None

    @property
    def anchor_steps(self) -> tuple[int, ...]:
        return tuple(anchor.step_id for anchor in self._anchors)

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def invalidate(self, reason: str) -> None:
        if self._invalid_reason is None:
            self._invalid_reason = str(reason)
        self._pending = None

    # ------------------------------------------------------------------
    # Native RES4LYF noise-sampler ownership
    # ------------------------------------------------------------------

    def bind_noise_sampler(self, noise_sampler: Any) -> bool:
        """Bind the single RES4LYF noise sampler created for this sampler call."""
        if self._closed:
            return False
        if self._bound_noise_sampler is not None:
            if self._bound_noise_sampler is not noise_sampler:
                self.invalidate(
                    "RES4LYF constructed a second noise sampler for the tracked run"
                )
            return False
        self._bound_noise_sampler = noise_sampler
        return True

    def owns(self, noise_sampler: Any) -> bool:
        return not self._closed and self._bound_noise_sampler is noise_sampler

    def set_integrity_check(self, check: Callable[[], str | None] | None) -> None:
        """Install a callable that reports a changed native contract, or None."""
        self._integrity_check = check

    def check_integrity(self) -> None:
        check = self._integrity_check
        if check is None or self._invalid_reason is not None:
            return
        reason = check()
        if reason is not None:
            self.invalidate(reason)

    def mark_fail_closed(self) -> bool:
        """Return True exactly once, when the run first fails closed."""
        if self._fail_closed_applied:
            return False
        self._fail_closed_applied = True
        return True

    def wrap_noise_generators(self, noise_sampler: Any) -> None:
        for attribute, channel in (("noise_sampler", "step"), ("noise_sampler2", "substep")):
            generator = getattr(noise_sampler, attribute, None)
            if generator is None:
                self.invalidate(f"RES4LYF {attribute} was not initialized")
                continue
            if isinstance(generator, _CountingNoiseGenerator):
                continue
            setattr(
                noise_sampler,
                attribute,
                _CountingNoiseGenerator(generator, self, channel),
            )

    def record_noise_draw(self, channel: str) -> None:
        active = self._active_swap
        if active is None:
            self.invalidate(
                f"untracked RES4LYF stochastic draw from the {channel} noise generator"
            )
            return
        if _SWAP_CHANNELS[active] != channel:
            self.invalidate(
                f"RES4LYF {active} noise swap drew from the {channel} noise generator"
            )
            return
        self._active_draws += 1

    def enter_swap(self, kind: str) -> None:
        if kind not in _SWAP_CHANNELS:
            raise ValueError(f"unknown RES4LYF noise swap kind {kind!r}")
        if self._active_swap is not None:
            self.invalidate("nested RES4LYF noise swaps are outside the reviewed contract")
        self._active_swap = kind
        self._active_draws = 0

    def abort_swap(self, reason: str) -> None:
        self._active_swap = None
        self._active_draws = 0
        self.invalidate(reason)

    def exit_swap(
        self,
        kind: str,
        *,
        landing: Any,
        result: Any,
        target_sigma: Any,
    ) -> None:
        active = self._active_swap
        draws = self._active_draws
        self._active_swap = None
        self._active_draws = 0
        self._swaps_since_call += 1
        if active != kind:
            self.invalidate("RES4LYF noise swap bookkeeping was reordered")
            return
        if self._pending is not None:
            # A noised state must be consumed by exactly the next model call.
            self.invalidate(
                f"RES4LYF {self._pending.kind} noise transition targeting sigma "
                f"{self._pending.target_sigma:.8g} was never consumed by a model call"
            )
            return
        applied = result is not landing
        if draws != (1 if applied else 0):
            self.invalidate(
                f"RES4LYF {kind} noise swap consumed {draws} draws "
                f"(applied={applied})"
            )
            return
        if not applied:
            return
        if not torch.is_tensor(result) or not torch.is_tensor(landing):
            self.invalidate("RES4LYF noise swap did not return a tensor state")
            return
        if result.shape != landing.shape or result.device != landing.device:
            self.invalidate("RES4LYF noise swap changed latent shape or device")
            return
        if not result.is_floating_point() or not landing.is_floating_point():
            self.invalidate("RES4LYF noise swap returned a non-floating state")
            return
        try:
            sigma = _scalar_sigma(target_sigma)
        except RES4LYFStochasticError as exc:
            self.invalidate(str(exc))
            return
        increment_rms = _rms(result - landing) if self.debug else None
        # RES4LYF's float64 noise promotes the swap result; the sampler writes it
        # into its work-dtype state buffer before the next model call. Retain an
        # owned copy of exactly that cast so a later in-place edit of RES4LYF's
        # tensor cannot pass association unnoticed.
        noised = result.detach().to(
            dtype=landing.dtype,
            memory_format=torch.contiguous_format,
            copy=True,
        )
        self._pending = _Transition(
            kind=kind,
            target_sigma=sigma,
            noised=noised,
            increment_rms=increment_rms,
        )
        self.transitions_recorded += 1

    # ------------------------------------------------------------------
    # Model-call association
    # ------------------------------------------------------------------

    def associate(
        self,
        x: Any,
        timestep: Any,
        *,
        run_id: int,
        step_id: int,
    ) -> str:
        """Bind the active model call to the stochastic state it receives."""
        if int(run_id) != self.run_id:
            self.invalidate(
                f"stale RES4LYF stochastic bridge belongs to run {self.run_id}, "
                f"current run is {run_id}"
            )
        self.check_integrity()
        if self._invalid_reason is not None:
            raise RES4LYFStochasticError(self._invalid_reason)
        if self._bound_noise_sampler is None:
            self.invalidate("RES4LYF noise sampler was not created through the tracked bridge")
            raise RES4LYFStochasticError(self._invalid_reason or "unbound bridge")
        if self._active_swap is not None:
            self.invalidate("H3 model call arrived inside a RES4LYF noise swap")
            raise RES4LYFStochasticError(self._invalid_reason or "active swap")
        try:
            sigma = _scalar_sigma(timestep)
        except RES4LYFStochasticError as exc:
            self.invalidate(str(exc))
            raise

        if self._associated_step_id == int(step_id):
            # Only a retry of the same validated call may reuse its association:
            # the identical input object at the same sigma, with no noise swap or
            # pending transition since it was validated.
            if (
                self._associated_kind is None
                or x is not self._associated_input
                or not self._input_unchanged(x)
                or sigma != self._associated_sigma
                or self._swaps_since_call
                or self._pending is not None
            ):
                self.invalidate(
                    f"H3 model call step {step_id} was associated again with a "
                    "different input, sigma or RES4LYF noise transition"
                )
                raise RES4LYFStochasticError(self._invalid_reason or "re-association")
            return self._associated_kind

        if self._associated_step_id is not None and not self._swaps_since_call:
            # RES4LYF runs at least one (possibly no-op) noise swap between
            # consecutive SDE model calls. Without one, this call's stochastic
            # state is unverified rather than deterministic.
            self.invalidate(
                "no RES4LYF noise swap was observed between consecutive H3 calls"
            )
            raise RES4LYFStochasticError(self._invalid_reason or "missing swap")

        if not torch.is_tensor(x) or not x.is_floating_point():
            self.invalidate("H3 model input is not a floating-point tensor")
            raise RES4LYFStochasticError(self._invalid_reason or "input type")

        pending = self._pending
        # Transfer ownership before validation so no exit leaves a stale state.
        self._pending = None
        if pending is None:
            # Either the run's first call, or every swap since the previous call
            # was a verified no-op.
            kind = "deterministic"
            snapshot = x.detach().clone(memory_format=torch.contiguous_format)
        else:
            noised = pending.noised
            if (
                not torch.is_tensor(x)
                or x.shape != noised.shape
                or x.device != noised.device
                or x.dtype != noised.dtype
            ):
                self.invalidate(
                    f"RES4LYF {pending.kind} noised state does not match the next "
                    "H3 model input layout"
                )
                raise RES4LYFStochasticError(self._invalid_reason or "layout mismatch")
            if not math.isclose(sigma, pending.target_sigma, rel_tol=1e-5, abs_tol=1e-8):
                self.invalidate(
                    f"RES4LYF {pending.kind} noise transition targets sigma "
                    f"{pending.target_sigma:.8g}, next H3 call is at sigma {sigma:.8g}"
                )
                raise RES4LYFStochasticError(self._invalid_reason or "sigma mismatch")
            if not torch.equal(x, noised):
                self.invalidate(
                    f"RES4LYF {pending.kind} noised state was modified before the next "
                    "H3 model call"
                )
                raise RES4LYFStochasticError(self._invalid_reason or "state mismatch")
            kind = f"stochastic_{pending.kind}"
            # The owned noised copy already equals x bit for bit.
            snapshot = noised
            self.transitions_consumed += 1
            if self.debug:
                LOG.warning(
                    "Spectrum H3 RES4LYF stochastic transition step=%s kind=%s "
                    "sigma=%.8f increment_rms=%.8f consumed_by=next_model_call",
                    step_id,
                    pending.kind,
                    sigma,
                    float(pending.increment_rms or 0.0),
                )
        self._associated_step_id = int(step_id)
        self._associated_kind = kind
        self._associated_input = x
        self._associated_snapshot = snapshot
        self._associated_sigma = sigma
        self._swaps_since_call = 0
        return kind

    def _input_unchanged(self, x: Any) -> bool:
        snapshot = self._associated_snapshot
        return (
            snapshot is not None
            and torch.is_tensor(x)
            and x.shape == snapshot.shape
            and x.dtype == snapshot.dtype
            and x.device == snapshot.device
            and bool(torch.equal(x, snapshot))
        )

    # ------------------------------------------------------------------
    # Solver-space result ownership
    # ------------------------------------------------------------------

    def consume(
        self,
        result: Any,
        timestep: Any,
        descriptor: RES4LYFStepDescriptor,
        *,
        model_input: Any,
    ) -> torch.Tensor:
        """Return the denoised value RES4LYF may consume for this model call."""
        if descriptor.run_id != self.run_id:
            raise RES4LYFStochasticError(
                f"RES4LYF dense output belongs to stale run {descriptor.run_id}"
            )
        if not torch.is_tensor(result):
            raise RES4LYFStochasticError("RES4LYF model result is not a tensor")
        self.check_integrity()
        if self._invalid_reason is not None:
            # Tracking failed after association (or during the model call): a
            # forecast for this call must not reach RES4LYF's solver state.
            if descriptor.mode == "forecast":
                raise RES4LYFStochasticError(self._invalid_reason)
            return result
        if self._associated_step_id != descriptor.step_id:
            raise RES4LYFStochasticError(
                "RES4LYF model result arrived without its stochastic-state association"
            )
        # Every attempt of this call, including an exact retry after a failed
        # forecast, must have run on the validated input contents.
        if model_input is not self._associated_input or not self._input_unchanged(model_input):
            self.invalidate(
                f"H3 model input for step {descriptor.step_id} changed between "
                "association and its result"
            )
            if descriptor.mode == "forecast":
                raise RES4LYFStochasticError(self._invalid_reason or "input changed")
            return result
        coordinate = self._coordinate(timestep)
        if descriptor.mode == "actual":
            self._observe_actual(result, coordinate, descriptor)
            return result
        if descriptor.mode != "forecast":
            raise RES4LYFStochasticError(
                f"RES4LYF stochastic forecasting does not support {descriptor.mode!r} steps"
            )
        prediction = self.predict(result, coordinate, descriptor)
        if self.debug:
            LOG.warning(
                "Spectrum H3 RES4LYF dense output step=%s input=%s mode=%s "
                "anchor_steps=%s alpha=%.8f raw_feature_forecast=ignored "
                "solver_history=native",
                descriptor.step_id,
                self._associated_kind,
                prediction.mode,
                prediction.anchor_steps,
                prediction.alpha,
            )
        return prediction.value

    def _coordinate(self, timestep: Any) -> float:
        sigma = _scalar_sigma(timestep)
        coordinate = float(self._coordinate_fn(sigma))
        if math.isnan(coordinate):
            raise RES4LYFStochasticError(
                f"RES4LYF dense-output coordinate is undefined at sigma {sigma:.8g}"
            )
        return coordinate

    def _observe_actual(
        self,
        value: torch.Tensor,
        coordinate: float,
        descriptor: RES4LYFStepDescriptor,
    ) -> None:
        if self._anchors:
            latest = self._anchors[-1]
            if descriptor.step_id <= latest.step_id:
                raise RES4LYFStochasticError(
                    "RES4LYF actual denoised anchors are not strictly increasing"
                )
            if (
                latest.value.shape != value.shape
                or latest.value.device != value.device
                or latest.value.dtype != value.dtype
            ):
                raise RES4LYFStochasticError(
                    "RES4LYF actual denoised anchor layout changed within the run"
                )
        self._anchors.append(
            _Anchor(
                step_id=int(descriptor.step_id),
                coordinate=coordinate,
                value=value.detach().clone(memory_format=torch.contiguous_format),
            )
        )
        if len(self._anchors) > 2:
            del self._anchors[:-2]

    def predict(
        self,
        raw: torch.Tensor,
        coordinate: float,
        descriptor: RES4LYFStepDescriptor,
    ) -> RES4LYFDenseOutputPrediction:
        """Bounded causal denoised-space prediction from exact H3 anchors.

        The coordinate is the model's half-log-SNR. H3's per-stream audio
        schedule is a flow time shift, which only offsets logit(sigma) by a
        constant, so one scalar extrapolation weight is exact for both streams.
        """
        if not self._anchors:
            raise RES4LYFStochasticError("RES4LYF dense output has no exact H3 anchor")
        latest = self._anchors[-1]
        if latest.step_id != descriptor.step_id - 1:
            raise RES4LYFStochasticError(
                "RES4LYF dense output requires the latest exact anchor immediately "
                f"before the forecast (anchor={latest.step_id}, "
                f"forecast={descriptor.step_id})"
            )
        if (
            latest.value.shape != raw.shape
            or latest.value.device != raw.device
            or latest.value.dtype != raw.dtype
        ):
            raise RES4LYFStochasticError(
                "RES4LYF dense-output anchor does not match the current model result"
            )

        def hold(mode: str) -> RES4LYFDenseOutputPrediction:
            return RES4LYFDenseOutputPrediction(
                value=latest.value.clone(memory_format=torch.contiguous_format),
                mode=mode,
                anchor_steps=(latest.step_id,),
                alpha=0.0,
            )

        if self.hold_only:
            return hold("latest_actual_hold_continuum")
        if len(self._anchors) == 1:
            return hold("latest_actual_hold")
        previous = self._anchors[-2]
        # Consecutive exact anchors are prefix/refresh history rather than
        # evidence across a skipped interval; keep the first forecast after them
        # a zero-order hold, matching the reviewed ER-SDE/SA dense-output rule.
        if latest.step_id - previous.step_id == 1:
            return hold("latest_actual_hold_consecutive_anchors")

        denominator = latest.coordinate - previous.coordinate
        advance = coordinate - latest.coordinate
        scale = max(abs(previous.coordinate), abs(latest.coordinate), 1.0)
        if (
            not math.isfinite(denominator)
            or not math.isfinite(advance)
            or abs(denominator) <= 1e-12 * scale
            or denominator * advance < 0.0
        ):
            return hold("latest_actual_hold_extrapolation_guard")
        alpha = advance / denominator
        if not math.isfinite(alpha) or alpha < -1e-6 or alpha > 1.0 + 1e-6:
            return hold("latest_actual_hold_extrapolation_guard")
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha == 0.0:
            return hold("latest_actual_hold_same_coordinate")
        predicted = torch.lerp(latest.value, previous.value, -alpha)
        return RES4LYFDenseOutputPrediction(
            value=predicted,
            mode="coordinate_bounded_extrapolation",
            anchor_steps=(previous.step_id, latest.step_id),
            alpha=alpha,
        )

    def clear(self) -> None:
        self._closed = True
        self._integrity_check = None
        self._associated_input = None
        self._associated_snapshot = None
        self._associated_sigma = None
        self._swaps_since_call = 0
        self._pending = None
        self._active_swap = None
        self._active_draws = 0
        self._anchors.clear()
        self._associated_step_id = None
        self._associated_kind = None
        self._bound_noise_sampler = None


def tracked_noise_sampler_class(
    base_class: type,
    bridge: RES4LYFStochasticBridge,
    owner_guider: Any,
) -> type:
    """Return a RES4LYF noise-sampler subclass bound to one Spectrum run.

    Every overridden method delegates to the reviewed native implementation
    unchanged, so RNG order, AV-specific noise scaling and the numerical state
    remain owned by RES4LYF. Instances created for another guider stay native.
    """

    class SpectrumTrackedRKNoiseSampler(base_class):  # type: ignore[misc, valid-type]
        def __init__(self, RK: Any, model: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(RK, model, *args, **kwargs)
            if getattr(model, "inner_model", None) is owner_guider:
                bridge.bind_noise_sampler(self)

        def init_noise_samplers(self, *args: Any, **kwargs: Any) -> Any:
            result = super().init_noise_samplers(*args, **kwargs)
            if bridge.owns(self):
                bridge.wrap_noise_generators(self)
            return result

        def swap_noise_step(self, x_0: Any, x_next: Any, *args: Any, **kwargs: Any) -> Any:
            if not bridge.owns(self):
                return super().swap_noise_step(x_0, x_next, *args, **kwargs)
            bridge.enter_swap("step")
            try:
                result = super().swap_noise_step(x_0, x_next, *args, **kwargs)
            except BaseException:
                bridge.abort_swap("RES4LYF outer noise swap raised")
                raise
            bridge.exit_swap(
                "step",
                landing=x_next,
                result=result,
                target_sigma=self.sigma_next,
            )
            return result

        def swap_noise_substep(self, x_0: Any, x_next: Any, *args: Any, **kwargs: Any) -> Any:
            if not bridge.owns(self):
                return super().swap_noise_substep(x_0, x_next, *args, **kwargs)
            bridge.enter_swap("substep")
            try:
                result = super().swap_noise_substep(x_0, x_next, *args, **kwargs)
            except BaseException:
                bridge.abort_swap("RES4LYF substep noise swap raised")
                raise
            bridge.exit_swap(
                "substep",
                landing=x_next,
                result=result,
                target_sigma=self.sub_sigma_next,
            )
            return result

    SpectrumTrackedRKNoiseSampler.__qualname__ = (
        f"SpectrumTracked[{getattr(base_class, '__qualname__', 'RK_NoiseSampler')}]"
    )
    return SpectrumTrackedRKNoiseSampler


_INSTALL_LOCK = threading.Lock()


_EMPTY_CELL = object()


def _referenced_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _referenced_names(constant)
    return names


def _implementation_state(cls: type) -> dict[str, tuple[Any, ...]]:
    """Capture everything that determines what each class callable executes.

    Function objects are mutable: their code, defaults, keyword defaults and
    closure cells can be replaced in place, and the module globals their code
    reads can be rebound. Record those objects themselves so a later identity
    comparison detects any change without trusting the function reference.
    """
    state: dict[str, tuple[Any, ...]] = {}
    for name, value in vars(cls).items():
        if isinstance(value, (staticmethod, classmethod)):
            function = value.__func__
        elif inspect.isfunction(value):
            function = value
        elif callable(value) or any(hasattr(type(value), hook) for hook in ("__get__", "__set__", "__delete__")):
            state[name] = (value,)
            continue
        else:
            continue
        if not inspect.isfunction(function):
            state[name] = (value, function)
            continue
        cells = []
        for cell in function.__closure__ or ():
            try:
                cells.append(cell.cell_contents)
            except ValueError:
                cells.append(_EMPTY_CELL)
        defaults = function.__defaults__ or ()
        kwdefaults = sorted((function.__kwdefaults__ or {}).items(), key=lambda item: item[0])
        module_globals = function.__globals__
        referenced = sorted(_referenced_names(function.__code__) & set(module_globals))
        state[name] = (
            value,
            function,
            function.__code__,
            module_globals,
            len(defaults),
            *defaults,
            len(kwdefaults),
            *(item for pair in kwdefaults for item in pair),
            len(cells),
            *cells,
            len(referenced),
            *(item for key in referenced for item in (key, module_globals[key])),
        )
    return state


def _same_implementation(
    current: dict[str, tuple[Any, ...]],
    snapshot: dict[str, tuple[Any, ...]],
) -> bool:
    if current.keys() != snapshot.keys():
        return False
    for name, recorded in snapshot.items():
        live = current[name]
        if len(live) != len(recorded):
            return False
        for live_item, recorded_item in zip(live, recorded, strict=True):
            if live_item is recorded_item:
                continue
            # Lengths and global names are plain values; everything else must
            # be the identical object.
            if type(live_item) in (int, str) and live_item == recorded_item:
                continue
            return False
    return True


@contextlib.contextmanager
def tracked_res4lyf_noise_sampler(
    rk_sampler_module: Any,
    original_class: type,
    bridge: RES4LYFStochasticBridge,
    owner_guider: Any,
) -> Iterator[type]:
    """Install the tracked noise sampler for exactly one RES4LYF sampler call.

    Installation and restoration are serialized. Spectrum restores the audited
    class only while its own tracked class is still installed, so a replacement
    made by another component during the run is left in place. Any change to
    the audited class's methods during the run makes later forecasts fail
    closed through the bridge's integrity check.
    """
    with _INSTALL_LOCK:
        if getattr(rk_sampler_module, "RK_NoiseSampler", None) is not original_class:
            raise RES4LYFStochasticError(
                "RES4LYF RK_NoiseSampler is already replaced; nested, overlapping or "
                "foreign patching is outside the reviewed contract"
            )
        tracked = tracked_noise_sampler_class(original_class, bridge, owner_guider)
        rk_sampler_module.RK_NoiseSampler = tracked
    snapshot = _implementation_state(original_class)

    def integrity() -> str | None:
        if not _same_implementation(_implementation_state(original_class), snapshot):
            return "RES4LYF RK_NoiseSampler methods changed during the tracked run"
        return None

    bridge.set_integrity_check(integrity)
    try:
        yield tracked
    finally:
        bridge.set_integrity_check(None)
        with _INSTALL_LOCK:
            current = getattr(rk_sampler_module, "RK_NoiseSampler", None)
            owned = current is tracked
            if owned:
                rk_sampler_module.RK_NoiseSampler = original_class
        if not owned:
            LOG.warning(
                "Spectrum H3 left RES4LYF RK_NoiseSampler as replaced by another "
                "component during a tracked run; the tracked class is inert once "
                "the run ends"
            )


__all__ = [
    "RES4LYF_STOCHASTIC_BRIDGE_KEY",
    "RES4LYFDenseOutputPrediction",
    "RES4LYFStepDescriptor",
    "RES4LYFStochasticBridge",
    "RES4LYFStochasticError",
    "tracked_noise_sampler_class",
    "tracked_res4lyf_noise_sampler",
]
