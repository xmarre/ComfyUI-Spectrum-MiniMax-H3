from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from . import sampling
from .er_sde_stochastic import ERSDEStochasticTracker
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)

_KJ_PREVIEW_MODULE_SUFFIX = ".preview_override_node"
_KJ_PREVIEW_FILENAME_SUFFIX = "/nodes/preview_override_node.py"
_KJ_PREVIEW_QUALNAME = "_PreviewOverrideWrapper.__call__.<locals>.new_callback"
_NOT_KJ = object()


def _strict_kj_original_callback(callback: Any) -> object:
    """Return KJ Preview Override's wrapped callback only for its reviewed shape.

    KJ's preview callback performs synchronous GPU preview work before forwarding to
    the callback Spectrum supplied for the current offline pass. During ER-SDE's
    transformer-free replay that work is unnecessary for the accepted sampler result
    and can execute at a much higher cadence than during the capture pass.

    This intentionally recognizes only KJNodes' concrete nested callback provenance;
    arbitrary callback wrappers are never unwrapped.
    """
    if not inspect.isfunction(callback):
        return _NOT_KJ
    module = str(getattr(callback, "__module__", ""))
    qualname = str(getattr(callback, "__qualname__", ""))
    code = getattr(callback, "__code__", None)
    closure = getattr(callback, "__closure__", None)
    if code is None or closure is None:
        return _NOT_KJ
    filename = str(code.co_filename).replace("\\", "/")
    if not module.endswith(_KJ_PREVIEW_MODULE_SUFFIX):
        return _NOT_KJ
    if qualname != _KJ_PREVIEW_QUALNAME:
        return _NOT_KJ
    if code.co_name != "new_callback" or not filename.endswith(
        _KJ_PREVIEW_FILENAME_SUFFIX
    ):
        return _NOT_KJ
    freevars = tuple(code.co_freevars)
    if "original_callback" not in freevars or len(freevars) != len(closure):
        return _NOT_KJ
    try:
        original = closure[freevars.index("original_callback")].cell_contents
    except (IndexError, ValueError):
        return _NOT_KJ
    if original is not None and not callable(original):
        return _NOT_KJ
    return original


def _trace_er_sde_callback(
    callback: Callable[..., Any] | None,
    runtime: SpectrumH3Runtime,
) -> Callable[..., Any] | None:
    """Bracket the native callback boundary without touching tensors or syncing CUDA."""
    if callback is None or not runtime.config.debug:
        return callback

    def traced(step, x0, x, total_steps):
        runtime.log_offline_transition(
            "er_sde_callback_begin",
            run_id=runtime.active_run_id,
            step=step,
            phase=runtime.offline_phase,
        )
        try:
            return callback(step, x0, x, total_steps)
        finally:
            runtime.log_offline_transition(
                "er_sde_callback_end",
                run_id=runtime.active_run_id,
                step=step,
                phase=runtime.offline_phase,
            )

    return traced


def _install_replay_finalize_breadcrumb() -> None:
    current = SpectrumH3Runtime.finalize_step
    if getattr(current, "_spectrum_er_sde_replay_finalize_breadcrumb", False):
        return

    original = current

    def finalize_step(self: SpectrumH3Runtime, run_id: int, step_id: int) -> None:
        replay = self.offline_phase == "replay" and self.active_step_id == int(step_id)
        started = time.perf_counter()
        original(self, run_id, step_id)
        if replay:
            # runtime.finalize_step historically returned early for replay steps,
            # so its common finalize_step_end breadcrumb was never reached.
            self.log_offline_transition(
                "finalize_step_end",
                run_id=run_id,
                step=step_id,
                mode="replay",
                elapsed_s=f"{time.perf_counter() - started:.6f}",
            )

    finalize_step._spectrum_er_sde_replay_finalize_breadcrumb = True  # type: ignore[attr-defined]
    finalize_step._spectrum_er_sde_replay_finalize_original = original  # type: ignore[attr-defined]
    SpectrumH3Runtime.finalize_step = finalize_step


def _install_noise_sampler_breadcrumbs() -> None:
    current = ERSDEStochasticTracker.noise_sampler
    if getattr(current, "_spectrum_er_sde_noise_breadcrumbs", False):
        return

    original = current

    def noise_sampler(self: ERSDEStochasticTracker, sigma, sigma_next):
        if self.debug:
            LOG.warning(
                "Spectrum H3 ER-SDE replay boundary event=noise_sampler_begin "
                "run_id=%s source_step=%s target_step=%s",
                self.run_id,
                self.noise_calls,
                self.noise_calls + 1,
            )
        result = original(self, sigma, sigma_next)
        if self.debug:
            LOG.warning(
                "Spectrum H3 ER-SDE replay boundary event=noise_sampler_end "
                "run_id=%s source_step=%s target_step=%s pending_step=%s",
                self.run_id,
                self.noise_calls - 1,
                self.noise_calls,
                self.pending_step_id,
            )
        return result

    noise_sampler._spectrum_er_sde_noise_breadcrumbs = True  # type: ignore[attr-defined]
    noise_sampler._spectrum_er_sde_noise_original = original  # type: ignore[attr-defined]
    ERSDEStochasticTracker.noise_sampler = noise_sampler


def _install_tracked_er_sde_callback_guard() -> None:
    current = sampling._run_tracked_er_sde
    if getattr(current, "_spectrum_er_sde_offline_callback_guard", False):
        return

    original = current

    def run_tracked_er_sde(
        executor,
        runtime: SpectrumH3Runtime,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image,
        denoise_mask,
        disable_pbar,
    ):
        guarded_callback = callback
        if runtime.offline_phase == "replay":
            underlying = _strict_kj_original_callback(callback)
            if underlying is not _NOT_KJ:
                # Keep Spectrum's replay progress/external callback semantics, but
                # skip KJ's observational GPU preview decode during the fast replay.
                guarded_callback = underlying  # type: ignore[assignment]
                if runtime.config.debug:
                    runtime.log_offline_transition(
                        "er_sde_replay_preview_bypass",
                        run_id=runtime.active_run_id,
                        integration="KJNodes Model Preview Override",
                        reason="avoid synchronous GPU preview work during fast ER-SDE replay",
                    )
        guarded_callback = _trace_er_sde_callback(guarded_callback, runtime)
        return original(
            executor,
            runtime,
            model_wrap,
            sigmas,
            extra_args,
            guarded_callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    run_tracked_er_sde._spectrum_er_sde_offline_callback_guard = True  # type: ignore[attr-defined]
    run_tracked_er_sde._spectrum_er_sde_offline_callback_original = original  # type: ignore[attr-defined]
    sampling._run_tracked_er_sde = run_tracked_er_sde


def install_er_sde_offline_replay_safety() -> None:
    """Install narrow ER-SDE offline-replay diagnostics and preview protection."""
    _install_replay_finalize_breadcrumb()
    _install_noise_sampler_breadcrumbs()
    _install_tracked_er_sde_callback_guard()
