from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import generic_correction as _generic
from .generic_correction_calibration import GenericCalibrationState
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)

# Generic-correction research is diagnostic-only. A single daemon worker is enough:
# if one report job ever wedges, later jobs are skipped rather than accumulating
# threads or blocking completed generations.
_RESEARCH_SLOT = threading.BoundedSemaphore(1)
_CORE_RUNTIME_END = _generic._ORIGINAL_RUNTIME_END


def _teardown_log(runtime: Any, event: str, **fields: Any) -> None:
    config = getattr(runtime, "config", None)
    if not bool(getattr(config, "debug", False)):
        return
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    LOG.warning(
        "Spectrum H3 teardown transition ts=%.6f event=%s%s",
        time.time(),
        event,
        f" {rendered}" if rendered else "",
    )


def _research_worker(block: dict[str, Any]) -> None:
    try:
        result = _generic.persist_and_analyze(block)
        duplicate_note = " (duplicate ignored)" if result.duplicate else ""
        LOG.warning("\n%s%s", result.console_summary, duplicate_note)
        LOG.warning(
            "Spectrum H3 generic-correction post-run analysis completed in %.3f s",
            result.elapsed_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - research must never affect generation
        LOG.warning(
            "Spectrum H3 generic-correction background research failed; "
            "the completed generation remains valid: %s",
            exc,
        )
    finally:
        _RESEARCH_SLOT.release()


def _dispatch_research(block: dict[str, Any]) -> bool:
    """Start one bounded daemon research job without joining the sampling path."""
    if not _RESEARCH_SLOT.acquire(blocking=False):
        LOG.warning(
            "Spectrum H3 generic-correction post-run research skipped because a "
            "previous background analysis is still active"
        )
        return False
    try:
        worker = threading.Thread(
            target=_research_worker,
            args=(block,),
            name="SpectrumH3GenericResearch",
            daemon=True,
        )
        worker.start()
    except Exception as exc:  # noqa: BLE001 - thread creation must not affect generation
        _RESEARCH_SLOT.release()
        LOG.warning(
            "Spectrum H3 generic-correction post-run research could not be dispatched; "
            "the completed generation remains valid: %s",
            exc,
        )
        return False
    return True


def _safe_end_run(self: SpectrumH3Runtime, run_id: int) -> None:
    """End a run before any optional research and expose teardown crash boundaries.

    The previous generic-correction hook performed persistence/evaluation inline in
    ``SpectrumH3Runtime.end_run``. That made diagnostic research part of the sampler
    critical path: a filesystem/evaluator hang after the native sampler had already
    returned could prevent downstream VAE/video nodes from ever receiving the valid
    result. Keep calibration export synchronous and bounded, release runtime/VRAM
    state synchronously, then transfer the tensor-free calibration block to at most
    one daemon research worker.
    """
    active = getattr(self, "_run", None)
    if active is None or active.run_id != int(run_id):
        _CORE_RUNTIME_END(self, run_id)
        return

    calibration = getattr(self, "_generic_correction_calibration", None)
    block: dict[str, Any] | None = None
    if isinstance(calibration, GenericCalibrationState):
        _teardown_log(
            self,
            "calibration_export_begin",
            run_id=run_id,
            rows=len(calibration.rows),
            failures=calibration.failures,
        )
        try:
            block = _generic.emit_calibration_block(self, calibration)
        except Exception as exc:  # noqa: BLE001 - diagnostic export is non-critical
            calibration.failures += 1
            LOG.warning(
                "Spectrum H3 generic-correction calibration export failed; "
                "the completed generation remains valid: %s",
                exc,
            )
        _teardown_log(
            self,
            "calibration_export_end",
            run_id=run_id,
            compatible=bool(block is not None and block.get("compatible")),
        )

    forecaster = self.forecaster
    history_bytes = int(forecaster.history_tensor_bytes)
    persistent_bytes = int(forecaster.persistent_tensor_bytes)
    history_device = forecaster.history_device
    _teardown_log(
        self,
        "runtime_release_begin",
        run_id=run_id,
        history_entries=forecaster.history_length,
        history_bytes=history_bytes,
        persistent_bytes=persistent_bytes,
        history_device=history_device,
    )
    try:
        _CORE_RUNTIME_END(self, run_id)
    finally:
        # Generic-correction controller/calibration state must never retain a run
        # after the core runtime has been asked to tear it down.
        self._generic_correction_controller = None
        self._generic_correction_calibration = None
        self.model_aware._generic_correction_controller = None
        self.forecaster._generic_correction_capture_mode = None
    _teardown_log(
        self,
        "runtime_release_end",
        run_id=run_id,
        released_history_bytes=history_bytes,
    )

    if block is not None and block.get("compatible"):
        dispatched = _dispatch_research(block)
        _teardown_log(
            self,
            "research_dispatch",
            run_id=run_id,
            dispatched=dispatched,
        )


# Keep the post-run implementation's identity stable. Later compatibility layers
# may wrap SpectrumH3Runtime.end_run and publish that outermost callable here.
EFFECTIVE_END_RUN_HOOK = _safe_end_run


def install_postrun_safety() -> None:
    """Replace the synchronous generic-correction end hook once per interpreter."""
    global EFFECTIVE_END_RUN_HOOK
    if getattr(SpectrumH3Runtime, "_postrun_safety_installed", False):
        return
    current = SpectrumH3Runtime.end_run
    if current is not _generic._end_run:
        raise RuntimeError(
            "post-run safety must be installed immediately after generic correction"
        )
    SpectrumH3Runtime.end_run = _safe_end_run
    SpectrumH3Runtime._postrun_safety_installed = True
    EFFECTIVE_END_RUN_HOOK = SpectrumH3Runtime.end_run


__all__ = ["EFFECTIVE_END_RUN_HOOK", "install_postrun_safety"]