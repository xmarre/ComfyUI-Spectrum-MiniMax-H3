from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import generic_correction as _generic
from .generic_correction_calibration import GenericCalibrationState
from .generic_correction_research import default_store_root
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)

# Generic-correction research is diagnostic-only. Keep at most one analysis in
# flight, and run the evaluator in a separate interpreter so a native crash in
# diagnostic code cannot terminate the live ComfyUI process. The watcher thread
# only owns subprocess I/O/lifetime management; it never runs the evaluator.
_RESEARCH_SLOT = threading.BoundedSemaphore(1)
_RESEARCH_TIMEOUT_SECONDS = 30.0
_RESEARCH_WORKER = Path(__file__).with_name("generic_correction_worker.py")
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


def _stderr_tail(text: str, *, limit: int = 2000) -> str:
    rendered = (text or "").strip()
    if not rendered:
        return ""
    if len(rendered) > limit:
        rendered = "..." + rendered[-limit:]
    return rendered.replace("\n", " | ")


def _signal_name(returncode: int) -> str:
    number = -int(returncode)
    try:
        return signal.Signals(number).name
    except (ValueError, TypeError):
        return f"signal {number}"


def _research_process_watcher(
    process: subprocess.Popen[str],
    payload: str,
) -> None:
    try:
        try:
            stdout, stderr = process.communicate(
                payload,
                timeout=_RESEARCH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            LOG.warning(
                "Spectrum H3 generic-correction isolated research timed out after "
                "%.1f s and was terminated; the completed generation remains valid%s",
                _RESEARCH_TIMEOUT_SECONDS,
                f": {_stderr_tail(stderr)}" if _stderr_tail(stderr) else "",
            )
            return

        returncode = int(process.returncode or 0)
        if returncode != 0:
            detail = _stderr_tail(stderr)
            if returncode < 0:
                failure = f"terminated by {_signal_name(returncode)}"
            else:
                failure = f"exited with status {returncode}"
            LOG.warning(
                "Spectrum H3 generic-correction isolated research %s; the completed "
                "generation remains valid%s",
                failure,
                f": {detail}" if detail else "",
            )
            return

        try:
            result = json.loads(stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            LOG.warning(
                "Spectrum H3 generic-correction isolated research returned malformed "
                "output; the completed generation remains valid: %s%s",
                exc,
                f"; stderr={_stderr_tail(stderr)}" if _stderr_tail(stderr) else "",
            )
            return
        if not isinstance(result, dict) or result.get("ok") is not True:
            LOG.warning(
                "Spectrum H3 generic-correction isolated research returned an invalid "
                "result; the completed generation remains valid"
            )
            return

        duplicate_note = " (duplicate ignored)" if result.get("duplicate") else ""
        summary = str(result.get("console_summary") or "").strip()
        if summary:
            LOG.warning("\n%s%s", summary, duplicate_note)
        elapsed = float(result.get("elapsed_seconds", 0.0))
        LOG.warning(
            "Spectrum H3 generic-correction post-run analysis completed in %.3f s",
            elapsed,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never affect generation
        LOG.warning(
            "Spectrum H3 generic-correction research watcher failed; the completed "
            "generation remains valid: %s",
            exc,
        )
    finally:
        _RESEARCH_SLOT.release()


def _dispatch_research(block: dict[str, Any]) -> bool:
    """Dispatch one bounded, process-isolated research job without blocking sampling."""
    if not _RESEARCH_SLOT.acquire(blocking=False):
        LOG.warning(
            "Spectrum H3 generic-correction post-run research skipped because a "
            "previous isolated analysis is still active"
        )
        return False

    process: subprocess.Popen[str] | None = None
    try:
        payload = json.dumps(
            {
                "block": block,
                "root": str(default_store_root()),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-X",
                "faulthandler",
                str(_RESEARCH_WORKER),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        watcher = threading.Thread(
            target=_research_process_watcher,
            args=(process, payload),
            name="SpectrumH3GenericResearchWatcher",
            daemon=True,
        )
        watcher.start()
    except Exception as exc:  # noqa: BLE001 - dispatch must not affect generation
        if process is not None:
            try:
                process.kill()
                process.communicate(timeout=1.0)
            except Exception as cleanup_exc:  # noqa: BLE001 - best-effort cleanup
                LOG.debug(
                    "Spectrum H3 isolated research cleanup after dispatch failure failed: %s",
                    cleanup_exc,
                )
        _RESEARCH_SLOT.release()
        LOG.warning(
            "Spectrum H3 generic-correction isolated research could not be dispatched; "
            "the completed generation remains valid: %s",
            exc,
        )
        return False
    return True


def _safe_end_run(self: SpectrumH3Runtime, run_id: int) -> None:
    """End a run before optional research and expose teardown crash boundaries.

    Calibration export stays synchronous and bounded. Runtime/VRAM ownership is
    released synchronously. The tensor-free calibration block is then handed to
    at most one process-isolated research job. A Python exception, hang, or native
    crash in that diagnostic evaluator therefore cannot terminate the ComfyUI
    process or invalidate the completed generation.
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
            isolation="subprocess",
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
