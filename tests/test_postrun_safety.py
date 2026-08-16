from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from comfyui_spectrum_h3 import external_patch_compat as external_compat
from comfyui_spectrum_h3 import postrun_safety as postrun
from comfyui_spectrum_h3.generic_correction_calibration import GenericCalibrationState
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _fake_runtime(run_id: int = 7):
    calibration = GenericCalibrationState(
        enabled=True,
        run_id=run_id,
        sampler_name="sample_er_sde",
        total_steps=2,
        schedule=(1.0, 0.0),
        config_snapshot={},
        rows=[{"scalar": 1.0}],
    )
    forecaster = SimpleNamespace(
        history_tensor_bytes=3_400_000_000,
        persistent_tensor_bytes=3_400_262_144,
        history_device="cuda:0",
        history_length=8,
        _generic_correction_capture_mode="full",
    )
    return SimpleNamespace(
        _run=SimpleNamespace(run_id=run_id),
        config=SimpleNamespace(debug=False),
        forecaster=forecaster,
        model_aware=SimpleNamespace(_generic_correction_controller=object()),
        _generic_correction_calibration=calibration,
        _generic_correction_controller=object(),
    )


def test_safe_end_releases_runtime_before_dispatching_research(monkeypatch):
    runtime = _fake_runtime()
    events = []

    def emit(_runtime, _state):
        events.append("emit")
        return {"compatible": True}

    def core_end(instance, run_id):
        events.append("core_end")
        assert run_id == 7
        instance._run = None

    def dispatch(block):
        events.append("dispatch")
        assert block == {"compatible": True}
        assert runtime._run is None
        return True

    monkeypatch.setattr(postrun._generic, "emit_calibration_block", emit)
    monkeypatch.setattr(postrun, "_CORE_RUNTIME_END", core_end)
    monkeypatch.setattr(postrun, "_dispatch_research", dispatch)

    postrun._safe_end_run(runtime, 7)

    assert events == ["emit", "core_end", "dispatch"]
    assert runtime._generic_correction_controller is None
    assert runtime._generic_correction_calibration is None
    assert runtime.model_aware._generic_correction_controller is None
    assert runtime.forecaster._generic_correction_capture_mode is None


def test_dispatch_research_never_waits_for_worker_completion(monkeypatch):
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def persist(_block):
        started.set()
        assert release.wait(timeout=2.0)
        finished.set()
        return SimpleNamespace(
            duplicate=False,
            console_summary="synthetic research",
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(postrun._generic, "persist_and_analyze", persist)

    began = time.perf_counter()
    assert postrun._dispatch_research({"compatible": True})
    elapsed = time.perf_counter() - began

    assert elapsed < 0.25
    assert started.wait(timeout=1.0)
    assert not finished.is_set()
    assert not postrun._dispatch_research({"compatible": True})

    release.set()
    assert finished.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    acquired = False
    while time.monotonic() < deadline:
        if postrun._RESEARCH_SLOT.acquire(blocking=False):
            acquired = True
            break
        time.sleep(0.01)
    assert acquired
    postrun._RESEARCH_SLOT.release()


def test_background_research_failure_releases_single_worker_slot(monkeypatch):
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    attempted = threading.Event()

    def persist(_block):
        attempted.set()
        raise OSError("synthetic report failure")

    monkeypatch.setattr(postrun._generic, "persist_and_analyze", persist)

    assert postrun._dispatch_research({"compatible": True})
    assert attempted.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    acquired = False
    while time.monotonic() < deadline:
        if postrun._RESEARCH_SLOT.acquire(blocking=False):
            acquired = True
            break
        time.sleep(0.01)
    assert acquired
    postrun._RESEARCH_SLOT.release()


def test_installed_runtime_end_run_preserves_postrun_chain_identity():
    assert SpectrumH3Runtime.end_run is postrun.EFFECTIVE_END_RUN_HOOK
    assert external_compat._ORIGINAL_END_RUN is postrun._safe_end_run
