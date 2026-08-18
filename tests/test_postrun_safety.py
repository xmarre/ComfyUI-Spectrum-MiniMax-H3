from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

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


def _wait_for_research_slot(timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if postrun._RESEARCH_SLOT.acquire(blocking=False):
            postrun._RESEARCH_SLOT.release()
            return
        time.sleep(0.01)
    raise AssertionError("isolated research slot was not released")


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


def test_dispatch_research_never_waits_for_isolated_worker_completion(
    monkeypatch,
    tmp_path,
):
    worker = tmp_path / "slow_worker.py"
    worker.write_text("import time\ntime.sleep(0.5)\n", encoding="utf-8")
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(postrun, "_RESEARCH_WORKER", worker)
    monkeypatch.setattr(postrun, "default_store_root", lambda: tmp_path)
    monkeypatch.setattr(postrun, "_RESEARCH_TIMEOUT_SECONDS", 2.0)

    began = time.perf_counter()
    assert postrun._dispatch_research({"compatible": True})
    elapsed = time.perf_counter() - began

    assert elapsed < 0.25
    assert not postrun._dispatch_research({"compatible": True})
    _wait_for_research_slot()


@pytest.mark.skipif(os.name != "posix", reason="POSIX rlimit semantics required")
def test_research_command_disables_core_dumps_before_worker(tmp_path):
    worker = tmp_path / "core_limit_worker.py"
    worker.write_text(
        "import resource\nprint(resource.getrlimit(resource.RLIMIT_CORE)[0])\n",
        encoding="utf-8",
    )

    completed = postrun.subprocess.run(
        postrun._research_command(worker),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "0"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics required")
def test_isolated_research_sigsegv_cannot_terminate_parent(
    monkeypatch,
    tmp_path,
    caplog,
):
    worker = tmp_path / "segv_worker.py"
    worker.write_text(
        "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(postrun, "_RESEARCH_WORKER", worker)
    monkeypatch.setattr(postrun, "default_store_root", lambda: tmp_path)
    monkeypatch.setattr(postrun, "_RESEARCH_TIMEOUT_SECONDS", 2.0)

    with caplog.at_level("WARNING"):
        assert postrun._dispatch_research({"compatible": True})
        _wait_for_research_slot()

    assert "isolated research terminated by SIGSEGV" in caplog.text
    assert "timed out" not in caplog.text
    assert "completed generation remains valid" in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics required")
def test_research_timeout_kills_descendants_and_releases_slot(
    monkeypatch,
    tmp_path,
    caplog,
):
    worker = tmp_path / "descendant_worker.py"
    worker.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])\n"
        "time.sleep(2)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(postrun, "_RESEARCH_WORKER", worker)
    monkeypatch.setattr(postrun, "default_store_root", lambda: tmp_path)
    monkeypatch.setattr(postrun, "_RESEARCH_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(postrun, "_RESEARCH_TERMINATION_GRACE_SECONDS", 0.25)

    with caplog.at_level("WARNING"):
        assert postrun._dispatch_research({"compatible": True})
        _wait_for_research_slot(timeout=1.0)

    assert "isolated research timed out after 0.1 s and was terminated" in caplog.text
    assert "completed generation remains valid" in caplog.text


def test_isolated_research_python_failure_releases_worker_slot(
    monkeypatch,
    tmp_path,
    caplog,
):
    worker = tmp_path / "failed_worker.py"
    worker.write_text(
        "import sys\nsys.stderr.write('synthetic report failure\\n')\nraise SystemExit(3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(postrun, "_RESEARCH_SLOT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(postrun, "_RESEARCH_WORKER", worker)
    monkeypatch.setattr(postrun, "default_store_root", lambda: tmp_path)
    monkeypatch.setattr(postrun, "_RESEARCH_TIMEOUT_SECONDS", 2.0)

    with caplog.at_level("WARNING"):
        assert postrun._dispatch_research({"compatible": True})
        _wait_for_research_slot()

    assert "isolated research exited with status 3" in caplog.text
    assert "synthetic report failure" in caplog.text


def test_worker_script_loads_research_without_package_entrypoint(tmp_path):
    completed = postrun.subprocess.run(
        postrun._research_command(),
        input='{"block":{},"root":"' + str(tmp_path).replace("\\", "\\\\") + '"}',
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 1
    assert "CalibrationError" in completed.stderr
    assert "unsupported generic calibration schema" in completed.stderr


def test_installed_runtime_end_run_preserves_postrun_chain_identity():
    assert SpectrumH3Runtime.end_run is postrun.EFFECTIVE_END_RUN_HOOK
    assert external_compat._ORIGINAL_END_RUN is postrun._safe_end_run
