from __future__ import annotations

import sys
import types

import pytest
import torch

from comfyui_spectrum_h3 import SpectrumH3Runtime
from comfyui_spectrum_h3 import comfy_compiler_compat as compat
from comfyui_spectrum_h3.config import SpectrumH3Config


@pytest.fixture(autouse=True)
def _reset_scope():
    compat._clear_compiler_bypass()
    yield
    compat._clear_compiler_bypass()


def _install_fake_model_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.ModuleType, list[object]]:
    calls: list[object] = []
    module = types.ModuleType("comfy.model_prefetch")

    def malloc_graph_enabled(device):
        calls.append(device)
        return True

    module.malloc_graph_enabled = malloc_graph_enabled
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", module)
    assert compat._ensure_model_prefetch_hook()
    return module, calls


def _runtime() -> SpectrumH3Runtime:
    return SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=1,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
            offline_smoothing_replay=False,
        )
    )


def test_runtime_step_scope_disables_compiler_until_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)
    runtime = _runtime()
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )

    decision = runtime.begin_step(torch.tensor([1.0]))
    assert compat._compiler_bypass_active()
    assert module.malloc_graph_enabled("cuda:0") is False
    assert calls == []

    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        labels=((0, "positive"),),
        expected_shape=(1, 2, 3),
    )
    assert actual
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        torch.zeros((1, 2, 3)),
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert not compat._compiler_bypass_active()
    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]
    runtime.end_run(run_id)


def test_end_run_clears_unfinalized_runtime_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)
    runtime = _runtime()
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )

    runtime.begin_step(torch.tensor([1.0]))
    assert module.malloc_graph_enabled("cuda:0") is False
    runtime.end_run(run_id)

    assert not compat._compiler_bypass_active()
    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]


def test_start_run_clears_stale_scope_from_prior_aborted_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)
    compat._enter_compiler_bypass()
    runtime = _runtime()

    run_id = runtime.start_run(
        torch.tensor([1.0, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )

    assert not compat._compiler_bypass_active()
    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]
    runtime.end_run(run_id)


def test_compiler_stays_enabled_outside_spectrum_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)

    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]


def test_active_spectrum_scope_disables_malloc_graph_without_calling_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)

    compat._enter_compiler_bypass()
    assert module.malloc_graph_enabled("cuda:0") is False
    assert calls == []

    compat._exit_compiler_bypass()
    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]


def test_nested_scope_remains_disabled_until_outer_scope_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)

    compat._enter_compiler_bypass()
    compat._enter_compiler_bypass()
    compat._exit_compiler_bypass()
    assert module.malloc_graph_enabled("cuda:0") is False
    assert calls == []

    compat._exit_compiler_bypass()
    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]


def test_end_run_style_clear_cannot_leak_scope_to_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, calls = _install_fake_model_prefetch(monkeypatch)

    compat._enter_compiler_bypass()
    compat._clear_compiler_bypass()

    assert module.malloc_graph_enabled("cuda:0") is True
    assert calls == ["cuda:0"]


def test_model_prefetch_hook_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _install_fake_model_prefetch(monkeypatch)
    installed = module.malloc_graph_enabled

    assert compat._ensure_model_prefetch_hook()
    assert module.malloc_graph_enabled is installed


def test_missing_model_prefetch_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "comfy.model_prefetch", raising=False)
    assert compat._ensure_model_prefetch_hook() is False


def test_reviewed_comfy_compiler_contract_when_present() -> None:
    from comfy import model_prefetch

    gate = getattr(model_prefetch, "malloc_graph_enabled", None)
    if gate is None:
        pytest.skip("reviewed ComfyUI revision predates Comfy Compiler malloc graphs")

    assert callable(gate)
    assert callable(getattr(model_prefetch, "malloc_graph_begin", None))
    assert callable(getattr(model_prefetch, "malloc_graph_end", None))

    # Current Core (a99d1f9 / #16148) exposes this public context manager for
    # long-lived allocations. Older reviewed compiler revisions do not, and the
    # Spectrum compatibility boundary deliberately does not require it.
    pause = getattr(model_prefetch, "pause_malloc_graph", None)
    if pause is not None:
        assert callable(pause)


def test_runtime_scope_compat_is_installed_once() -> None:
    start_run = SpectrumH3Runtime.start_run
    begin_step = SpectrumH3Runtime.begin_step
    finalize_step = SpectrumH3Runtime.finalize_step
    end_run = SpectrumH3Runtime.end_run
    assert getattr(
        SpectrumH3Runtime,
        "_comfy_compiler_compat_installed",
        False,
    )

    compat.install_comfy_compiler_compat()

    assert SpectrumH3Runtime.start_run is start_run
    assert SpectrumH3Runtime.begin_step is begin_step
    assert SpectrumH3Runtime.finalize_step is finalize_step
    assert SpectrumH3Runtime.end_run is end_run
