from __future__ import annotations

from functools import wraps
import sys
import threading
from typing import Any

from .runtime import SpectrumH3Runtime


_STATE = threading.local()
_RUNTIME_INSTALL_MARKER = "_comfy_compiler_compat_installed"


def _bypass_depth() -> int:
    return int(getattr(_STATE, "bypass_depth", 0))


def _compiler_bypass_active() -> bool:
    return _bypass_depth() > 0


def _enter_compiler_bypass() -> None:
    _STATE.bypass_depth = _bypass_depth() + 1


def _exit_compiler_bypass() -> None:
    depth = _bypass_depth()
    if depth <= 1:
        _STATE.bypass_depth = 0
    else:
        _STATE.bypass_depth = depth - 1


def _clear_compiler_bypass() -> None:
    _STATE.bypass_depth = 0


def _ensure_model_prefetch_hook() -> bool:
    """Make current-thread Spectrum steps opt out of Comfy Compiler malloc graphs.

    The hook is installed lazily because Spectrum's package can be imported by
    smoke tests before ComfyUI itself is importable. During an active Spectrum
    solver step the native MiniMax H3 ``forward`` therefore sees
    ``malloc_graph_enabled(...) == False`` and does not enter Aimdo's persistent
    allocation graph. DynamicVRAM and Comfy Compiler remain unchanged outside
    that narrow thread-local step scope.
    """
    model_prefetch = sys.modules.get("comfy.model_prefetch")
    if model_prefetch is None:
        return False

    current = getattr(model_prefetch, "malloc_graph_enabled", None)
    if not callable(current):
        return False
    installed = getattr(
        model_prefetch,
        "_spectrum_h3_malloc_graph_enabled_wrapper",
        None,
    )
    if current is installed:
        return True

    @wraps(current)
    def spectrum_safe_malloc_graph_enabled(device: Any) -> bool:
        if _compiler_bypass_active():
            return False
        return bool(current(device))

    model_prefetch.malloc_graph_enabled = spectrum_safe_malloc_graph_enabled
    model_prefetch._spectrum_h3_malloc_graph_enabled_wrapper = (
        spectrum_safe_malloc_graph_enabled
    )
    return True


def install_comfy_compiler_compat() -> None:
    """Install Spectrum's step-scoped Comfy Compiler compatibility boundary.

    The first #99 candidate only paused retained-history allocation. The reporter's
    CUDA retest completed one prompt but the next prompt still crashed in
    ``MallocGraph.pop``. Spectrum changes the native H3 allocation/control trace
    more broadly than its retained history alone (actual calls execute the
    transformer while forecast/replay calls skip it), so the safe boundary is the
    complete Spectrum solver step, before native H3 decides whether to start a
    malloc graph.

    This installer must run after Spectrum's other runtime compatibility layers so
    its start/step/finalize/end scopes remain the outermost lifecycle boundary.
    """
    if getattr(SpectrumH3Runtime, _RUNTIME_INSTALL_MARKER, False):
        _ensure_model_prefetch_hook()
        return

    original_start_run = SpectrumH3Runtime.start_run
    original_begin_step = SpectrumH3Runtime.begin_step
    original_finalize_step = SpectrumH3Runtime.finalize_step
    original_end_run = SpectrumH3Runtime.end_run

    @wraps(original_start_run)
    def start_run(self: SpectrumH3Runtime, *args: Any, **kwargs: Any):
        # Recover conservatively if an earlier Python exception escaped before the
        # normal end_run/finalize cleanup path on this worker thread.
        _clear_compiler_bypass()
        _ensure_model_prefetch_hook()
        return original_start_run(self, *args, **kwargs)

    @wraps(original_begin_step)
    def begin_step(self: SpectrumH3Runtime, *args: Any, **kwargs: Any):
        result = original_begin_step(self, *args, **kwargs)
        _ensure_model_prefetch_hook()
        _enter_compiler_bypass()
        return result

    @wraps(original_finalize_step)
    def finalize_step(self: SpectrumH3Runtime, *args: Any, **kwargs: Any):
        try:
            return original_finalize_step(self, *args, **kwargs)
        finally:
            _exit_compiler_bypass()

    @wraps(original_end_run)
    def end_run(self: SpectrumH3Runtime, *args: Any, **kwargs: Any):
        try:
            return original_end_run(self, *args, **kwargs)
        finally:
            # An exception can abort a sampler before finalize_step. Never let a
            # stale thread-local bypass leak into an unrelated later workflow.
            _clear_compiler_bypass()

    SpectrumH3Runtime.start_run = start_run
    SpectrumH3Runtime.begin_step = begin_step
    SpectrumH3Runtime.finalize_step = finalize_step
    SpectrumH3Runtime.end_run = end_run
    setattr(SpectrumH3Runtime, _RUNTIME_INSTALL_MARKER, True)
    _ensure_model_prefetch_hook()


__all__ = [
    "install_comfy_compiler_compat",
]
