from __future__ import annotations

import types

from comfyui_spectrum_h3.er_sde_offline_replay_safety import (
    _NOT_KJ,
    _strict_kj_original_callback,
    _trace_er_sde_callback,
)


def _synthetic_kj_callback(original_callback, *, filename="/tmp/ComfyUI-KJNodes/nodes/preview_override_node.py"):
    def factory(original_callback):
        def new_callback(step, x0, x, total_steps):
            if original_callback is not None:
                return original_callback(step, x0, x, total_steps)
            return None

        return new_callback

    callback = factory(original_callback)
    callback.__module__ = "custom_nodes.ComfyUI_KJNodes.nodes.preview_override_node"
    callback.__qualname__ = "_PreviewOverrideWrapper.__call__.<locals>.new_callback"
    callback.__code__ = callback.__code__.replace(co_filename=filename)
    return callback


def test_strict_kj_callback_unwrap_preserves_underlying_callback():
    calls = []

    def underlying(step, x0, x, total_steps):
        calls.append((step, total_steps))

    callback = _synthetic_kj_callback(underlying)
    assert _strict_kj_original_callback(callback) is underlying


def test_strict_kj_callback_unwrap_accepts_none_underlying_callback():
    callback = _synthetic_kj_callback(None)
    assert _strict_kj_original_callback(callback) is None


def test_strict_kj_callback_rejects_unrelated_callback_provenance():
    callback = _synthetic_kj_callback(
        lambda *_args: None,
        filename="/tmp/other/preview_override_node.py",
    )
    assert _strict_kj_original_callback(callback) is _NOT_KJ


def test_strict_kj_callback_rejects_nonfunction_callable():
    class Callable:
        def __call__(self, *_args):
            return None

    assert _strict_kj_original_callback(Callable()) is _NOT_KJ


def test_er_sde_callback_trace_brackets_callback_without_touching_arguments():
    events = []
    calls = []

    class Config:
        debug = True

    class Runtime:
        config = Config()
        active_run_id = 7
        offline_phase = "replay"

        @staticmethod
        def log_offline_transition(event, **fields):
            events.append((event, fields))

    sentinel_x0 = object()
    sentinel_x = object()

    def callback(step, x0, x, total_steps):
        calls.append((step, x0, x, total_steps))
        return "ok"

    traced = _trace_er_sde_callback(callback, Runtime())
    assert isinstance(traced, types.FunctionType)
    assert traced(13, sentinel_x0, sentinel_x, 20) == "ok"
    assert calls == [(13, sentinel_x0, sentinel_x, 20)]
    assert [event for event, _ in events] == [
        "er_sde_callback_begin",
        "er_sde_callback_end",
    ]
    assert events[0][1]["step"] == 13
    assert events[1][1]["step"] == 13


def test_er_sde_callback_trace_logs_end_when_callback_raises():
    events = []

    class Config:
        debug = True

    class Runtime:
        config = Config()
        active_run_id = 9
        offline_phase = "replay"

        @staticmethod
        def log_offline_transition(event, **fields):
            events.append(event)

    def callback(*_args):
        raise RuntimeError("boom")

    traced = _trace_er_sde_callback(callback, Runtime())
    try:
        traced(3, object(), object(), 20)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("callback exception was not propagated")
    assert events == ["er_sde_callback_begin", "er_sde_callback_end"]


def test_er_sde_callback_trace_is_inert_without_debug():
    class Config:
        debug = False

    class Runtime:
        config = Config()

    def callback(*_args):
        return None

    assert _trace_er_sde_callback(callback, Runtime()) is callback
