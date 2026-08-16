from __future__ import annotations

import inspect

from comfyui_spectrum_h3 import external_patch_compat as compat


def _assert_order(source: str, tokens: tuple[str, ...]) -> None:
    cursor = -1
    for token in tokens:
        position = source.find(token, cursor + 1)
        assert position >= 0, token
        cursor = position


def test_predict_noise_wrapper_transaction_operation_order_matches_native():
    original = compat._ORIGINAL_PREDICT_NOISE_WRAPPER
    assert original is not None
    base_source = inspect.getsource(original)
    compat_source = inspect.getsource(compat._predict_noise_wrapper)

    transaction_order = (
        "decision = runtime.begin_step(timestep)",
        "result = execute_attempt(decision)",
        "runtime.log_offline_transition(",
        "result = consume_er_sde_increment(result, decision)",
        'runtime.finalize_step(decision["run_id"], decision["step_id"])',
        "runtime.prepare_actual_retry(",
        "retry_decision = dict(decision)",
        'retry_decision["actual"] = True',
        "result = execute_attempt(retry_decision)",
        "runtime.log_offline_transition(",
        "result = consume_er_sde_increment(result, retry_decision)",
        'runtime.finalize_step(decision["run_id"], decision["step_id"])',
        "except BaseException:",
        'runtime.abort_step(decision["run_id"], decision["step_id"])',
    )
    _assert_order(base_source, transaction_order)
    _assert_order(compat_source, transaction_order)

    # The compat-only effective-mode log must observe the post-model promotion
    # before any solver-side consumption on both the primary and retry paths.
    compat_order = (
        "result = execute_attempt(decision)",
        "_log_effective_step(runtime, decision)",
        "runtime.log_offline_transition(",
        "result = consume_er_sde_increment(result, decision)",
        "result = execute_attempt(retry_decision)",
        "_log_effective_step(runtime, retry_decision)",
        "runtime.log_offline_transition(",
        "result = consume_er_sde_increment(result, retry_decision)",
    )
    _assert_order(compat_source, compat_order)
