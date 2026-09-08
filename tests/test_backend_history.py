import pytest
import torch

from comfyui_spectrum_h3.backend_history import BackendHistory, observe, prepare
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


class Policy:
    def __init__(
        self,
        identity="dense",
        safe=True,
        *,
        preflight_error=None,
        receipt_error=None,
    ):
        self.identity = identity
        self.safe = safe
        self.preflight_error = preflight_error
        self.receipt_error = receipt_error

    def __call__(self, **kw):
        if self.preflight_error is not None:
            raise self.preflight_error
        return self.identity if self.safe else None

    def accept_receipts(self, receipts):
        if self.receipt_error is not None:
            raise self.receipt_error
        return all(item[2] != "fallback" for item in receipts)


def runtime():
    config = SpectrumH3Config(
        degree=1,
        max_history=4,
        warmup_steps=2,
        tail_actual_steps=0,
        window_size=2,
        bootstrap_first_forecast=False,
        offline_smoothing_replay=False,
    )
    r = SpectrumH3Runtime(config)
    r.start_run(torch.linspace(1, 0, 10), "euler", supported_sampler=True)
    return r


def call(r, policy, t, receipt="dense", want_actual=None):
    decision = r.begin_step(torch.tensor([t]))
    run, step = decision["run_id"], decision["step_id"]
    options, pending = prepare(
        r,
        run,
        step,
        {"attention_backend_history_v1": {"test": policy}},
        None,
        None,
    )
    idx, actual = r.begin_model_call(
        run,
        step,
        topology=(
            ("video", (1, 24, 2, 4, 4)),
            ("audio", (1, 32, 2, 8)),
            ("hidden", 4),
            ("target_audio_rows", 1),
            ("target_video_rows", 2),
        ),
        labels=((0, "positive"),),
        expected_shape=(1, 3, 4),
    )
    if want_actual is not None:
        assert actual is want_actual
    if actual:
        options["attention_backend_receipts_v1"].append(("test", 0, receipt))
        observe(r, run, step, options, pending)
        r.observe_actual(run, step, idx, torch.ones(1, 3, 4) * t)
    else:
        assert (
            r.predict(
                run,
                step,
                idx,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            is not None
        )
    r.finalize_step(run, step)
    return actual


def test_dense_sol_fallback_and_resume_partition_history():
    r = runtime()
    p = Policy()
    call(r, p, 1.0, want_actual=True)
    call(r, p, 0.9, want_actual=True)
    call(r, p, 0.8, want_actual=False)
    p.identity = "sol"
    call(r, p, 0.7, receipt="sol", want_actual=True)
    assert r.forecaster.history_length == 1
    assert not r.stats.disabled
    call(r, p, 0.6, receipt="sol", want_actual=True)
    # An observed per-call fallback cannot pollute preceding SOL anchors.
    r._required_actual_refreshes = 1
    call(r, p, 0.5, receipt="fallback", want_actual=True)
    assert r.forecaster.history_length == 1
    call(r, p, 0.4, receipt="sol", want_actual=True)
    assert r.forecaster.history_length == 1
    assert r.stats.backend_history_resets >= 3


def test_opaque_provider_and_core_bsa_execute_without_run_disable():
    r = runtime()
    p = Policy(safe=False)
    for t in (1.0, 0.9, 0.8, 0.7):
        call(r, p, t, want_actual=True)
    assert not r.stats.disabled
    decision = r.begin_step(torch.tensor([0.6]))
    _options, pending = prepare(
        r,
        decision["run_id"],
        decision["step_id"],
        {"callbacks": {"on_prepare_state": {"block_sparse_attention": []}}},
        None,
        None,
    )
    assert pending is not None
    assert r._step.mode == "actual"
    assert not r.stats.disabled


def test_provider_callback_failures_are_actual_only_without_disabling_run():
    r = runtime()
    call(
        r,
        Policy(preflight_error=RuntimeError("preflight failed")),
        1.0,
        want_actual=True,
    )
    assert not r._backend_history.forecast_safe
    assert not r.stats.disabled
    call(
        r,
        Policy(receipt_error=RuntimeError("receipt failed")),
        0.9,
        want_actual=True,
    )
    assert not r._backend_history.forecast_safe
    assert not r.stats.disabled


def test_provider_callback_oom_still_propagates():
    r = runtime()
    decision = r.begin_step(torch.tensor([1.0]))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        prepare(
            r,
            decision["run_id"],
            decision["step_id"],
            {
                "attention_backend_history_v1": {
                    "test": Policy(
                        preflight_error=torch.cuda.OutOfMemoryError("preflight oom")
                    )
                }
            },
            None,
            None,
        )

    r = runtime()
    decision = r.begin_step(torch.tensor([1.0]))
    run, step = decision["run_id"], decision["step_id"]
    options, pending = prepare(
        r,
        run,
        step,
        {
            "attention_backend_history_v1": {
                "test": Policy(
                    receipt_error=torch.cuda.OutOfMemoryError("receipt oom")
                )
            }
        },
        None,
        None,
    )
    options["attention_backend_receipts_v1"].append(("test", 0, "dense"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        observe(r, run, step, options, pending)


def test_backend_identity_is_run_scoped_and_rollback_restored():
    r = runtime()
    call(r, Policy(), 1.0)
    snapshot = r.create_rollback_snapshot()
    call(r, Policy("sol"), 0.9, receipt="sol")
    r.restore_rollback_snapshot(snapshot)
    assert r._backend_history.policy == (("test", "dense"),)
    r.end_run(r.active_run_id)
    r.start_run(torch.linspace(1, 0, 10), "euler", supported_sampler=True)
    assert r._backend_history == BackendHistory()


def test_history_reset_clears_every_stage_bank():
    r = runtime()
    from comfyui_spectrum_h3.forecast import HistoryWeightForecaster

    bank = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.01,
        max_history=4,
        history_storage="system_ram",
    )
    r._stage_forecasters[1] = bank
    call(r, Policy("dense"), 1.0, receipt="dense")
    call(r, Policy("sol"), 0.9, receipt="sol")
    assert bank.history_length == 0


def test_same_step_mixed_receipts_are_executed_but_not_archived():
    r = runtime()
    p = Policy("sol")
    decision = r.begin_step(torch.tensor([1.0]))
    run, step = decision["run_id"], decision["step_id"]
    for branch, route in enumerate(("sol", "fallback")):
        options, pending = prepare(
            r,
            run,
            step,
            {"attention_backend_history_v1": {"test": p}},
            None,
            None,
        )
        idx, actual = r.begin_model_call(
            run,
            step,
            topology=("native",),
            labels=((branch, str(branch)),),
            expected_shape=(1, 3, 4),
        )
        assert actual
        options["attention_backend_receipts_v1"].append(("test", 0, route))
        observe(r, run, step, options, pending)
        r.observe_actual(run, step, idx, torch.ones(1, 3, 4))
    assert not r._step.retain_history
    assert not r._step.actual_records
    r.finalize_step(run, step)
    assert r.forecaster.history_length == 0
    assert r.stats.actual_transformer_calls == 2
    assert not r.stats.disabled


def test_initial_backend_identity_preserves_empty_offline_archive_then_transition_invalidates():
    r = runtime()
    r.begin_offline_capture(total_steps=9, sampler_name="euler")
    p = Policy("sol")
    call(r, p, 1.0, receipt="sol", want_actual=True)
    assert r._offline_archive.valid
    assert r.stats.backend_history_resets == 0
    p.identity = "dense"
    call(r, p, 0.9, receipt="dense", want_actual=True)
    assert not r._offline_archive.valid
    assert r.stats.backend_history_resets >= 1
    assert not r.stats.disabled


def test_removed_provider_does_not_reuse_sparse_anchors():
    r = runtime()
    call(r, Policy("sol"), 1.0, receipt="sol", want_actual=True)
    decision = r.begin_step(torch.tensor([0.9]))
    prepare(r, decision["run_id"], decision["step_id"], {}, None, None)
    assert r.forecaster.history_length == 0
    assert r._step.mode == "actual"
    assert not r.stats.disabled
