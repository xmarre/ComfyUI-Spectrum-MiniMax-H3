from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import math
import os
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.res4lyf_stochastic import (
    RES4LYF_STOCHASTIC_BRIDGE_KEY,
    RES4LYFStepDescriptor,
    RES4LYFStochasticBridge,
    RES4LYFStochasticError,
    _rms,
    tracked_noise_sampler_class,
    tracked_res4lyf_noise_sampler,
)


def _coordinate(sigma: float) -> float:
    # Half-log-SNR for a CONST/flow model: log((1 - sigma) / sigma).
    return math.log((1.0 - sigma) / sigma)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_debug_rms_accepts_native_noise_dtypes(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    scale = 1e100 if dtype == torch.float64 else 1.0
    value = torch.tensor([3.0 * scale, 4.0 * scale], dtype=dtype, device=device)
    assert _rms(value) == pytest.approx(math.sqrt(12.5) * scale)
    assert _rms(value[:0]) == 0.0


def _bridge(run_id: int = 1) -> RES4LYFStochasticBridge:
    bridge = RES4LYFStochasticBridge(
        run_id=run_id,
        coordinate_fn=_coordinate,
        debug=False,
    )
    assert bridge.bind_noise_sampler(object())
    return bridge


def _swap(
    bridge: RES4LYFStochasticBridge,
    kind: str,
    landing: torch.Tensor,
    result: torch.Tensor,
    sigma: float,
    *,
    draws: int = 1,
) -> None:
    bridge.enter_swap(kind)
    for _ in range(draws):
        bridge.record_noise_draw(kind)
    bridge.exit_swap(kind, landing=landing, result=result, target_sigma=torch.tensor(sigma))


def _noop_swap(bridge: RES4LYFStochasticBridge) -> None:
    """A verified no-op swap, as RES4LYF runs between consecutive model calls."""
    landing = torch.zeros(1, 2)
    _swap(bridge, "substep", landing, landing, 0.5, draws=0)


def _between_calls(bridge: RES4LYFStochasticBridge) -> None:
    if bridge._associated_step_id is not None:
        _noop_swap(bridge)


def _actual(bridge, value, sigma, step_id, x=None, run_id=1):
    _between_calls(bridge)
    model_input = torch.zeros_like(value) if x is None else x
    bridge.associate(
        model_input,
        torch.tensor([sigma]),
        run_id=run_id,
        step_id=step_id,
    )
    return bridge.consume(
        value,
        torch.tensor([sigma]),
        RES4LYFStepDescriptor(run_id, step_id, "actual"),
        model_input=model_input,
    )


def _forecast(bridge, raw, sigma, step_id, x=None, run_id=1):
    _between_calls(bridge)
    model_input = torch.zeros_like(raw) if x is None else x
    bridge.associate(
        model_input,
        torch.tensor([sigma]),
        run_id=run_id,
        step_id=step_id,
    )
    return bridge.consume(
        raw,
        torch.tensor([sigma]),
        RES4LYFStepDescriptor(run_id, step_id, "forecast"),
        model_input=model_input,
    )


def test_noised_state_is_associated_once_with_the_exact_following_call():
    bridge = _bridge()
    landing = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    # RES4LYF's float64 noise promotes the swap result; the sampler casts it
    # back into its float32 state buffer before the model call.
    noised = torch.tensor([[1.25, 1.5]], dtype=torch.float64)
    _swap(bridge, "step", landing, noised, 0.6)
    assert bridge.has_pending

    x = noised.to(torch.float32)
    kind = bridge.associate(
        x,
        torch.tensor([0.6, 0.6], dtype=torch.float32),
        run_id=1,
        step_id=4,
    )

    assert kind == "stochastic_step"
    assert not bridge.has_pending
    assert bridge.transitions_recorded == bridge.transitions_consumed == 1
    # A retry of the same call (same input object and sigma, no new swap)
    # reuses the validated association.
    assert bridge.associate(x, torch.tensor([0.6]), run_id=1, step_id=4) == kind
    # After a verified no-op swap the next call has a deterministic input.
    _noop_swap(bridge)
    assert bridge.associate(torch.zeros(1, 2), torch.tensor([0.5]), run_id=1, step_id=5) == (
        "deterministic"
    )


def test_missing_swap_between_calls_is_not_deterministic():
    bridge = _bridge()
    assert bridge.associate(torch.zeros(1, 2), torch.tensor([0.9]), run_id=1, step_id=0) == (
        "deterministic"
    )

    with pytest.raises(RES4LYFStochasticError, match="no RES4LYF noise swap"):
        bridge.associate(torch.zeros(1, 2), torch.tensor([0.8]), run_id=1, step_id=1)
    assert bridge.invalid_reason is not None


@pytest.mark.parametrize("change", ("input", "sigma", "new_swap", "new_transition"))
def test_reassociation_is_limited_to_an_unchanged_retry(change):
    bridge = _bridge()
    x = torch.zeros(1, 2)
    bridge.associate(x, torch.tensor([0.9]), run_id=1, step_id=0)
    sigma = 0.9
    if change == "input":
        x = x.clone()
    elif change == "sigma":
        sigma = 0.8
    elif change == "new_swap":
        _noop_swap(bridge)
    else:
        _swap(bridge, "substep", torch.zeros(1, 2), torch.ones(1, 2), 0.9)

    with pytest.raises(RES4LYFStochasticError, match="associated again"):
        bridge.associate(x, torch.tensor([sigma]), run_id=1, step_id=0)
    assert bridge.invalid_reason is not None


def test_in_place_edit_of_the_swap_result_is_detected():
    bridge = _bridge()
    landing = torch.zeros(1, 2, dtype=torch.float32)
    noised = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    _swap(bridge, "step", landing, noised, 0.6)
    noised.add_(1.0)

    with pytest.raises(RES4LYFStochasticError, match="was modified"):
        bridge.associate(
            noised.to(torch.float32),
            torch.tensor([0.6]),
            run_id=1,
            step_id=1,
        )


def test_noop_swap_records_no_transition():
    bridge = _bridge()
    landing = torch.ones(1, 2)
    _swap(bridge, "substep", landing, landing, 0.5, draws=0)

    assert not bridge.has_pending
    assert bridge.invalid_reason is None
    assert bridge.associate(landing, torch.tensor([0.5]), run_id=1, step_id=0) == "deterministic"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("modified", "was modified"),
        ("sigma", "targets sigma"),
        ("shape", "layout"),
    ),
)
def test_mismatched_following_call_fails_closed(mutation, message):
    bridge = _bridge()
    landing = torch.zeros(1, 2)
    noised = torch.tensor([[0.5, -0.5]])
    _swap(bridge, "substep", landing, noised, 0.7)
    x = noised.clone()
    sigma = 0.7
    if mutation == "modified":
        x[0, 0] += 1e-6
    elif mutation == "sigma":
        sigma = 0.69
    else:
        x = noised.reshape(2, 1)

    with pytest.raises(RES4LYFStochasticError, match=message):
        bridge.associate(x, torch.tensor([sigma]), run_id=1, step_id=1)
    assert bridge.invalid_reason is not None
    with pytest.raises(RES4LYFStochasticError):
        bridge.associate(torch.zeros(1, 2), torch.tensor([0.5]), run_id=1, step_id=2)


def test_unconsumed_transition_is_never_silently_replaced():
    bridge = _bridge()
    landing = torch.zeros(1, 2)
    _swap(bridge, "substep", landing, torch.ones(1, 2), 0.7)
    _swap(bridge, "step", landing, torch.full((1, 2), 2.0), 0.6)

    assert "never consumed" in (bridge.invalid_reason or "")
    with pytest.raises(RES4LYFStochasticError, match="never consumed"):
        bridge.associate(torch.full((1, 2), 2.0), torch.tensor([0.6]), run_id=1, step_id=1)


@pytest.mark.parametrize(
    ("kind", "channel", "draws", "applied", "message"),
    (
        (None, "step", 1, True, "untracked RES4LYF stochastic draw"),
        ("step", "substep", 1, True, "drew from the substep"),
        ("substep", "substep", 0, True, "consumed 0 draws"),
        ("substep", "substep", 2, True, "consumed 2 draws"),
        ("step", "step", 1, False, "consumed 1 draws"),
    ),
)
def test_untracked_or_miscounted_stochastic_draws_fail_closed(
    kind, channel, draws, applied, message
):
    bridge = _bridge()
    landing = torch.zeros(1, 2)
    result = torch.ones(1, 2) if applied else landing
    if kind is None:
        bridge.record_noise_draw(channel)
    else:
        bridge.enter_swap(kind)
        for _ in range(draws):
            bridge.record_noise_draw(channel)
        bridge.exit_swap(kind, landing=landing, result=result, target_sigma=0.5)

    assert message in (bridge.invalid_reason or "")


def test_unbound_or_stale_bridge_fails_closed():
    unbound = RES4LYFStochasticBridge(run_id=3, coordinate_fn=_coordinate, debug=False)
    with pytest.raises(RES4LYFStochasticError, match="not created through the tracked bridge"):
        unbound.associate(torch.zeros(1), torch.tensor([0.5]), run_id=3, step_id=0)

    stale = _bridge(run_id=3)
    with pytest.raises(RES4LYFStochasticError, match="stale"):
        stale.associate(torch.zeros(1), torch.tensor([0.5]), run_id=4, step_id=0)


def test_second_noise_sampler_for_the_same_run_invalidates_tracking():
    bridge = RES4LYFStochasticBridge(run_id=1, coordinate_fn=_coordinate, debug=False)
    first, second = object(), object()
    assert bridge.bind_noise_sampler(first)
    assert not bridge.bind_noise_sampler(second)
    assert "second noise sampler" in (bridge.invalid_reason or "")


def test_actual_results_are_returned_unchanged_and_retained_as_owned_anchors():
    bridge = _bridge()
    values = [torch.full((1, 2), float(i)) for i in range(3)]
    for step_id, (value, sigma) in enumerate(zip(values, (0.9, 0.8, 0.7))):
        returned = _actual(bridge, value, sigma, step_id)
        assert returned is value
    values[2].add_(100.0)

    assert bridge.anchor_steps == (1, 2)
    forecast = _forecast(bridge, torch.zeros(1, 2), 0.6, 3)
    # Consecutive exact anchors are held, and the held value is the owned copy.
    torch.testing.assert_close(forecast, torch.full((1, 2), 2.0), rtol=0, atol=0)


def test_first_forecast_holds_the_only_exact_anchor_and_ignores_raw_forecast():
    bridge = _bridge()
    anchor = torch.tensor([[1.0, -1.0]])
    _actual(bridge, anchor, 0.8, 0)
    raw = torch.tensor([[50.0, 50.0]])

    result = _forecast(bridge, raw, 0.7, 1)

    torch.testing.assert_close(result, anchor, rtol=0, atol=0)
    assert result is not anchor


def test_isolated_forecast_uses_bounded_coordinate_extrapolation():
    bridge = _bridge()
    previous = torch.tensor([[0.0, 2.0]])
    latest = torch.tensor([[1.0, 1.0]])
    _actual(bridge, previous, 0.8, 0)
    _forecast(bridge, torch.zeros(1, 2), 0.75, 1)
    _actual(bridge, latest, 0.7, 2)

    result = _forecast(bridge, torch.zeros(1, 2), 0.65, 3)

    alpha = (_coordinate(0.65) - _coordinate(0.7)) / (_coordinate(0.7) - _coordinate(0.8))
    assert 0.0 < alpha < 1.0
    expected = latest + alpha * (latest - previous)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("target_sigma", "reason"),
    (
        (0.75, "reversed geometry"),
        (0.2, "extrapolation beyond one anchor interval"),
    ),
)
def test_untrusted_extrapolation_geometry_holds_latest_anchor(target_sigma, reason):
    del reason
    bridge = _bridge()
    _actual(bridge, torch.tensor([[0.0, 2.0]]), 0.8, 0)
    _forecast(bridge, torch.zeros(1, 2), 0.75, 1)
    latest = torch.tensor([[1.0, 1.0]])
    _actual(bridge, latest, 0.7, 2)

    result = _forecast(bridge, torch.zeros(1, 2), target_sigma, 3)

    torch.testing.assert_close(result, latest, rtol=0, atol=0)


def test_continuum_keeps_a_zero_order_hold_for_every_forecast():
    bridge = RES4LYFStochasticBridge(
        run_id=1,
        coordinate_fn=_coordinate,
        debug=False,
        hold_only=True,
    )
    assert bridge.bind_noise_sampler(object())
    _actual(bridge, torch.tensor([[0.0, 2.0]]), 0.8, 0)
    _forecast(bridge, torch.zeros(1, 2), 0.75, 1)
    latest = torch.tensor([[1.0, 1.0]])
    _actual(bridge, latest, 0.7, 2)

    result = _forecast(bridge, torch.zeros(1, 2), 0.65, 3)

    torch.testing.assert_close(result, latest, rtol=0, atol=0)


def test_forecast_requires_an_immediately_preceding_exact_anchor():
    bridge = _bridge()
    with pytest.raises(RES4LYFStochasticError, match="no exact H3 anchor"):
        _forecast(bridge, torch.zeros(1, 2), 0.7, 0)

    bridge = _bridge()
    _actual(bridge, torch.zeros(1, 2), 0.8, 0)
    with pytest.raises(RES4LYFStochasticError, match="immediately before"):
        _forecast(bridge, torch.zeros(1, 2), 0.7, 2)


def test_result_requires_association_and_rejects_replay():
    bridge = _bridge()
    with pytest.raises(RES4LYFStochasticError, match="without its stochastic-state association"):
        bridge.consume(
            torch.zeros(1, 2),
            torch.tensor([0.5]),
            RES4LYFStepDescriptor(1, 0, "actual"),
            model_input=torch.zeros(1, 2),
        )
    x = torch.zeros(1, 2)
    bridge.associate(x, torch.tensor([0.5]), run_id=1, step_id=0)
    with pytest.raises(RES4LYFStochasticError, match="replay"):
        bridge.consume(
            torch.zeros(1, 2),
            torch.tensor([0.5]),
            RES4LYFStepDescriptor(1, 0, "replay"),
            model_input=x,
        )


class _NativeNoiseSampler:
    """Small stand-in exercising only the tracked subclass contract."""

    def __init__(self, RK, model, *args, **kwargs):
        del RK, args, kwargs
        self.model = model
        self.sigma_next = torch.tensor(0.5)
        self.sub_sigma_next = torch.tensor(0.75)
        self.noise_sampler = None
        self.noise_sampler2 = None

    def init_noise_samplers(self):
        self.noise_sampler = lambda **_kwargs: torch.ones(1, 2)
        self.noise_sampler2 = lambda **_kwargs: torch.full((1, 2), 2.0)

    def swap_noise_step(self, x_0, x_next, mask=None):
        del x_0, mask
        return x_next + self.noise_sampler(sigma=1.0, sigma_next=0.5)

    def swap_noise_substep(self, x_0, x_next, mask=None, guide=None):
        del x_0, mask, guide
        return x_next + self.noise_sampler2(sigma=1.0, sigma_next=0.75)


def test_tracked_subclass_binds_only_the_owner_run_and_counts_native_draws():
    bridge = RES4LYFStochasticBridge(run_id=1, coordinate_fn=_coordinate, debug=False)
    guider = object()
    tracked = tracked_noise_sampler_class(_NativeNoiseSampler, bridge, guider)

    foreign = tracked(None, SimpleNamespace(inner_model=object()))
    foreign.init_noise_samplers()
    torch.testing.assert_close(foreign.swap_noise_step(None, torch.zeros(1, 2)), torch.ones(1, 2))
    assert not bridge.bound
    assert bridge.invalid_reason is None

    owned = tracked(None, SimpleNamespace(inner_model=guider))
    owned.init_noise_samplers()
    landing = torch.zeros(1, 2)
    noised = owned.swap_noise_substep(None, landing)

    torch.testing.assert_close(noised, torch.full((1, 2), 2.0))
    assert bridge.bound
    assert bridge.has_pending
    assert bridge.associate(noised, torch.tensor([0.75]), run_id=1, step_id=0) == (
        "stochastic_substep"
    )
    # A direct draw outside a reviewed swap is an untracked stochastic path.
    owned.noise_sampler(sigma=1.0, sigma_next=0.5)
    assert "untracked" in (bridge.invalid_reason or "")


def test_tracked_class_is_installed_for_one_call_and_always_restored():
    module = SimpleNamespace(RK_NoiseSampler=_NativeNoiseSampler)
    bridge = RES4LYFStochasticBridge(run_id=1, coordinate_fn=_coordinate, debug=False)

    with (
        pytest.raises(RuntimeError, match="boom"),
        tracked_res4lyf_noise_sampler(module, _NativeNoiseSampler, bridge, object()) as tracked,
    ):
        assert module.RK_NoiseSampler is tracked
        assert issubclass(tracked, _NativeNoiseSampler)
        raise RuntimeError("boom")
    assert module.RK_NoiseSampler is _NativeNoiseSampler

    foreign = type("Foreign", (_NativeNoiseSampler,), {})
    module.RK_NoiseSampler = foreign
    with (
        pytest.raises(RES4LYFStochasticError, match="already replaced"),
        tracked_res4lyf_noise_sampler(module, _NativeNoiseSampler, bridge, object()),
    ):
        pass
    assert module.RK_NoiseSampler is foreign


# ---------------------------------------------------------------------------
# Reviewed RES4LYF runtime fixture
# ---------------------------------------------------------------------------

RES4LYF_SDE_WRAPPERS = ("res_2m", "res_3m", "res_2s", "res_3s", "res_5s", "res_6s")
_RES4LYF_IMPORT: dict[str, object] = {}


def _res4lyf_runtime_fixture():
    root = os.environ.get("RES4LYF_PATH")
    if not root:
        pytest.skip("reviewed RES4LYF source fixture is unavailable")
    if "modules" in _RES4LYF_IMPORT:
        return _RES4LYF_IMPORT["modules"]
    required = os.environ.get("SPECTRUM_REQUIRE_RES4LYF_RUNTIME") == "1"
    package = "_spectrum_res4lyf_runtime_fixture"
    try:
        import server

        had_instance = hasattr(server.PromptServer, "instance")
        if not had_instance:

            class _Routes:
                def __getattr__(self, _name):
                    return lambda *_args, **_kwargs: (lambda function: function)

            # RES4LYF registers settings routes at import time; the sampler code
            # under test does not use them.
            server.PromptServer.instance = SimpleNamespace(routes=_Routes())
        # RES4LYF's node definitions import ComfyUI's top-level ``nodes`` module,
        # which this repository's own entry point shadows on the test path.
        previous_nodes = sys.modules.get("nodes")
        comfy_root = os.environ.get("COMFYUI_PATH")
        if comfy_root and not hasattr(previous_nodes, "MAX_RESOLUTION"):
            spec = importlib.util.spec_from_file_location(
                "nodes",
                os.path.join(comfy_root, "nodes.py"),
            )
            comfy_nodes = importlib.util.module_from_spec(spec)
            sys.modules["nodes"] = comfy_nodes
            spec.loader.exec_module(comfy_nodes)
        try:
            namespace = types.ModuleType(package)
            namespace.__path__ = [root]
            sys.modules[package] = namespace
            beta = importlib.import_module(f"{package}.beta")
            rk_sampler_beta = importlib.import_module(f"{package}.beta.rk_sampler_beta")
        finally:
            if previous_nodes is None:
                sys.modules.pop("nodes", None)
            else:
                sys.modules["nodes"] = previous_nodes
            if not had_instance:
                del server.PromptServer.instance
    except Exception as exc:  # noqa: BLE001 - optional fixture on old ComfyUI lanes
        for name in [name for name in sys.modules if name.startswith(package)]:
            del sys.modules[name]
        if required:
            pytest.fail(f"reviewed RES4LYF runtime fixture failed to import: {exc!r}")
        pytest.skip(f"reviewed RES4LYF runtime fixture is unavailable here: {exc!r}")
    _RES4LYF_IMPORT["modules"] = (beta, rk_sampler_beta)
    return beta, rk_sampler_beta


def _flow_model_sampling():
    import comfy.model_sampling

    class FlowSampling(
        comfy.model_sampling.ModelSamplingDiscreteFlow,
        comfy.model_sampling.CONST,
    ):
        pass

    return FlowSampling(None)


_SHAPE = (1, 4, 16, 16)


def _manifold():
    generator = torch.Generator().manual_seed(1234)
    mu = torch.randn(_SHAPE, generator=generator) * 0.3
    basis, _ = torch.linalg.qr(
        torch.randn(mu.numel(), 12, generator=generator, dtype=torch.float64)
    )
    return mu, basis.to(torch.float32)


def _denoiser(mu, basis):
    """Exact posterior mean for data on a low-rank Gaussian manifold."""

    def denoise(x, sigma):
        s = sigma.reshape(-1).to(torch.float32)[0]
        a = 1.0 - s
        flat = (x - a * mu).reshape(x.shape[0], -1)
        coefficient = a / (a * a + s * s)
        return mu + (coefficient * (flat @ basis) @ basis.T).reshape(x.shape)

    return denoise


class _Guider:
    def __init__(self, model_sampling, call, *, av=False):
        self.inner_model = SimpleNamespace(
            device=torch.device("cpu"),
            model_sampling=model_sampling,
            diffusion_model=SimpleNamespace(),
        )
        self.model_options = {"transformer_options": {}}
        self.model_patcher = SimpleNamespace(
            get_model_object=lambda name: model_sampling if name == "model_sampling" else None
        )
        self._call = call
        if av:
            # Exercise RES4LYF's packed-column AV noise split and distinct
            # stream schedules without loading an H3 transformer or VAE.
            self.conds = {
                "positive": [{"model_conds": {"latent_shapes": SimpleNamespace(cond=[(1, 1, 8), (1, 1, 8)])}}]
            }
            self.inner_model.diffusion_model.sigma_shift_video = 12.0
            self.inner_model.diffusion_model.sigma_shift_audio = 8.0

    def __call__(self, x, sigma, model_options=None, seed=None):
        return self._call(x, sigma, model_options or {}, seed)


class _ModelK:
    """Native KSamplerX0Inpaint call shape without a denoise mask."""

    def __init__(self, guider):
        self.inner_model = guider

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None):
        assert denoise_mask is None
        return self.inner_model(x, sigma, model_options=model_options or {}, seed=seed)


def _initial_noise(seed: int = 7) -> torch.Tensor:
    return torch.randn(_SHAPE, generator=torch.Generator().manual_seed(seed))


_SIGMAS = torch.tensor(
    [1.0, 0.95, 0.89, 0.82, 0.74, 0.65, 0.55, 0.45, 0.35, 0.26, 0.18, 0.11, 0.06, 0.0]
)


def _native_res4lyf(beta, name, denoise, *, av=False):
    model_sampling = _flow_model_sampling()
    guider = _Guider(model_sampling, lambda x, sigma, _options, _seed: denoise(x, sigma), av=av)
    torch.manual_seed(4242)  # RES4LYF seeds SDE noise from torch.initial_seed() + 1.
    return getattr(beta, f"sample_{name}")(
        _ModelK(guider),
        _initial_noise(),
        _SIGMAS.clone(),
        extra_args={"model_options": {"transformer_options": {}}, "seed": 0},
        callback=None,
        disable=True,
    )


def _spectrum_res4lyf(
    beta,
    name,
    denoise,
    config,
    monkeypatch,
    *,
    leaky_forecast=True,
    perturb_call=None,
    denoise_mask=None,
    before_call=None,
    inside_call=None,
    av=False,
):
    from comfyui_spectrum_h3 import sampling as sampling_module
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.sampling import (
        BINDING_KEY,
        RUN_ID_KEY,
        STEP_ID_KEY,
        SpectrumH3Binding,
        outer_sample_wrapper,
        predict_noise_wrapper,
        sampler_sample_wrapper,
    )

    runtime = SpectrumH3Runtime(config)
    model_sampling = _flow_model_sampling()
    record = {
        "modes": [],
        "kinds": [],
        "returned": {},
        "raw": {},
        "exact": {},
        "last_v": None,
    }

    def fake_h3(x, timestep, model_options, _seed):
        options = model_options["transformer_options"]
        run_id, step_id = options[RUN_ID_KEY], options[STEP_ID_KEY]
        call_id, actual = runtime.begin_model_call(
            run_id,
            step_id,
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            labels=((0, "positive"),),
            expected_shape=(1, 2, 1),
        )
        sigma = timestep.reshape(-1)[0]
        exact = denoise(x, timestep)
        record["exact"][len(record["modes"])] = exact
        if inside_call is not None:
            inside_call(
                len(record["modes"]),
                actual,
                options[RES4LYF_STOCHASTIC_BRIDGE_KEY],
            )
        if actual:
            runtime.observe_actual(
                run_id,
                step_id,
                call_id,
                torch.full((1, 2, 1), float(sigma)),
            )
            record["last_v"] = (x - exact) / sigma
            return exact
        assert runtime.predict(
            run_id,
            step_id,
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ) is not None
        # Stand-in for a hidden-feature forecast that cannot see fresh noise.
        raw = x - sigma * record["last_v"] if leaky_forecast else exact
        record["raw"][step_id] = raw
        return raw

    class PredictExecutor:
        def __init__(self, guider):
            self.class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            return fake_h3(x, timestep, model_options, seed)

    def guided_call(x, sigma, model_options, seed):
        if before_call is not None:
            before_call(len(record["modes"]))
        if perturb_call is not None and len(record["modes"]) == perturb_call:
            # Stand-in for an unreviewed outer PREDICT_NOISE wrapper that edits x.
            x = x + 1e-3
        result = predict_noise_wrapper(PredictExecutor(guider), x, sigma, model_options, seed)
        record["modes"].append(runtime.last_completed_mode)
        record["returned"][len(record["modes"]) - 1] = result
        return result

    guider = _Guider(model_sampling, guided_call, av=av)
    guider.model_options[BINDING_KEY] = SpectrumH3Binding(runtime)

    sampler = SimpleNamespace(
        sampler_function=getattr(beta, f"sample_{name}"),
        extra_options={},
    )

    class SamplerExecutor:
        class_obj = sampler

        def __call__(
            self,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        ):
            del latent_image
            bridge = extra_args["model_options"]["transformer_options"].get(
                RES4LYF_STOCHASTIC_BRIDGE_KEY
            )
            record["bridge"] = bridge
            torch.manual_seed(4242)
            return sampler.sampler_function(
                _ModelK(model_wrap),
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable_pbar,
            )

    class OuterExecutor:
        class_obj = guider

        def __call__(
            self,
            noise,
            latent_image,
            sampler_,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=None,
        ):
            del latent_image, seed, latent_shapes, denoise_mask
            return sampler_sample_wrapper(
                SamplerExecutor(),
                guider,
                sigmas,
                {"model_options": {"transformer_options": {}}, "seed": 0},
                callback,
                noise,
                None,
                denoise_mask_,
                disable_pbar,
            )

    denoise_mask_ = denoise_mask
    monkeypatch.setattr(sampling_module, "_res4lyf_preflight_reason", lambda *_args: None)
    result = outer_sample_wrapper(
        OuterExecutor(),
        _initial_noise(),
        torch.zeros(_SHAPE),
        sampler,
        _SIGMAS.clone(),
        None,
        None,
        True,
        0,
    )
    record["runtime"] = runtime
    return result, record


def _config(**overrides):
    from comfyui_spectrum_h3.config import SpectrumH3Config

    values = dict(
        warmup_steps=2,
        tail_actual_steps=1,
        model_aware_mode="off",
        offline_smoothing_replay=False,
        bootstrap_first_forecast=False,
    )
    values.update(overrides)
    return SpectrumH3Config(**values)


@pytest.mark.parametrize("name", RES4LYF_SDE_WRAPPERS)
@pytest.mark.parametrize("av", (False, True))
def test_tracked_all_actual_res4lyf_sde_is_bitwise_native(name, av, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    from comfyui_spectrum_h3.sampling import (
        _res4lyf_effective_sigmas,
        _res4lyf_expected_model_calls,
    )

    original = rk_sampler_beta.RK_NoiseSampler
    denoise = _denoiser(*_manifold())
    native = _native_res4lyf(beta, name, denoise, av=av)
    av_shifts = []

    def observe_av(_index, _actual, bridge):
        noise_sampler = bridge._bound_noise_sampler
        av_shifts.append((noise_sampler.av_split, noise_sampler.av_shift_audio))

    tracked, record = _spectrum_res4lyf(
        beta,
        name,
        denoise,
        _config(warmup_steps=10_000),
        monkeypatch,
        av=av,
        inside_call=observe_av,
    )

    assert torch.equal(tracked, native)
    assert set(av_shifts) == ({(8, 8.0)} if av else {(None, None)})
    if av:
        # Prove the AV branch affects this fixture's numerical trajectory.
        assert not torch.equal(native, _native_res4lyf(beta, name, denoise))
    assert rk_sampler_beta.RK_NoiseSampler is original
    assert set(record["modes"]) == {"actual"}
    effective, reason = _res4lyf_effective_sigmas(_SIGMAS, _flow_model_sampling())
    assert reason is None
    sampler = SimpleNamespace(sampler_function=getattr(beta, f"sample_{name}"))
    assert len(record["modes"]) == _res4lyf_expected_model_calls(sampler, effective)
    bridge = record["bridge"]
    assert isinstance(bridge, RES4LYFStochasticBridge)
    assert bridge.invalid_reason is None
    # Every model call after the first receives exactly one freshly noised
    # state; the final landing at sigma_min is consumed by RES4LYF's terminal
    # denoise rather than by another H3 call.
    assert bridge.transitions_consumed == len(record["modes"]) - 1
    assert bridge.transitions_recorded == len(record["modes"])
    assert record["runtime"].active_run_id is None


@pytest.mark.parametrize("name", RES4LYF_SDE_WRAPPERS)
def test_res4lyf_sde_forecasts_use_noise_free_solver_space_dense_output(name, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    mu, basis = _manifold()
    denoise = _denoiser(mu, basis)

    result, record = _spectrum_res4lyf(beta, name, denoise, _config(), monkeypatch)

    modes = Counter(record["modes"])
    assert modes["forecast"] > 0
    assert torch.isfinite(result).all()
    assert record["bridge"].invalid_reason is None
    assert record["runtime"].stats.forecast_fallbacks == 0
    assert not record["runtime"].stats.disabled
    assert rk_sampler_beta.RK_NoiseSampler.__name__ == "RK_NoiseSampler"

    def off_manifold(value):
        flat = (value - mu).reshape(1, -1)
        return float((flat - (flat @ basis) @ basis.T).pow(2).mean())

    for step_id, raw in record["raw"].items():
        returned = record["returned"][step_id]
        assert not torch.equal(returned, raw)
        # The exact denoiser output lies on the data manifold. Dense output is a
        # bounded combination of exact anchors, so it carries none of the fresh
        # off-manifold noise that the noise-blind raw forecast leaks.
        assert off_manifold(returned) < 1e-10
        assert off_manifold(raw) > 1e-4


def test_modified_noised_state_forces_exact_call_and_disables_forecasts(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    denoise = _denoiser(*_manifold())
    _, baseline = _spectrum_res4lyf(beta, "res_2s", denoise, _config(), monkeypatch)
    first_forecast = baseline["modes"].index("forecast")

    result, record = _spectrum_res4lyf(
        beta,
        "res_2s",
        denoise,
        _config(),
        monkeypatch,
        perturb_call=first_forecast,
    )

    assert torch.isfinite(result).all()
    assert record["modes"][first_forecast] == "actual"
    assert "forecast" not in record["modes"][first_forecast:]
    assert "was modified" in (record["bridge"].invalid_reason or "")
    assert record["runtime"].stats.disabled
    assert rk_sampler_beta.RK_NoiseSampler.__name__ == "RK_NoiseSampler"


def test_denoise_mask_keeps_res4lyf_sde_all_actual(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    original = rk_sampler_beta.RK_NoiseSampler
    denoise = _denoiser(*_manifold())

    class _MaskedModelK(_ModelK):
        def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None):
            return self.inner_model(x, sigma, model_options=model_options or {}, seed=seed)

    monkeypatch.setattr(sys.modules[__name__], "_ModelK", _MaskedModelK)
    _, record = _spectrum_res4lyf(
        beta,
        "res_2m",
        denoise,
        _config(),
        monkeypatch,
        denoise_mask=torch.ones(_SHAPE),
    )

    assert set(record["modes"]) == {"actual"}
    assert record.get("bridge") is None
    assert record["runtime"].stats.disabled
    assert "denoise mask" in (record["runtime"].stats.disable_reason or "")
    assert rk_sampler_beta.RK_NoiseSampler is original


def test_dense_output_failure_retries_the_same_step_as_exact():
    from comfyui_spectrum_h3.config import SpectrumH3Config
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.sampling import (
        BINDING_KEY,
        RUN_ID_KEY,
        STEP_ID_KEY,
        SpectrumH3Binding,
        predict_noise_wrapper,
    )

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=0,
            max_history=4,
            window_size=3.0,
            offline_smoothing_replay=False,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([0.9, 0.8, 0.6, 0.0]),
        "sample_res_2m",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    # The bridge has no exact anchor, so dense output is impossible for the
    # forecast below even though the runtime schedule selects one.
    bridge = RES4LYFStochasticBridge(run_id=run_id, coordinate_fn=_coordinate, debug=False)
    assert bridge.bind_noise_sampler(object())
    guider = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(runtime)})
    attempts = []
    exact = torch.tensor([[7.0, 8.0]])

    class Executor:
        class_obj = guider

        def __call__(self, _x, _timestep, model_options, _seed):
            options = model_options["transformer_options"]
            call_id, actual = runtime.begin_model_call(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                labels=((0, "positive"),),
                expected_shape=(1, 2, 1),
            )
            attempts.append(actual)
            if actual:
                runtime.observe_actual(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    torch.full((1, 2, 1), float(len(attempts))),
                )
            else:
                assert runtime.predict(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ) is not None
            return exact

    options = {"transformer_options": {RES4LYF_STOCHASTIC_BRIDGE_KEY: bridge}}
    for sigma in (0.9, 0.8):
        predict_noise_wrapper(Executor(), torch.zeros(1, 2), torch.tensor([sigma]), options, 0)
        # Drop the warm-up anchors so the forecast has no exact anchor to use.
        bridge._anchors.clear()
        _noop_swap(bridge)
    attempts.clear()

    result = predict_noise_wrapper(Executor(), torch.zeros(1, 2), torch.tensor([0.6]), options, 0)

    assert attempts == [False, True]
    assert result is exact
    assert runtime.last_completed_mode == "actual"
    assert runtime.stats.forecast_fallbacks == 1
    assert bridge.anchor_steps == (2,)
    runtime.end_run(run_id)


@pytest.mark.parametrize("name", ("res_2m", "res_3s_ode"))
def test_res4lyf_contract_accepts_native_api_and_rejects_split_phi(name, monkeypatch):
    beta, _rk_sampler_beta = _res4lyf_runtime_fixture()
    import comfy.samplers

    from comfyui_spectrum_h3 import sampling as sampling_module

    sampler = comfy.samplers.KSAMPLER(getattr(beta, f"sample_{name}"))

    assert sampling_module._res4lyf_sampler_contract(sampler) == (True, None)

    method = inspect.getmodule(_rk_sampler_beta.RK_Method_Beta)
    monkeypatch.setattr(method, "Phi", object())
    supported, reason = sampling_module._res4lyf_sampler_contract(sampler)

    assert not supported
    assert "Phi implementation provenance" in reason


def test_tracking_failure_after_association_retries_the_forecast_exactly(monkeypatch):
    beta, _rk_sampler_beta = _res4lyf_runtime_fixture()
    denoise = _denoiser(*_manifold())
    _, baseline = _spectrum_res4lyf(beta, "res_2m", denoise, _config(), monkeypatch)
    target = baseline["modes"].index("forecast")
    planned = []

    def invalidate_inside_forecast(index, actual, bridge):
        if index == target and not actual:
            planned.append(index)
            # Tracking fails after this call was associated as a forecast.
            bridge.invalidate("probe: tracking failed after association")

    _, record = _spectrum_res4lyf(
        beta,
        "res_2m",
        denoise,
        _config(),
        monkeypatch,
        inside_call=invalidate_inside_forecast,
    )

    assert planned == [target]
    # The same call was retried exactly before RES4LYF received a result.
    assert record["modes"][target] == "actual"
    assert torch.equal(record["returned"][target], record["exact"][target])
    assert "forecast" not in record["modes"][target:]
    assert record["runtime"].stats.disabled
    assert record["runtime"].stats.forecast_fallbacks == 1


@pytest.mark.parametrize(
    ("placement", "expected"),
    (
        ("foreign_module", "does not execute in its audited module"),
        ("audited_globals", "is not the reviewed native implementation"),
    ),
)
def test_replaced_noise_sampler_method_keeps_the_run_all_actual(
    placement,
    expected,
    monkeypatch,
):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    import inspect

    import comfy.samplers

    from comfyui_spectrum_h3 import sampling as sampling_module

    native_class = rk_sampler_beta.RK_NoiseSampler
    native_swap = native_class.swap_noise_step

    def shifted_swap(self, *args, **kwargs):
        return native_swap(self, *args, **kwargs) + 0.01

    if placement == "audited_globals":
        # Same globals as the audited module and no closure; only the code differs.
        namespace: dict[str, object] = {}
        exec(  # noqa: S102 - builds a deliberately altered test method
            "def swap_noise_step(self, x_0, x_next, brownian_sigma=None,\n"
            "                    brownian_sigma_next=None, mask=None):\n"
            "    return x_next + 0.01\n",
            namespace,
        )
        shifted_swap = types.FunctionType(
            namespace["swap_noise_step"].__code__,
            vars(inspect.getmodule(native_class)),
            "swap_noise_step",
            namespace["swap_noise_step"].__defaults__,
        )
    monkeypatch.setattr(native_class, "swap_noise_step", shifted_swap)
    sampler = comfy.samplers.KSAMPLER(beta.sample_res_2m)

    supported, reason = sampling_module._res4lyf_sampler_contract(sampler)
    assert not supported
    assert f"RK_NoiseSampler.swap_noise_step {expected}" in reason

    denoise = _denoiser(*_manifold())
    _, record = _spectrum_res4lyf(beta, "res_2m", denoise, _config(), monkeypatch)

    assert set(record["modes"]) == {"actual"}
    assert record.get("bridge") is None
    assert "swap_noise_step" in (record["runtime"].stats.disable_reason or "")
    assert rk_sampler_beta.RK_NoiseSampler is native_class


def test_method_replaced_during_the_run_fails_closed(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    denoise = _denoiser(*_manifold())
    _, baseline = _spectrum_res4lyf(beta, "res_2s", denoise, _config(), monkeypatch)
    replace_at = baseline["modes"].index("forecast")
    native_class = rk_sampler_beta.RK_NoiseSampler
    native_substep = native_class.swap_noise_substep

    def replacement(self, *args, **kwargs):
        return native_substep(self, *args, **kwargs)

    def replace(index):
        if index == replace_at:
            monkeypatch.setattr(native_class, "swap_noise_substep", replacement)

    _, record = _spectrum_res4lyf(
        beta,
        "res_2s",
        denoise,
        _config(),
        monkeypatch,
        before_call=replace,
    )

    assert "forecast" not in record["modes"][replace_at:]
    assert "methods changed" in (record["bridge"].invalid_reason or "")
    assert record["runtime"].stats.disabled


def test_foreign_replacement_during_the_run_is_preserved(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    native_class = rk_sampler_beta.RK_NoiseSampler
    monkeypatch.setattr(rk_sampler_beta, "RK_NoiseSampler", native_class)
    foreign = type("ForeignNoiseSampler", (native_class,), {})
    overlap = []

    def interfere(index):
        if index == 3:
            other = RES4LYFStochasticBridge(
                run_id=99,
                coordinate_fn=_coordinate,
                debug=False,
            )
            with (
                pytest.raises(RES4LYFStochasticError, match="already replaced"),
                tracked_res4lyf_noise_sampler(
                    rk_sampler_beta,
                    native_class,
                    other,
                    object(),
                ),
            ):
                pass
            overlap.append(rk_sampler_beta.RK_NoiseSampler)
            rk_sampler_beta.RK_NoiseSampler = foreign

    denoise = _denoiser(*_manifold())
    _spectrum_res4lyf(
        beta,
        "res_2m",
        denoise,
        _config(),
        monkeypatch,
        before_call=interfere,
    )

    # The overlapping install was refused while Spectrum's class was installed,
    # and cleanup did not erase the replacement made by the other component.
    assert len(overlap) == 1
    assert overlap[0] is not native_class
    assert issubclass(overlap[0], native_class)
    assert rk_sampler_beta.RK_NoiseSampler is foreign


@pytest.mark.parametrize("inference", (False, True))
def test_in_place_edit_of_the_same_input_is_not_an_unchanged_retry(inference):
    context = torch.inference_mode() if inference else contextlib.nullcontext()
    with context:
        bridge = _bridge()
        landing = torch.zeros(1, 2)
        noised = torch.tensor([[0.5, -0.5]])
        _swap(bridge, "step", landing, noised, 0.8)
        x = noised.clone()
        assert bridge.associate(x, torch.tensor([0.8]), run_id=1, step_id=3) == "stochastic_step"
        x.add_(0.01)

        with pytest.raises(RES4LYFStochasticError, match="associated again"):
            bridge.associate(x, torch.tensor([0.8]), run_id=1, step_id=3)
    assert bridge.invalid_reason is not None


@pytest.mark.parametrize("mode", ("actual", "forecast"))
def test_result_for_an_input_edited_after_association_fails_closed(mode):
    bridge = _bridge()
    _actual(bridge, torch.ones(1, 2), 0.9, 0)
    _noop_swap(bridge)
    x = torch.zeros(1, 2)
    bridge.associate(x, torch.tensor([0.8]), run_id=1, step_id=1)
    x.add_(0.01)
    result = torch.full((1, 2), 3.0)
    descriptor = RES4LYFStepDescriptor(1, 1, mode)

    if mode == "forecast":
        with pytest.raises(RES4LYFStochasticError, match="changed between association"):
            bridge.consume(result, torch.tensor([0.8]), descriptor, model_input=x)
    else:
        assert bridge.consume(result, torch.tensor([0.8]), descriptor, model_input=x) is result
    assert "changed between association" in (bridge.invalid_reason or "")


def test_retry_after_input_edit_during_forecast_disables_forecasting():
    from comfyui_spectrum_h3.config import SpectrumH3Config
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.sampling import (
        BINDING_KEY,
        RUN_ID_KEY,
        STEP_ID_KEY,
        SpectrumH3Binding,
        predict_noise_wrapper,
    )

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=0,
            max_history=4,
            window_size=3.0,
            offline_smoothing_replay=False,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([0.9, 0.8, 0.6, 0.0]),
        "sample_res_2m",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    bridge = RES4LYFStochasticBridge(run_id=run_id, coordinate_fn=_coordinate, debug=False)
    assert bridge.bind_noise_sampler(object())
    guider = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(runtime)})
    attempts = []

    class Executor:
        class_obj = guider

        def __call__(self, x, _timestep, model_options, _seed):
            options = model_options["transformer_options"]
            call_id, actual = runtime.begin_model_call(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                labels=((0, "positive"),),
                expected_shape=(1, 2, 1),
            )
            attempts.append(actual)
            if actual:
                runtime.observe_actual(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    torch.full((1, 2, 1), float(len(attempts))),
                )
            else:
                assert runtime.predict(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ) is not None
                # A misbehaving layer edits the solver state in place, and the
                # missing anchors force the executor retry below.
                x.add_(0.01)
                bridge._anchors.clear()
            return torch.full((1, 2), 7.0)

    options = {"transformer_options": {RES4LYF_STOCHASTIC_BRIDGE_KEY: bridge}}
    for sigma in (0.9, 0.8):
        predict_noise_wrapper(Executor(), torch.zeros(1, 2), torch.tensor([sigma]), options, 0)
        _noop_swap(bridge)
    attempts.clear()

    predict_noise_wrapper(Executor(), torch.zeros(1, 2), torch.tensor([0.6]), options, 0)

    # The retry ran without re-association; its result check caught the edit.
    assert attempts == [False, True]
    assert "changed between association" in (bridge.invalid_reason or "")
    assert runtime.stats.disabled
    assert runtime.last_completed_mode == "actual"
    runtime.end_run(run_id)


def _altered_constant_code(code, old, new):
    return code.replace(
        co_consts=tuple(new if value == old and type(value) is float else value for value in code.co_consts)
    )


@pytest.mark.parametrize("mutation", ("code", "defaults"))
def test_in_place_function_mutation_during_the_run_fails_closed(mutation, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    denoise = _denoiser(*_manifold())
    _, baseline = _spectrum_res4lyf(beta, "res_2s", denoise, _config(), monkeypatch)
    target = baseline["modes"].index("forecast")
    native_class = rk_sampler_beta.RK_NoiseSampler
    substep = native_class.swap_noise_substep
    assert 0.999 in substep.__code__.co_consts

    def mutate(index, actual, _bridge):
        if index == target and not actual:
            # The function object stays the same; only its implementation changes.
            if mutation == "code":
                monkeypatch.setattr(
                    substep,
                    "__code__",
                    _altered_constant_code(substep.__code__, 0.999, 0.02),
                )
            else:
                monkeypatch.setattr(substep, "__defaults__", (torch.tensor(0.7), None, None, None))

    _, record = _spectrum_res4lyf(
        beta,
        "res_2s",
        denoise,
        _config(),
        monkeypatch,
        inside_call=mutate,
    )

    assert record["modes"][target] == "actual"
    assert torch.equal(record["returned"][target], record["exact"][target])
    assert "forecast" not in record["modes"][target:]
    assert "methods changed" in (record["bridge"].invalid_reason or "")
    assert record["runtime"].stats.disabled
    assert native_class.swap_noise_substep is substep


def test_replaced_noise_sampler_defaults_keep_the_run_all_actual(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    import comfy.samplers

    from comfyui_spectrum_h3 import sampling as sampling_module

    swap = rk_sampler_beta.RK_NoiseSampler.swap_noise_step
    monkeypatch.setattr(swap, "__defaults__", (torch.tensor(0.7), None, None))
    sampler = comfy.samplers.KSAMPLER(beta.sample_res_2m)

    _module, cls, reason = sampling_module._res4lyf_reviewed_noise_sampler(sampler)
    assert cls is None
    assert "swap_noise_step defaults are not the reviewed native defaults" in reason
    supported, contract_reason = sampling_module._res4lyf_sampler_contract(sampler)
    assert not supported
    assert "defaults" in contract_reason

    denoise = _denoiser(*_manifold())
    _, record = _spectrum_res4lyf(beta, "res_2m", denoise, _config(), monkeypatch)
    assert set(record["modes"]) == {"actual"}
    assert record.get("bridge") is None


def test_source_defaults_audit_compares_literals_names_and_mutability(tmp_path):
    from pathlib import Path

    from comfyui_spectrum_h3 import source_code_audit

    source = tmp_path / "audited.py"
    source.write_text(
        "import torch\n"
        "MUTABLE = []\n"
        "class Audited:\n"
        "    def method(self, a=1, b=-0.5, c='x', d=None, *, e=torch.float64):\n"
        "        return a\n"
        "    def mutable(self, value=MUTABLE):\n"
        "        return value\n"
    )
    namespace: dict[str, object] = {"__name__": "audited"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)  # noqa: S102
    audited = namespace["Audited"]
    method = audited.method

    def matches(function, name):
        return source_code_audit.matches_source_defaults(
            function,
            Path(source),
            ("Audited", name),
            immutable_types=(torch.dtype,),
        )

    assert matches(method, "method")
    for defaults in ((True, -0.5, "x", None), (1, -0.5, "y", None), (1, -0.5, "x")):
        method.__defaults__ = defaults
        assert not matches(method, "method")
    method.__defaults__ = (1, -0.5, "x", None)
    method.__kwdefaults__ = {"e": torch.float32}
    assert not matches(method, "method")
    method.__kwdefaults__ = {"e": torch.float64}
    assert matches(method, "method")
    # A name resolving to a mutable object cannot be frozen by identity.
    assert not matches(audited.mutable, "mutable")


def test_default_that_changes_native_output_is_rejected(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    from comfyui_spectrum_h3 import sampling as sampling_module

    denoise = _denoiser(*_manifold())
    before = _native_res4lyf(beta, "res_2m", denoise)
    get_sde_step = rk_sampler_beta.RK_NoiseSampler.get_sde_step
    monkeypatch.setattr(
        get_sde_step,
        "__defaults__",
        (*get_sde_step.__defaults__[:-1], False),
    )
    after = _native_res4lyf(beta, "res_2m", denoise)

    # The altered default changes the untouched sampler's numerical trajectory.
    assert not torch.equal(before, after)
    sampler = SimpleNamespace(sampler_function=beta.sample_res_2m)
    _module, cls, reason = sampling_module._res4lyf_reviewed_noise_sampler(sampler)
    assert cls is None
    assert "get_sde_step defaults are not the reviewed native defaults" in reason


def _native_av_sampler(rk_sampler_beta):
    native = object.__new__(rk_sampler_beta.RK_NoiseSampler)
    native.av_split, native.av_total = 2, 4
    native.av_audio_noise_scale = 1.0
    native.av_shift_video, native.av_shift_audio = 12.0, 8.0
    return native


@pytest.mark.parametrize(
    "change",
    ("unwrapped_staticmethod", "noncallable", "removed", "kind_flip", "extra_callable", "extra_descriptor"),
)
def test_live_callable_inventory_and_binding_kinds_are_audited(change, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    from comfyui_spectrum_h3 import sampling as sampling_module

    cls = rk_sampler_beta.RK_NoiseSampler
    native = _native_av_sampler(rk_sampler_beta)
    native.scale_av_noise(torch.ones(1, 4), 0.8, 0.6)
    original = vars(cls)["_av_renoise_var"]
    assert isinstance(original, staticmethod)

    if change == "unwrapped_staticmethod":
        # Code, globals and defaults are unchanged; only the binding kind differs.
        monkeypatch.setattr(cls, "_av_renoise_var", original.__func__)
        expected = "_av_renoise_var is missing or not the reviewed staticmethod"
    elif change == "noncallable":
        monkeypatch.setattr(cls, "_av_renoise_var", None)
        expected = "_av_renoise_var is missing or not the reviewed staticmethod"
    elif change == "removed":
        monkeypatch.delattr(cls, "linear_noise_init")
        expected = "linear_noise_init is missing or not the reviewed function"
    elif change == "kind_flip":
        monkeypatch.setattr(cls, "get_sde_coeff", staticmethod(vars(cls)["get_sde_coeff"]))
        expected = "get_sde_coeff is missing or not the reviewed function"
    elif change == "extra_callable":
        monkeypatch.setattr(cls, "extra_helper", lambda self: None, raising=False)
        expected = "extra_helper is an unreviewed callable attribute"
    else:
        class ForeignDescriptor:
            def __get__(self, instance, owner=None):
                return lambda: None

        monkeypatch.setattr(cls, "extra_helper", ForeignDescriptor(), raising=False)
        expected = "extra_helper is an unreviewed callable attribute"

    if change in {"unwrapped_staticmethod", "noncallable"}:
        # The native AV noise-scaling branch now breaks.
        with pytest.raises(TypeError):
            native.scale_av_noise(torch.ones(1, 4), 0.8, 0.6)

    sampler = SimpleNamespace(sampler_function=beta.sample_res_2m)
    _module, reviewed, reason = sampling_module._res4lyf_reviewed_noise_sampler(sampler)
    assert reviewed is None
    assert expected in reason


@pytest.mark.parametrize("change", ("unwrapped_staticmethod", "noncallable", "extra_descriptor"))
def test_descriptor_change_during_the_run_fails_closed(change, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    denoise = _denoiser(*_manifold())
    _, baseline = _spectrum_res4lyf(beta, "res_2s", denoise, _config(), monkeypatch)
    target = baseline["modes"].index("forecast")
    cls = rk_sampler_beta.RK_NoiseSampler
    original = vars(cls)["_av_renoise_var"]

    def mutate(index, actual, _bridge):
        if index == target and not actual:
            if change == "extra_descriptor":
                class ForeignDescriptor:
                    def __get__(self, instance, owner=None):
                        return lambda: None

                monkeypatch.setattr(cls, "extra_helper", ForeignDescriptor(), raising=False)
                return
            replacement = original.__func__ if change == "unwrapped_staticmethod" else None
            monkeypatch.setattr(cls, "_av_renoise_var", replacement)

    _, record = _spectrum_res4lyf(
        beta,
        "res_2s",
        denoise,
        _config(),
        monkeypatch,
        inside_call=mutate,
    )

    assert record["modes"][target] == "actual"
    assert torch.equal(record["returned"][target], record["exact"][target])
    assert "forecast" not in record["modes"][target:]
    assert "methods changed" in (record["bridge"].invalid_reason or "")
    assert record["runtime"].stats.disabled


def test_foreign_descriptor_subclass_is_not_a_reviewed_binding(monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    from comfyui_spectrum_h3 import sampling as sampling_module

    cls = rk_sampler_beta.RK_NoiseSampler
    native = _native_av_sampler(rk_sampler_beta)
    before = native.scale_av_noise(torch.ones(1, 4), 0.8, 0.6)
    original = vars(cls)["_av_renoise_var"]

    class ForeignStaticMethod(staticmethod):
        def __get__(self, instance, owner=None):
            return lambda _sigma_from, _sigma_to: 0.0

    # The underlying function still has the reviewed code and defaults, but
    # this descriptor substitutes another callable on instance access.
    monkeypatch.setattr(cls, "_av_renoise_var", ForeignStaticMethod(original.__func__))
    after = native.scale_av_noise(torch.ones(1, 4), 0.8, 0.6)
    assert not torch.equal(before, after)
    sampler = SimpleNamespace(sampler_function=beta.sample_res_2m)
    assert sampling_module._res4lyf_reviewed_noise_sampler(sampler)[2] is not None


def test_input_mutation_on_an_exact_call_disables_forecasting_immediately():
    from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
    from comfyui_spectrum_h3.sampling import res4lyf_consume_model_result

    runtime = SpectrumH3Runtime(_config())
    run_id = runtime.start_run(
        torch.tensor([0.9, 0.8, 0.6, 0.0]),
        "sample_res_2m",
        supported_sampler=True,
    )
    decision = runtime.begin_step(torch.tensor([0.9]))
    assert decision["actual"]
    bridge = _bridge(run_id)
    x = torch.zeros(1, 2)
    bridge.associate(x, torch.tensor([0.9]), run_id=run_id, step_id=decision["step_id"])
    x.add_(0.01)
    result = torch.ones(1, 2)
    options = {"transformer_options": {RES4LYF_STOCHASTIC_BRIDGE_KEY: bridge}}

    returned = res4lyf_consume_model_result(
        runtime, options, x, torch.tensor([0.9]), result, decision
    )

    assert returned is result
    assert "changed between association" in bridge.invalid_reason
    assert runtime.stats.disabled
    assert "changed between association" in runtime.stats.disable_reason
    runtime.end_run(run_id)


@pytest.mark.parametrize("name", RES4LYF_SDE_WRAPPERS)
def test_current_res4lyf_packed_av_noise_is_native_with_real_forecasts(name, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    monkeypatch.setattr(sys.modules[__name__], "_SHAPE", (1, 1, 16))
    denoise = _denoiser(*_manifold())
    native = _native_res4lyf(beta, name, denoise, av=True)
    generator_types = []

    def observe_generator(_index, _actual, bridge):
        generator = bridge._bound_noise_sampler.noise_sampler
        generator_types.append(type(generator._base).__name__)

    actual, actual_record = _spectrum_res4lyf(
        beta, name, denoise, _config(warmup_steps=10_000, debug=True), monkeypatch,
        av=True, inside_call=observe_generator,
    )
    assert torch.equal(actual, native)
    assert actual_record["bridge"].invalid_reason is None
    if hasattr(rk_sampler_beta.RK_NoiseSampler, "_build_noise_sampler"):
        assert set(generator_types) == {"PackedNoiseGenerator"}

    _, forecast_record = _spectrum_res4lyf(
        beta, name, denoise, _config(debug=True), monkeypatch, av=True,
    )
    assert "forecast" in forecast_record["modes"]
    assert forecast_record["runtime"].stats.forecast_model_calls > 0
    assert forecast_record["bridge"].invalid_reason is None


@pytest.mark.parametrize("name", ("res_2m", "res_3s_ode"))
def test_unrelated_source_edit_does_not_disable_res4lyf(name, tmp_path, monkeypatch):
    beta, rk_sampler_beta = _res4lyf_runtime_fixture()
    import comfy.samplers

    from comfyui_spectrum_h3 import sampling as sampling_module

    sampler = comfy.samplers.KSAMPLER(getattr(beta, f"sample_{name}"))
    assert sampling_module._res4lyf_sampler_contract(sampler) == (True, None)
    for module in (beta, rk_sampler_beta, inspect.getmodule(rk_sampler_beta.RK_NoiseSampler)):
        source = Path(module.__file__)
        changed = tmp_path / (module.__name__.replace(".", "_") + ".py")
        changed.write_bytes(source.read_bytes() + b"\n# unrelated upstream documentation change\n")
        monkeypatch.setattr(module, "__file__", str(changed))
    assert sampling_module._res4lyf_sampler_contract(sampler) == (True, None)
