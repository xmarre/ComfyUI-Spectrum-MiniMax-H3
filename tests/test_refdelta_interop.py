from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.er_sde_stochastic import (
    ERSDEStepDescriptor,
    ERSDEStochasticTracker,
)
from comfyui_spectrum_h3.refdelta_interop import (
    REFDELTA_BACKEND_INTEROP_CONTRACT,
    REFDELTA_INTEROP_CONTRACT,
    RefDeltaBackendInteropBridge,
    RefDeltaInteropBridge,
    RefDeltaInteropError,
)
from comfyui_spectrum_h3.sampling import (
    _refdelta_backend_sampler_contract,
    _refdelta_sampler_contract,
    _sa_solver_sampler_contract,
    _seeds_sampler_contract,
)


def _installed_refdelta():
    try:
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde
    except ImportError:
        pytest.skip("reviewed RefDelta package is not on PYTHONPATH")
    return RefDeltaSamplerConfig, sample_refdelta_er_sde


def test_installed_refdelta_api_v1_is_admitted_for_exact_increment_ownership():
    config_type, function = _installed_refdelta()

    accepted, reason, external_increment = _refdelta_sampler_contract(
        function,
        {"config": config_type(), "s_noise": 1.0, "max_stage": 3},
    )

    assert accepted, reason
    assert external_increment


def test_refdelta_native_equivalence_uses_native_increment_tracking():
    config_type, function = _installed_refdelta()
    config = config_type(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )

    accepted, reason, external_increment = _refdelta_sampler_contract(
        function,
        {"config": config, "s_noise": 1.0, "max_stage": 3},
    )

    assert accepted, reason
    assert not external_increment


def test_comfyui_nested_refdelta_namespace_is_resolved_without_top_level_package():
    nested_package = "custom_nodes.synthetic_refdelta.comfyui_refdelta_solver"
    canonical_names = (
        "comfyui_refdelta_solver",
        "comfyui_refdelta_solver.config",
        "comfyui_refdelta_solver.sampler",
        "comfyui_refdelta_solver.spectrum_interop",
    )
    nested_names = (
        nested_package,
        f"{nested_package}.config",
        f"{nested_package}.sampler",
        f"{nested_package}.spectrum_interop",
    )
    previous = {name: sys.modules.get(name) for name in canonical_names}

    package_module = ModuleType(nested_package)
    package_module.__path__ = []
    config_module = ModuleType(nested_names[1])
    sampler_module = ModuleType(nested_names[2])
    interop_module = ModuleType(nested_names[3])

    class RefDeltaSamplerConfig:
        is_native_equivalence_mode = False

        def validate(self):
            return None

    def sample_refdelta_er_sde():
        return None

    sample_refdelta_er_sde.__spectrum_interop_contract__ = REFDELTA_INTEROP_CONTRACT
    config_module.RefDeltaSamplerConfig = RefDeltaSamplerConfig
    sampler_module.sample_refdelta_er_sde = sample_refdelta_er_sde
    interop_module.SPECTRUM_INTEROP_CONTRACT = REFDELTA_INTEROP_CONTRACT

    try:
        for name in canonical_names:
            sys.modules.pop(name, None)
        sys.modules[nested_names[0]] = package_module
        sys.modules[nested_names[1]] = config_module
        sys.modules[nested_names[2]] = sampler_module
        sys.modules[nested_names[3]] = interop_module

        config = RefDeltaSamplerConfig()
        accepted, reason, external_increment = _refdelta_sampler_contract(
            sample_refdelta_er_sde,
            {"config": config, "s_noise": 1.0, "max_stage": 3},
        )

        assert accepted, reason
        assert external_increment
        assert sys.modules["comfyui_refdelta_solver.config"].RefDeltaSamplerConfig is RefDeltaSamplerConfig
        assert (
            sys.modules["comfyui_refdelta_solver.sampler"].sample_refdelta_er_sde
            is sample_refdelta_er_sde
        )
    finally:
        for name in canonical_names:
            sys.modules.pop(name, None)
        for name in nested_names:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module


def test_bridge_and_tracker_transfer_exact_gated_increment_end_to_end():
    noise = torch.tensor([[1.0, -2.0]])
    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: noise,
        noise_scaler=lambda value: value**2,
        effective_s_noise=1.0,
        max_stage=3,
        debug=False,
        run_id=19,
        external_increment=True,
    )
    bridge = RefDeltaInteropBridge(run_id=19, tracker=tracker)
    actual = ERSDEStepDescriptor(19, 0, "actual", None, False)
    forecast = ERSDEStepDescriptor(19, 1, "forecast", None, True)

    bridge.note_model_result(actual)
    assert bridge.model_result_is_actual(0)
    tracker.noise_scaler(torch.tensor(0.5))
    tracker.noise_scaler(torch.tensor(1.0))
    tracker.noise_sampler(torch.tensor(0.8), torch.tensor(0.4))
    gated_increment = torch.tensor([[0.15, -0.3]])
    bridge.publish_stochastic_increment(0, gated_increment)

    raw = torch.tensor([[3.0, 4.0]]) + gated_increment
    corrected = tracker.consume(raw, forecast)
    bridge.note_model_result(forecast)

    torch.testing.assert_close(corrected, torch.tensor([[3.0, 4.0]]), rtol=0, atol=0)
    assert not bridge.model_result_is_actual(1)



@pytest.mark.parametrize(
    "function_name",
    (
        "sample_refdelta_seeds_2",
        "sample_refdelta_seeds_3",
        "sample_refdelta_sa_solver",
    ),
)
def test_installed_refdelta_multi_backend_contract_is_admitted(function_name):
    try:
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver import sampler_backends
    except ImportError:
        pytest.skip("reviewed RefDelta multi-backend package is not on PYTHONPATH")

    function = getattr(sampler_backends, function_name)
    accepted, reason = _refdelta_backend_sampler_contract(
        function_name,
        function,
        {"config": RefDeltaSamplerConfig()},
    )

    assert accepted, reason
    assert (
        getattr(function, "__spectrum_interop_contract__", None)
        == REFDELTA_BACKEND_INTEROP_CONTRACT
    )


def test_refdelta_backend_bridge_classifies_only_the_completed_logical_step():
    runtime = SimpleNamespace(last_completed_step_id=7, last_completed_mode="actual")
    bridge = RefDeltaBackendInteropBridge(runtime)

    assert bridge.model_result_is_actual(7)

    runtime.last_completed_step_id = 8
    runtime.last_completed_mode = "forecast"
    assert not bridge.model_result_is_actual(8)

    with pytest.raises(RefDeltaInteropError, match="wrong Spectrum step"):
        bridge.model_result_is_actual(7)


def test_refdelta_backend_bridge_rejects_unreviewed_runtime_mode():
    runtime = SimpleNamespace(last_completed_step_id=2, last_completed_mode="replay")
    bridge = RefDeltaBackendInteropBridge(runtime)

    with pytest.raises(RefDeltaInteropError, match="unreviewed"):
        bridge.model_result_is_actual(2)



def test_installed_refdelta_ksamplers_satisfy_reviewed_solver_contracts():
    try:
        import comfy.k_diffusion.sampling as native_sampling
        import comfy.samplers
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver.sampler_backends import (
            sample_refdelta_sa_solver,
            sample_refdelta_seeds_2,
            sample_refdelta_seeds_3,
        )
    except ImportError:
        pytest.skip("reviewed ComfyUI and RefDelta packages are required")
    if not all(
        callable(getattr(native_sampling, name, None))
        for name in ("sample_seeds_2", "sample_seeds_3", "sample_sa_solver")
    ):
        pytest.skip("this reviewed ComfyUI lane predates native SEEDS/SA support")

    config = RefDeltaSamplerConfig()
    seeds_2 = comfy.samplers.KSAMPLER(
        sample_refdelta_seeds_2,
        extra_options={
            "config": config,
            "eta": 1.0,
            "s_noise": 1.0,
            "r": 0.5,
            "solver_type": "phi_1",
        },
    )
    seeds_3 = comfy.samplers.KSAMPLER(
        sample_refdelta_seeds_3,
        extra_options={
            "config": config,
            "eta": 1.0,
            "s_noise": 1.0,
            "r_1": 1.0 / 3.0,
            "r_2": 2.0 / 3.0,
        },
    )
    sa = comfy.samplers.KSAMPLER(
        sample_refdelta_sa_solver,
        extra_options={
            "config": config,
            "s_noise": 1.0,
            "predictor_order": 3,
            "corrector_order": 4,
            "simple_order_2": False,
        },
    )

    assert _seeds_sampler_contract(seeds_2) == (True, None)
    assert _seeds_sampler_contract(seeds_3) == (True, None)
    assert _sa_solver_sampler_contract(sa) == (True, None)
