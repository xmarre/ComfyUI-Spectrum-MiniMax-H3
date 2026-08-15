from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.sampling import (
    ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST,
    ER_SDE_KSAMPLER_SAMPLE_DIGEST,
    ER_SDE_NATIVE_FUNCTION_DIGEST,
)

RES_VARIANTS = (
    "sample_res_multistep",
    "sample_res_multistep_cfg_pp",
)

ANCESTRAL_VARIANTS = (
    "sample_euler_ancestral",
    "sample_euler_ancestral_RF",
    "sample_res_multistep_ancestral",
    "sample_res_multistep_ancestral_cfg_pp",
)


def _native_tree(relative_path: str) -> ast.Module:
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler contract tests")
    source_path = Path(comfyui_path) / relative_path
    return ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def _native_sampling_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in _native_tree("comfy/k_diffusion/sampling.py").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _compile_native_sampling_function(name: str, globals_: dict):
    function = copy.deepcopy(_native_sampling_functions()[name])
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(globals_)
    exec(compile(module, f"<native {name}>", "exec"), namespace)  # noqa: S102
    return namespace[name]


def _named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
    ]


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _ast_digest(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = copy.deepcopy(node)
    normalized.decorator_list = []
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _delegates_to_ancestral_res(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    eta_default = next(
        (default for argument, default in zip(positional, defaults, strict=True) if argument.arg == "eta"),
        None,
    )
    if (
        not isinstance(eta_default, ast.Constant)
        or isinstance(eta_default.value, bool)
        or not isinstance(eta_default.value, (int, float))
        or eta_default.value <= 0
    ):
        return False
    return any(
        any(
            keyword.arg == "eta"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "eta"
            for keyword in call.keywords
        )
        for call in _named_calls(node, "res_multistep")
    )


def test_native_res_multistep_makes_one_model_call_per_solver_iteration():
    function = _native_sampling_functions()["res_multistep"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


def test_native_res_multistep_second_order_update_reuses_current_and_previous_denoised():
    function = _native_sampling_functions()["res_multistep"]
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "x" for target in node.targets)
    ]

    assert any({"denoised", "old_denoised"} <= _loaded_names(node.value) for node in assignments)


def test_native_res_multistep_replaces_old_denoised_after_second_order_update():
    function = _native_sampling_functions()["res_multistep"]
    loop = next(node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor)))
    second_order_updates = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "x" for target in node.targets)
        and "old_denoised" in _loaded_names(node.value)
    ]
    history_updates = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "old_denoised" for target in node.targets)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"denoised", "uncond_denoised"}
    ]

    assert second_order_updates
    assert history_updates
    assert min(node.lineno for node in history_updates) > max(
        node.lineno for node in second_order_updates
    )


def test_native_euler_makes_one_model_call_per_solver_iteration():
    function = _native_sampling_functions()["sample_euler"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


@pytest.mark.parametrize("function_name", RES_VARIANTS)
def test_native_res_variant_delegates_once_to_reviewed_core(function_name):
    function = _native_sampling_functions()[function_name]

    assert len(_named_calls(function, "res_multistep")) == 1
    assert not _named_calls(function, "model")


@pytest.mark.parametrize("function_name", ANCESTRAL_VARIANTS)
def test_native_ancestral_variants_inject_or_delegate_to_noise(function_name):
    function = _native_sampling_functions()[function_name]
    loaded = _loaded_names(function)
    delegates_to_ancestral_core = bool(
        _named_calls(function, "sample_euler_ancestral_RF")
        or _delegates_to_ancestral_res(function)
    )

    assert "noise_sampler" in loaded or delegates_to_ancestral_core

def test_native_er_sde_makes_one_model_call_per_outer_solver_iteration():
    function = _native_sampling_functions()["sample_er_sde"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


def test_native_er_sde_runtime_sources_match_reviewed_compensation_contract():
    functions = _native_sampling_functions()
    assert _ast_digest(functions["sample_er_sde"]) == ER_SDE_NATIVE_FUNCTION_DIGEST
    assert (
        _ast_digest(functions["default_noise_sampler"])
        == ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST
    )

    samplers_tree = _native_tree("comfy/samplers.py")
    ksampler = next(
        node
        for node in samplers_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KSAMPLER"
    )
    sample_method = next(
        node
        for node in ksampler.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample"
    )
    assert _ast_digest(sample_method) == ER_SDE_KSAMPLER_SAMPLE_DIGEST


def test_native_er_sde_stages_reuse_only_solver_local_denoised_history():
    function = _native_sampling_functions()["sample_er_sde"]
    loop = next(node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor)))
    loaded = _loaded_names(loop)
    initial_history = {
        target.id
        for assignment in function.body
        if isinstance(assignment, ast.Assign)
        and isinstance(assignment.value, ast.Constant)
        and assignment.value.value is None
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }

    assert {"old_denoised", "old_denoised_d"} <= initial_history
    assert {"max_stage", "old_denoised", "old_denoised_d"} <= loaded
    assert len(_named_calls(loop, "model")) == 1
    assert len(_named_calls(loop, "min")) == 1

    history_targets = {
        target.id
        for assignment in ast.walk(loop)
        if isinstance(assignment, ast.Assign)
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }
    assert {"old_denoised", "old_denoised_d"} <= history_targets


def test_native_er_sde_noise_and_callback_contract_is_one_per_nonterminal_or_outer_step():
    function = _native_sampling_functions()["sample_er_sde"]
    loop = next(node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor)))

    callback_calls = _named_calls(loop, "callback")
    noise_calls = _named_calls(loop, "noise_sampler")
    assert len(callback_calls) == 1
    assert len(noise_calls) == 1
    assert len(_named_calls(function, "default_noise_sampler")) == 1

    terminal_branch = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == 0
            for comparator in node.test.comparators
        )
        and any(
            isinstance(item, ast.Name) and item.id == "sigmas"
            for item in ast.walk(node.test)
        )
    )
    assert not _named_calls(ast.Module(body=terminal_branch.body, type_ignores=[]), "noise_sampler")
    terminal_updates = [
        assignment
        for assignment in ast.walk(ast.Module(body=terminal_branch.orelse, type_ignores=[]))
        if isinstance(assignment, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "x" for target in assignment.targets)
        and not _named_calls(assignment, "noise_sampler")
    ]
    assert len(
        _named_calls(ast.Module(body=terminal_branch.orelse, type_ignores=[]), "noise_sampler")
    ) == 1
    assert terminal_updates
    assert max(assignment.lineno for assignment in terminal_updates) < noise_calls[0].lineno
    assert callback_calls[0].lineno < noise_calls[0].lineno


def test_native_er_sde_runtime_counts_are_independent_of_max_stage():
    native_er_sde = _compile_native_sampling_function(
        "sample_er_sde",
        {
            "torch": torch,
            "trange": lambda count, disable=None: range(count),
            "default_noise_sampler": object(),
            "offset_first_sigma_for_snr": lambda sigmas, _model_sampling: sigmas,
            "sigma_to_half_log_snr": (
                lambda sigmas, _model_sampling: -torch.log(sigmas.clamp_min(1e-6))
            ),
        },
    )
    model_sampling = SimpleNamespace(noise_scale=1.0)

    class Patcher:
        def get_model_object(self, name):
            assert name == "model_sampling"
            return model_sampling

    def run(max_stage, s_noise=1.0):
        model_calls = []
        callbacks = []
        noise_draws = []

        class Model:
            inner_model = SimpleNamespace(model_patcher=Patcher())

            def __call__(self, x, sigma, **_extra_args):
                model_calls.append(sigma.detach().clone())
                sigma_view = sigma.reshape(-1, *([1] * (x.ndim - 1)))
                return x * 0.25 + sigma_view * 0.05

        def noise_sampler(sigma, sigma_next):
            noise_draws.append((float(sigma), float(sigma_next)))
            return torch.zeros((1, 2), dtype=torch.float32)

        result = native_er_sde(
            Model(),
            torch.ones((1, 2), dtype=torch.float32),
            torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
            extra_args={"seed": 17},
            callback=lambda state: callbacks.append(state["i"]),
            disable=True,
            noise_sampler=noise_sampler,
            noise_scaler=lambda value: value + 10.0,
            max_stage=max_stage,
            s_noise=s_noise,
        )
        return result, model_calls, callbacks, noise_draws

    stage_one = run(1)
    stage_three = run(3)
    noise_disabled = run(3, s_noise=0.0)

    for result, model_calls, callbacks, noise_draws in (stage_one, stage_three):
        assert torch.isfinite(result).all()
        assert len(model_calls) == 4
        assert callbacks == [0, 1, 2, 3]
        assert noise_draws == [(4.0, 3.0), (3.0, 2.0), (2.0, 1.0)]

    assert len(noise_disabled[1]) == 4
    assert noise_disabled[2] == [0, 1, 2, 3]
    assert noise_disabled[3] == []


def test_native_default_noise_sampler_restarts_the_same_seeded_sequence():
    default_noise_sampler = _compile_native_sampling_function(
        "default_noise_sampler",
        {"torch": torch},
    )
    x = torch.zeros((2, 3), dtype=torch.float32)
    first = default_noise_sampler(x, seed=1234)
    replay = default_noise_sampler(x, seed=1234)

    for _ in range(3):
        first_draw = first(torch.tensor(1.0), torch.tensor(0.5))
        replay_draw = replay(torch.tensor(1.0), torch.tensor(0.5))
        torch.testing.assert_close(first_draw, replay_draw)


def test_current_comfyui_registers_er_sde_as_native_ksampler():
    tree = _native_tree("comfy/samplers.py")
    sampler_names = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "KSAMPLER_NAMES" for target in node.targets)
    )

    assert isinstance(sampler_names, ast.List)
    assert "er_sde" in {
        item.value
        for item in sampler_names.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }

    ksampler = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "ksampler"
    )
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "getattr"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "k_diffusion_sampling"
        and isinstance(call.args[1], ast.Call)
        and isinstance(call.args[1].func, ast.Attribute)
        and call.args[1].func.attr == "format"
        and isinstance(call.args[1].func.value, ast.Constant)
        and call.args[1].func.value.value == "sample_{}"
        and "sampler_name" in _loaded_names(call.args[1])
        for call in ast.walk(ksampler)
        if isinstance(call, ast.Call)
    )
