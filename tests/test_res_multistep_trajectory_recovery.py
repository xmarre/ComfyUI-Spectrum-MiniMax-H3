"""RES numerical-trajectory counterexample using the installed native solver.

This synthetic test does not claim MiniMax H3 media quality. It isolates whether
one exact solver evaluation can generally erase a preceding denoised error.
"""

from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _native_res_multistep():
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler tests")
    path = Path(comfyui_path) / "comfy/k_diffusion/sampling.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "res_multistep"
    )
    function = copy.deepcopy(function)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "torch": torch,
        "trange": lambda length, disable=None: range(length),
        # sample_res_multistep delegates to res_multistep with eta=0.
        "get_ancestral_step": lambda sigma, next_sigma, eta: (
            next_sigma,
            torch.zeros_like(next_sigma),
        ),
        "to_d": lambda x, sigma, denoised: (x - denoised) / sigma,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace["res_multistep"]


class _SyntheticDenoiser:
    def __init__(self, error_at_step: int | None):
        self.error_at_step = error_at_step
        self.inputs: list[torch.Tensor] = []
        model_sampling = SimpleNamespace(noise_scale=1.0)
        model_patcher = SimpleNamespace(
            get_model_object=lambda name: model_sampling
        )
        self.inner_model = SimpleNamespace(model_patcher=model_patcher)

    def __call__(self, x, sigma, **kwargs):
        step = len(self.inputs)
        self.inputs.append(x.detach().clone())
        # A deterministic, smooth state-dependent denoiser.
        denoised = 0.3 * x + 0.1 * x.square()
        if step == self.error_at_step:
            denoised = denoised + 0.2
        return denoised


def test_native_res_final_exact_evaluation_does_not_erase_penultimate_error():
    """A refresh clears the retained estimate, not the altered latent state."""
    native = _native_res_multistep()
    sigmas = torch.linspace(1.0, 0.0, 26, dtype=torch.float64)
    initial = torch.tensor([1.0], dtype=torch.float64)

    def run(error_at_step: int | None):
        model = _SyntheticDenoiser(error_at_step)
        output = native(
            model,
            initial.clone(),
            sigmas,
            disable=True,
            eta=0.0,
            cfg_pp=False,
            s_noise=0.0,
            noise_sampler=lambda start, end: torch.zeros_like(initial),
        )
        assert len(model.inputs) == 25
        return model, output

    baseline, baseline_output = run(None)
    _earlier_error, earlier_output = run(22)
    penultimate_error, penultimate_output = run(23)

    # The injected error is the sole difference before the changed RES update.
    assert torch.equal(baseline.inputs[23], penultimate_error.inputs[23])
    assert not torch.allclose(
        baseline.inputs[24], penultimate_error.inputs[24],
        atol=1e-5, rtol=0,
    )

    # Step 24 runs the exact denoiser but cannot recover the reference state.
    terminal_error = (penultimate_output - baseline_output).abs().item()
    earlier_terminal_error = (earlier_output - baseline_output).abs().item()
    assert terminal_error > 1e-3
    # On this specified synthetic trajectory, a late error has much more leverage.
    assert terminal_error > 10 * earlier_terminal_error
