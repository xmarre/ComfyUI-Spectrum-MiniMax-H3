from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3.er_sde_ksampler_contract import validate_ksampler_sample
from comfyui_spectrum_h3.sampling import ER_SDE_KSAMPLER_SAMPLE_DIGEST


def _native_ksampler_sample() -> ast.FunctionDef:
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler contract tests")
    source = (Path(comfyui_path) / "comfy/samplers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    ksampler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KSAMPLER"
    )
    return next(
        node
        for node in ksampler.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample"
    )


def _compile_native_method(tmp_path: Path):
    function = _native_ksampler_sample()
    source_path = tmp_path / "native_ksampler.py"
    source_path.write_text(ast.unparse(ast.fix_missing_locations(function)) + "\n", encoding="utf-8")
    adapter = type("KSamplerX0Inpaint", (), {})
    namespace = {
        "KSamplerX0Inpaint": adapter,
        "detail": lambda *_args: None,
    }
    exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), namespace)  # noqa: S102
    method = namespace["sample"]
    method.__module__ = "comfy.samplers"
    method.__qualname__ = "KSAMPLER.sample"
    return method, adapter


def _compile_tiled_wrapper(
    tmp_path: Path,
    target,
    *,
    name: str,
    pop_key: str = "sigmas",
    altered_return: bool = False,
):
    source_path = tmp_path / f"{name}.py"
    return_expression = (
        "orig_fn(*args, **dict(kwargs))"
        if altered_return
        else "orig_fn(*args, **kwargs)"
    )
    source_path.write_text(
        f'''def KSAMPLER_sample(*args, **kwargs):
    orig_fn = store.KSAMPLER_sample
    extra_args = None
    model_options = None
    try:
        extra_args = kwargs["extra_args"] if "extra_args" in kwargs else args[3]
        model_options = extra_args["model_options"]
    except Exception:
        ...
    if model_options is not None and "tiled_diffusion" in model_options and extra_args is not None:
        sigmas_ = kwargs["sigmas"] if "sigmas" in kwargs else args[2]
        sigmas_all = model_options.pop("{pop_key}", None)
        sigmas = sigmas_all if sigmas_all is not None else sigmas_
        store.sigmas = sigmas
        store.model_options = model_options
        store.extra_args = extra_args
    else:
        for attr in ["sigmas", "model_options", "extra_args"]:
            _delattr(store, attr)
    return {return_expression}
''',
        encoding="utf-8",
    )
    store = SimpleNamespace(KSAMPLER_sample=target)

    def _delattr(owner, attr):
        if hasattr(owner, attr):
            delattr(owner, attr)

    namespace = {"store": store, "_delattr": _delattr, "dict": dict}
    exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), namespace)  # noqa: S102
    wrapper = namespace["KSAMPLER_sample"]
    wrapper.__module__ = "/home/toor/ComfyUI/custom_nodes/tiled-diffusion.utils"
    wrapper.__qualname__ = "KSAMPLER_sample"
    return wrapper, store


def _validate(wrapper, adapter):
    return validate_ksampler_sample(
        wrapper,
        expected_adapter=adapter,
        expected_reference_digest=ER_SDE_KSAMPLER_SAMPLE_DIGEST,
    )


def test_tiled_diffusion_variadic_ksampler_wrapper_is_accepted(tmp_path):
    native, adapter = _compile_native_method(tmp_path)
    wrapper, _store = _compile_tiled_wrapper(tmp_path, native, name="tiled_wrapper")

    result = _validate(wrapper, adapter)

    assert result.accepted, result.failure
    assert result.provenance.qualname == "KSAMPLER_sample"
    assert "tiled-diffusion.utils" in (result.provenance.module or "")


def test_tiled_diffusion_wrapper_chain_is_accepted(tmp_path):
    native, adapter = _compile_native_method(tmp_path)
    inner, _inner_store = _compile_tiled_wrapper(tmp_path, native, name="tiled_inner")
    outer, _outer_store = _compile_tiled_wrapper(tmp_path, inner, name="tiled_outer")

    result = _validate(outer, adapter)

    assert result.accepted, result.failure


def test_tiled_diffusion_wrapper_rejects_changed_sigmas_handoff(tmp_path):
    native, adapter = _compile_native_method(tmp_path)
    wrapper, _store = _compile_tiled_wrapper(
        tmp_path,
        native,
        name="tiled_wrong_pop",
        pop_key="other",
    )

    result = _validate(wrapper, adapter)

    assert not result.accepted
    assert "TiledDiffusion sigmas handoff" in (result.failure or "")


def test_tiled_diffusion_wrapper_rejects_changed_delegate_arguments(tmp_path):
    native, adapter = _compile_native_method(tmp_path)
    wrapper, _store = _compile_tiled_wrapper(
        tmp_path,
        native,
        name="tiled_changed_return",
        altered_return=True,
    )

    result = _validate(wrapper, adapter)

    assert not result.accepted
    assert "does not forward *args/**kwargs unchanged" in (result.failure or "")


def test_tiled_diffusion_wrapper_cycle_fails_closed(tmp_path):
    native, adapter = _compile_native_method(tmp_path)
    wrapper, store = _compile_tiled_wrapper(tmp_path, native, name="tiled_cycle")
    store.KSAMPLER_sample = wrapper

    result = _validate(wrapper, adapter)

    assert not result.accepted
    assert "delegates to itself" in (result.failure or "") or "cycle" in (result.failure or "")
