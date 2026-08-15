from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .er_sde_stochastic import function_node_ast_digest


@dataclass(frozen=True, slots=True)
class KSamplerSampleProvenance:
    observed_digest: str | None
    expected_reference_digest: str
    module: str | None
    qualname: str | None
    source_file: str | None
    source_inspected: bool
    source_error: str | None
    unwrapped_same: bool
    unwrapped_module: str | None
    unwrapped_qualname: str | None
    unwrapped_source_file: str | None
    comfyui_version: str | None

    def log_fields(self) -> str:
        return (
            f"observed_digest={self.observed_digest or 'unavailable'} "
            f"expected_reference_digest={self.expected_reference_digest} "
            f"module={self.module or 'unknown'} "
            f"qualname={self.qualname or 'unknown'} "
            f"source_file={self.source_file or 'unknown'} "
            f"source_inspected={self.source_inspected} "
            f"source_error={self.source_error or 'none'} "
            f"unwrapped_same={self.unwrapped_same} "
            f"unwrapped_module={self.unwrapped_module or 'unknown'} "
            f"unwrapped_qualname={self.unwrapped_qualname or 'unknown'} "
            f"unwrapped_source_file={self.unwrapped_source_file or 'unknown'} "
            f"comfyui_version={self.comfyui_version or 'unknown'}"
        )


@dataclass(frozen=True, slots=True)
class KSamplerSampleContract:
    accepted: bool
    failure: str | None
    provenance: KSamplerSampleProvenance


def _safe_source_file(function: Any) -> str | None:
    try:
        return inspect.getsourcefile(function)
    except (OSError, TypeError):
        return None


def _loaded_comfyui_version() -> str | None:
    module = sys.modules.get("comfyui_version")
    value = getattr(module, "__version__", None) if module is not None else None
    return str(value) if value is not None else None


def _function_source_node(
    function: Callable[..., Any],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str | None]:
    try:
        source = inspect.getsource(function)
        module = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        return None, f"source contains {len(functions)} top-level functions"
    return functions[0], None


def _provenance(
    function: Callable[..., Any],
    expected_reference_digest: str,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    source_error: str | None,
) -> KSamplerSampleProvenance:
    try:
        unwrapped = inspect.unwrap(function)
    except (TypeError, ValueError):
        unwrapped = function
    return KSamplerSampleProvenance(
        observed_digest=(
            function_node_ast_digest(function_node)
            if function_node is not None
            else None
        ),
        expected_reference_digest=expected_reference_digest,
        module=getattr(function, "__module__", None),
        qualname=getattr(function, "__qualname__", None),
        source_file=_safe_source_file(function),
        source_inspected=function_node is not None,
        source_error=source_error,
        unwrapped_same=unwrapped is function,
        unwrapped_module=getattr(unwrapped, "__module__", None),
        unwrapped_qualname=getattr(unwrapped, "__qualname__", None),
        unwrapped_source_file=_safe_source_file(unwrapped),
        comfyui_version=_loaded_comfyui_version(),
    )


def _name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _subscript(node: ast.AST, name: str, key: str | int) -> bool:
    if not isinstance(node, ast.Subscript) or not _name(node.value, name):
        return False
    slice_node = node.slice
    if isinstance(key, int) and key < 0:
        return (
            isinstance(slice_node, ast.UnaryOp)
            and isinstance(slice_node.op, ast.USub)
            and isinstance(slice_node.operand, ast.Constant)
            and slice_node.operand.value == -key
        )
    return isinstance(slice_node, ast.Constant) and slice_node.value == key


def _single_assigned_name(root: ast.AST, value: ast.AST) -> str | None:
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Assign)
            and node.value is value
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id
    return None


def _direct_assignment(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    value: ast.AST,
) -> ast.Assign | None:
    return next(
        (
            node
            for node in function_node.body
            if isinstance(node, ast.Assign) and node.value is value
        ),
        None,
    )


def _resolve_global(function: Callable[..., Any], node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return function.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve_global(function, node.value)
        return getattr(owner, node.attr, None) if owner is not None else None
    return None


def _assignment_line_to_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _contract_failure(
    failure: str,
    provenance: KSamplerSampleProvenance,
) -> KSamplerSampleContract:
    return KSamplerSampleContract(False, failure, provenance)


def validate_ksampler_sample(
    function: Callable[..., Any],
    *,
    expected_adapter: type,
    expected_reference_digest: str,
) -> KSamplerSampleContract:
    """Prove only the native adapter flow required by ER-SDE compensation."""
    function_node, source_error = _function_source_node(function)
    provenance = _provenance(
        function,
        expected_reference_digest,
        function_node,
        source_error,
    )
    if not inspect.isfunction(function):
        return _contract_failure(
            "KSAMPLER.sample is not a plain function descriptor",
            provenance,
        )
    if function_node is None:
        return _contract_failure(
            "source is unavailable, so adapter semantics cannot be proven",
            provenance,
        )
    if isinstance(function_node, ast.AsyncFunctionDef):
        return _contract_failure("KSAMPLER.sample cannot be asynchronous", provenance)
    if function_node.decorator_list:
        return _contract_failure(
            "KSAMPLER.sample decorators have unproven runtime semantics",
            provenance,
        )

    try:
        signature = inspect.signature(function, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        return _contract_failure(f"signature inspection failed: {exc}", provenance)
    expected_parameters = (
        "self",
        "model_wrap",
        "sigmas",
        "extra_args",
        "callback",
        "noise",
        "latent_image",
        "denoise_mask",
        "disable_pbar",
    )
    parameters = list(signature.parameters.values())
    if len(parameters) < len(expected_parameters) or any(
        parameter.name != expected
        or parameter.kind
        not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter, expected in zip(parameters, expected_parameters, strict=False)
    ):
        return _contract_failure(
            "signature does not preserve the required positional invocation contract",
            provenance,
        )

    sampler_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func) == ("self", "sampler_function")
    ]
    if len(sampler_calls) != 1:
        return _contract_failure(
            f"expected exactly one self.sampler_function dispatch, found {len(sampler_calls)}",
            provenance,
        )
    dispatch = sampler_calls[0]
    dispatch_assignment = _direct_assignment(function_node, dispatch)
    dispatch_result = (
        _single_assigned_name(function_node, dispatch)
        if dispatch_assignment is not None
        else None
    )
    if dispatch_result is None or dispatch_assignment is None:
        return _contract_failure(
            "sampler dispatch is not a direct method-body assignment",
            provenance,
        )

    adapter_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and _resolve_global(function, node.func) is expected_adapter
    ]
    if len(adapter_calls) != 1:
        return _contract_failure(
            "expected exactly one native KSamplerX0Inpaint adapter construction",
            provenance,
        )
    adapter_call = adapter_calls[0]
    adapter_assignment = _direct_assignment(function_node, adapter_call)
    adapter_name = (
        _single_assigned_name(function_node, adapter_call)
        if adapter_assignment is not None
        else None
    )
    if (
        adapter_name is None
        or adapter_assignment is None
        or len(adapter_call.args) != 2
        or adapter_call.keywords
        or not _name(adapter_call.args[0], "model_wrap")
        or not _name(adapter_call.args[1], "sigmas")
    ):
        return _contract_failure(
            "native model adapter is not constructed from model_wrap and sigmas",
            provenance,
        )

    noise_scaling_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func)
        == ("model_wrap", "inner_model", "model_sampling", "noise_scaling")
    ]
    if len(noise_scaling_calls) != 1:
        return _contract_failure(
            f"expected one native noise_scaling call, found {len(noise_scaling_calls)}",
            provenance,
        )
    noise_scaling = noise_scaling_calls[0]
    noise_scaling_assignment = _direct_assignment(function_node, noise_scaling)
    scaled_noise_name = (
        _single_assigned_name(function_node, noise_scaling)
        if noise_scaling_assignment is not None
        else None
    )
    expected_max_denoise = (
        len(noise_scaling.args) == 4
        and isinstance(noise_scaling.args[3], ast.Call)
        and _attribute_path(noise_scaling.args[3].func) == ("self", "max_denoise")
        and len(noise_scaling.args[3].args) == 2
        and _name(noise_scaling.args[3].args[0], "model_wrap")
        and _name(noise_scaling.args[3].args[1], "sigmas")
        and not noise_scaling.args[3].keywords
    )
    if (
        scaled_noise_name is None
        or noise_scaling_assignment is None
        or noise_scaling.keywords
        or len(noise_scaling.args) != 4
        or not _subscript(noise_scaling.args[0], "sigmas", 0)
        or not _name(noise_scaling.args[1], "noise")
        or not _name(noise_scaling.args[2], "latent_image")
        or not expected_max_denoise
    ):
        return _contract_failure(
            "initial noise_scaling inputs or max_denoise forwarding changed",
            provenance,
        )

    adapter_index = function_node.body.index(adapter_assignment)
    noise_scaling_index = function_node.body.index(noise_scaling_assignment)
    dispatch_index = function_node.body.index(dispatch_assignment)
    if not adapter_index < noise_scaling_index < dispatch_index:
        return _contract_failure(
            "native adapter/noise_scaling must occur before sampler dispatch",
            provenance,
        )

    denoise_mask_assignments = [
        node
        for node in function_node.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _subscript(node.targets[0], "extra_args", "denoise_mask")
        and _name(node.value, "denoise_mask")
    ]
    if (
        len(denoise_mask_assignments) != 1
        or function_node.body.index(denoise_mask_assignments[0]) >= dispatch_index
    ):
        return _contract_failure(
            "denoise_mask is not forwarded through extra_args before dispatch",
            provenance,
        )
    extra_args_stores = [
        target
        for node in ast.walk(function_node)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if (
            isinstance(target, ast.Name)
            and target.id == "extra_args"
        )
        or (
            isinstance(target, ast.Subscript)
            and _name(target.value, "extra_args")
            and not _subscript(target, "extra_args", "denoise_mask")
        )
    ]
    if extra_args_stores:
        return _contract_failure(
            "extra_args are modified beyond native denoise_mask forwarding",
            provenance,
        )
    extra_args_mutating_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _name(node.func.value, "extra_args")
        and node.func.attr != "get"
    ]
    if extra_args_mutating_calls:
        return _contract_failure(
            "extra_args method calls have unproven mutation semantics",
            provenance,
        )
    sigmas_stores = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, (ast.Name, ast.Subscript))
        and isinstance(getattr(node, "ctx", None), ast.Store)
        and (
            (isinstance(node, ast.Name) and node.id == "sigmas")
            or (isinstance(node, ast.Subscript) and _name(node.value, "sigmas"))
        )
    ]
    if sigmas_stores:
        return _contract_failure("sigmas are mutated inside KSAMPLER.sample", provenance)
    extra_options_stores = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, (ast.Attribute, ast.Subscript))
        and isinstance(getattr(node, "ctx", None), ast.Store)
        and any(
            _attribute_path(item) == ("self", "extra_options")
            for item in ast.walk(node)
        )
    ]
    if extra_options_stores:
        return _contract_failure(
            "self.extra_options are mutated before native dispatch",
            provenance,
        )

    latent_assignments = [
        node
        for node in function_node.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _attribute_path(node.targets[0]) == (adapter_name, "latent_image")
        and _name(node.value, "latent_image")
    ]
    if len(latent_assignments) != 1 or function_node.body.index(latent_assignments[0]) >= dispatch_index:
        return _contract_failure(
            "native adapter latent_image assignment is missing or reordered",
            provenance,
        )
    inpaint_branches = [
        node
        for node in function_node.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and _attribute_path(node.test.func) == ("self", "inpaint_options", "get")
        and len(node.test.args) == 2
        and isinstance(node.test.args[0], ast.Constant)
        and node.test.args[0].value == "random"
        and isinstance(node.test.args[1], ast.Constant)
        and node.test.args[1].value is False
    ]
    proven_inpaint_branches = []
    for branch in inpaint_branches:
        normal_noise_assignments = [
            node
            for node in ast.walk(ast.Module(body=branch.orelse, type_ignores=[]))
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and _attribute_path(node.targets[0]) == (adapter_name, "noise")
            and _name(node.value, "noise")
        ]
        random_noise_assignments = [
            node
            for node in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and _attribute_path(node.targets[0]) == (adapter_name, "noise")
        ]
        if len(normal_noise_assignments) == 1 and len(random_noise_assignments) == 1:
            proven_inpaint_branches.append(branch)
    if len(proven_inpaint_branches) != 1:
        return _contract_failure(
            "native model adapter noise ownership is not preserved",
            provenance,
        )

    if (
        len(dispatch.args) != 3
        or not _name(dispatch.args[0], adapter_name)
        or not _name(dispatch.args[1], scaled_noise_name)
        or not _name(dispatch.args[2], "sigmas")
    ):
        return _contract_failure(
            "sampler dispatch does not receive the native adapter, scaled noise, and sigmas",
            provenance,
        )
    keyword_map = {keyword.arg: keyword.value for keyword in dispatch.keywords if keyword.arg}
    star_keywords = [keyword.value for keyword in dispatch.keywords if keyword.arg is None]
    if set(keyword_map) != {"extra_args", "callback", "disable"}:
        return _contract_failure(
            "sampler dispatch keyword forwarding changed",
            provenance,
        )
    if not _name(keyword_map["extra_args"], "extra_args"):
        return _contract_failure(
            "sampler dispatch does not forward extra_args unchanged",
            provenance,
        )
    if not _name(keyword_map["disable"], "disable_pbar"):
        return _contract_failure(
            "sampler dispatch does not forward disable_pbar as disable",
            provenance,
        )
    if (
        len(star_keywords) != 1
        or _attribute_path(star_keywords[0]) != ("self", "extra_options")
    ):
        return _contract_failure(
            "self.extra_options are not forwarded unchanged",
            provenance,
        )

    callback_name_node = keyword_map["callback"]
    if not isinstance(callback_name_node, ast.Name):
        return _contract_failure(
            "sampler callback is not the local KSAMPLER callback adapter",
            provenance,
        )
    callbacks = [
        node
        for node in function_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node is not function_node
        and node.name == callback_name_node.id
    ]
    if len(callbacks) != 1:
        return _contract_failure(
            "local KSAMPLER callback adapter cannot be identified uniquely",
            provenance,
        )
    callback_function = callbacks[0]
    callback_parameters = [
        *callback_function.args.posonlyargs,
        *callback_function.args.args,
    ]
    if len(callback_parameters) != 1:
        return _contract_failure(
            "local KSAMPLER callback adapter signature changed",
            provenance,
        )
    callback_state = callback_parameters[0].arg
    total_step_assignments = [
        node
        for node in function_node.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Sub)
        and isinstance(node.value.left, ast.Call)
        and _name(node.value.left.func, "len")
        and len(node.value.left.args) == 1
        and _name(node.value.left.args[0], "sigmas")
        and isinstance(node.value.right, ast.Constant)
        and node.value.right.value == 1
    ]
    total_steps_names = {node.targets[0].id for node in total_step_assignments}
    if not total_steps_names:
        return _contract_failure("native total_steps calculation changed", provenance)
    all_forwarded_callbacks = [
        node
        for node in ast.walk(callback_function)
        if isinstance(node, ast.Call) and _name(node.func, "callback")
    ]
    callback_guards = [
        node
        for node in callback_function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _name(node.test.left, "callback")
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.IsNot)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
        and any(call in set(ast.walk(node)) for call in all_forwarded_callbacks)
    ]
    if len(callback_guards) != 1:
        return _contract_failure(
            "external callback forwarding guard changed",
            provenance,
        )
    forwarded_callbacks = [
        node
        for node in ast.walk(callback_guards[0])
        if isinstance(node, ast.Call) and _name(node.func, "callback")
    ]
    if len(forwarded_callbacks) != 1 or len(all_forwarded_callbacks) != 1:
        return _contract_failure(
            "expected one guarded external callback forwarding call",
            provenance,
        )
    forwarded_callback = forwarded_callbacks[0]
    expected_callback_keys = ("i", "denoised", "x")
    forwarded_total_steps = (
        forwarded_callback.args[3]
        if len(forwarded_callback.args) == 4
        else None
    )
    if (
        len(forwarded_callback.args) != 4
        or forwarded_callback.keywords
        or any(
            not _subscript(argument, callback_state, key)
            for argument, key in zip(
                forwarded_callback.args[:3], expected_callback_keys, strict=True
            )
        )
        or not isinstance(forwarded_total_steps, ast.Name)
        or forwarded_total_steps.id not in total_steps_names
    ):
        return _contract_failure(
            "callback does not forward native i/denoised/x/total_steps values",
            provenance,
        )
    callback_state_stores = [
        node
        for node in ast.walk(callback_function)
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript))
        and isinstance(getattr(node, "ctx", None), ast.Store)
        and (
            (isinstance(node, ast.Name) and node.id == callback_state)
            or (
                isinstance(node, (ast.Attribute, ast.Subscript))
                and any(_name(item, callback_state) for item in ast.walk(node))
            )
        )
    ]
    if callback_state_stores:
        return _contract_failure(
            "local callback mutates the native ER-SDE callback state",
            provenance,
        )

    inverse_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func)
        == (
            "model_wrap",
            "inner_model",
            "model_sampling",
            "inverse_noise_scaling",
        )
    ]
    if len(inverse_calls) != 1:
        return _contract_failure(
            f"expected one final inverse_noise_scaling call, found {len(inverse_calls)}",
            provenance,
        )
    inverse = inverse_calls[0]
    inverse_assignment = _direct_assignment(function_node, inverse)
    final_result = (
        _single_assigned_name(function_node, inverse)
        if inverse_assignment is not None
        else None
    )
    if (
        final_result is None
        or inverse_assignment is None
        or inverse.keywords
        or len(inverse.args) != 2
        or not _subscript(inverse.args[0], "sigmas", -1)
        or not _name(inverse.args[1], dispatch_result)
        or function_node.body.index(inverse_assignment) <= dispatch_index
    ):
        return _contract_failure(
            "final inverse_noise_scaling inputs or ordering changed",
            provenance,
        )
    returns = [node for node in function_node.body if isinstance(node, ast.Return)]
    if (
        len(returns) != 1
        or not _name(returns[0].value, final_result)
        or function_node.body.index(returns[0])
        <= function_node.body.index(inverse_assignment)
    ):
        return _contract_failure(
            "KSAMPLER.sample does not return the final inverse-scaled result",
            provenance,
        )

    scaled_noise_reassignments = [
        node
        for node in ast.walk(function_node)
        if _assignment_line_to_name(node, scaled_noise_name)
        and noise_scaling.lineno < node.lineno < dispatch.lineno
    ]
    result_reassignments = [
        node
        for node in ast.walk(function_node)
        if _assignment_line_to_name(node, dispatch_result)
        and dispatch.lineno < node.lineno < inverse.lineno
    ]
    adapter_reassignments = [
        node
        for node in ast.walk(function_node)
        if _assignment_line_to_name(node, adapter_name)
        and adapter_call.lineno < node.lineno < dispatch.lineno
    ]
    if scaled_noise_reassignments or result_reassignments or adapter_reassignments:
        return _contract_failure(
            "adapter, scaled noise, or sampler result is reassigned before its required consumer",
            provenance,
        )

    return KSamplerSampleContract(True, None, provenance)


__all__ = [
    "KSamplerSampleContract",
    "KSamplerSampleProvenance",
    "validate_ksampler_sample",
]
