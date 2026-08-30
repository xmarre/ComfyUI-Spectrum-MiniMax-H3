from __future__ import annotations

import copy
import inspect
import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

from .er_sde_ksampler_contract import (
    KSamplerSampleContract,
    validate_ksampler_sample,
)
from .er_sde_stochastic import (
    ERSDEStochasticTracker,
    ERSDETrackingError,
    function_ast_digest,
    native_default_er_sde_noise_scaler,
)
from .model_aware import get_model_forecastability_profile
from .refdelta_interop import (
    REFDELTA_BACKEND_INTEROP_CONTRACT,
    REFDELTA_BRIDGE_KEY,
    REFDELTA_INTEROP_CONTRACT,
    RefDeltaBackendInteropBridge,
    RefDeltaInteropBridge,
    RefDeltaInteropError,
)
from .rollback import run_selective_rollback_euler
from .runtime import (
    ForecastRetryActual,
    OfflineReplayAbort,
    SolverCallDescriptor,
    SpectrumH3Runtime,
)

LOG = logging.getLogger(__name__)

BINDING_KEY = "spectrum_h3_binding"
RUNTIME_KEY = "spectrum_h3_runtime"
RUN_ID_KEY = "spectrum_h3_run_id"
STEP_ID_KEY = "spectrum_h3_step_id"
OUTER_STEP_ID_KEY = "spectrum_h3_outer_step_id"
SOLVER_PHASE_KEY = "spectrum_h3_solver_phase"
COORDINATE_KEY = "spectrum_h3_coordinate"
ACTUAL_KEY = "spectrum_h3_actual"
REASON_KEY = "spectrum_h3_reason"
ER_SDE_TRACKER_KEY = "spectrum_h3_er_sde_stochastic_tracker"
WRAPPER_KEY = "spectrum_minimax_h3"
KJ_PREVIEW_WRAPPER_KEY = "kj_preview_override"
CONTINUUM_REQUEST_KEY = "h3_continuum"

SUPPORTED_SINGLE_CALL_SAMPLERS = frozenset(
    {
        "_turbo_sampler",
        "sample_euler",
        "sample_er_sde",
        "sample_refdelta_er_sde",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

NATIVE_SEEDS_SAMPLERS = frozenset({"sample_seeds_2", "sample_seeds_3"})
REFDELTA_SEEDS_SAMPLERS = frozenset(
    {"sample_refdelta_seeds_2", "sample_refdelta_seeds_3"}
)
SEEDS_SAMPLERS = NATIVE_SEEDS_SAMPLERS | REFDELTA_SEEDS_SAMPLERS
NATIVE_SA_SOLVER_SAMPLERS = frozenset({"sample_sa_solver", "sample_sa_solver_pece"})
REFDELTA_SA_SOLVER_SAMPLERS = frozenset({"sample_refdelta_sa_solver"})
SA_SOLVER_SAMPLERS = NATIVE_SA_SOLVER_SAMPLERS | REFDELTA_SA_SOLVER_SAMPLERS
REFDELTA_BACKEND_SAMPLERS = REFDELTA_SEEDS_SAMPLERS | REFDELTA_SA_SOLVER_SAMPLERS
SUPPORTED_SAMPLERS = SUPPORTED_SINGLE_CALL_SAMPLERS | SEEDS_SAMPLERS | SA_SOLVER_SAMPLERS
SEEDS_STAGE_COUNTS = {
    "sample_seeds_2": 2,
    "sample_seeds_3": 3,
    "sample_refdelta_seeds_2": 2,
    "sample_refdelta_seeds_3": 3,
}
REFDELTA_SEEDS_BASE_NAMES = {
    "sample_refdelta_seeds_2": "sample_seeds_2",
    "sample_refdelta_seeds_3": "sample_seeds_3",
}
SEEDS_NATIVE_FUNCTION_DIGESTS = {
    "sample_seeds_2": "e7bcf519718453f77e7ade9b71678c1b593472c8e9b0af142f64f97a9267f383",
    "sample_seeds_3": "8cb90838a30f6ed0d9ba267f0a16275ce8ec5d65a9a5475f1a10cb573d45ce42",
}
SEEDS_TRACKED_OPTIONS = {
    "sample_seeds_2": frozenset({"eta", "s_noise", "noise_sampler", "r", "solver_type"}),
    "sample_seeds_3": frozenset({"eta", "s_noise", "noise_sampler", "r_1", "r_2"}),
    "sample_refdelta_seeds_2": frozenset(
        {"eta", "s_noise", "noise_sampler", "r", "solver_type", "config"}
    ),
    "sample_refdelta_seeds_3": frozenset(
        {"eta", "s_noise", "noise_sampler", "r_1", "r_2", "config"}
    ),
}


SA_SOLVER_NATIVE_FUNCTION_DIGESTS = {
    "sample_sa_solver": "37bd2f94f27426b9e3e6f0cc2a0f439c95d3fec927339e0a7cc2c64e731647a4",
    "sample_sa_solver_pece": "6ce4b3d604ef789d69a8abf832cf93637646e387039e494e37bf9ed35e9b23a0",
}
SA_SOLVER_HELPER_DIGESTS = {
    "compute_exponential_coeffs": "d1bcd6cfa194ceec7a4e497c205ba59dfb3824d2816a5e159df67b57845ce72e",
    "compute_simple_stochastic_adams_b_coeffs": "46d46ac17ec32e6c0a047d0f45d9e0e3734efb9fbf64b2fb984aa9311354c4c6",
    "compute_stochastic_adams_b_coeffs": "f2480f0fc36f70f48909779e9604fa9dcb4e2a475b0040fd83c6e9e01f65e6e2",
    "get_tau_interval_func": "f301a5d271a219a7f8b709e264327477945ddb7b2cadd40d700392c987afb3bb",
}
SA_SOLVER_TRACKED_OPTIONS = {
    "sample_sa_solver": frozenset(
        {
            "tau_func",
            "s_noise",
            "noise_sampler",
            "predictor_order",
            "corrector_order",
            "use_pece",
            "simple_order_2",
        }
    ),
    "sample_sa_solver_pece": frozenset(
        {
            "tau_func",
            "s_noise",
            "noise_sampler",
            "predictor_order",
            "corrector_order",
            "simple_order_2",
        }
    ),
    "sample_refdelta_sa_solver": frozenset(
        {
            "tau_func",
            "s_noise",
            "noise_sampler",
            "predictor_order",
            "corrector_order",
            "simple_order_2",
            "config",
        }
    ),
}

RES_MULTISTEP_SAMPLERS = frozenset(
    {
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

ER_SDE_SAMPLERS = frozenset({"sample_er_sde", "sample_refdelta_er_sde"})
REFDELTA_SAMPLER_NAME = "sample_refdelta_er_sde"
ER_SDE_NATIVE_SCALER_MODULE = "comfy_extras.nodes_custom_sampler"
ER_SDE_NATIVE_SCALER_FREEVARS = {
    "SamplerER_SDE.execute.<locals>.er_sde_noise_scaler": ("eta",),
    "SamplerER_SDE.execute.<locals>.reverse_time_sde_noise_scaler": ("eta",),
    "SamplerER_SDE.execute.<locals>.ode_noise_scaler": (),
}
ER_SDE_NATIVE_FUNCTION_DIGEST = (
    "55b76bd3a76d44fbd363de39f2ab3ea672c78de9f001f47168b47ec6ff6d2447"
)
ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST = (
    "11cfe81f36f0b43e96c12eff32a4f074f35227a53ca116e837bf268b6383f9ad"
)
ER_SDE_KSAMPLER_SAMPLE_DIGEST = (
    "cacf00387fb2b9d0e076c68e0bd3f75d104801aa8d34239115b3208153ed8dac"
)
ER_SDE_TRACKED_OPTIONS = frozenset(
    {"s_noise", "noise_sampler", "noise_scaler", "max_stage"}
)
REFDELTA_TRACKED_OPTIONS = ER_SDE_TRACKED_OPTIONS | {"config"}


@dataclass(slots=True)
class SpectrumH3Binding:
    runtime: SpectrumH3Runtime


def _continuum_actual_prefix(model_options: dict[str, Any] | None) -> int:
    transformer_options = (model_options or {}).get("transformer_options")
    if not isinstance(transformer_options, dict):
        return 0
    request = transformer_options.get(CONTINUUM_REQUEST_KEY)
    if not isinstance(request, dict):
        return 0

    api = request.get("api")
    active = request.get("active")
    prefix = request.get("min_actual_prefix_steps")
    if type(api) is not int or type(active) is not bool or type(prefix) is not int:
        return 0
    if api != 1 or active is not True or prefix < 0:
        return 0
    return prefix


def _continuum_prefix_for_phase(prefix: int, phase: str) -> int:
    return 0 if phase == "offline_replay" else prefix


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def sampler_is_supported(sampler: Any) -> bool:
    return sampler_name(sampler) in SUPPORTED_SAMPLERS


def _seeds_expected_model_calls(sampler: Any, sigmas: Any) -> int | None:
    """Count native SEEDS denoiser calls for the supplied outer sigma schedule."""
    stage_count = SEEDS_STAGE_COUNTS.get(sampler_name(sampler))
    if stage_count is None or not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return None
    sigma_values = sigmas.detach().reshape(-1)
    outer_steps = max(0, int(sigma_values.numel()) - 1)
    if outer_steps == 0:
        return 0
    nonterminal_steps = int(torch.count_nonzero(sigma_values[1:]).item())
    return outer_steps + (stage_count - 1) * nonterminal_steps


def _seeds_prefix_model_calls(
    sampler: Any,
    sigmas: Any,
    outer_prefix: int,
) -> int | None:
    """Translate an outer-step prefix contract into SEEDS logical model calls."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return None
    outer_steps = max(0, int(sigmas.numel()) - 1)
    prefix_steps = min(max(0, int(outer_prefix)), outer_steps)
    if prefix_steps == 0:
        return 0
    return _seeds_expected_model_calls(sampler, sigmas[: prefix_steps + 1])


def _seeds_stage_schedule_reason(sigmas: Any) -> str | None:
    """Validate the native stage lane topology used by stochastic SEEDS forecasting."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        return "SEEDS sigma schedule is not a one-dimensional tensor with at least one interval"
    values = sigmas.detach().reshape(-1)
    if not bool(torch.isfinite(values).all().item()):
        return "SEEDS sigma schedule contains nonfinite values"
    if bool((values[:-1] <= 0).any().item()):
        return "SEEDS sigma schedule contains a nonpositive pre-terminal sigma"
    if float(values[-1].item()) < 0.0:
        return "SEEDS sigma schedule ends at a negative sigma"
    if bool((values[1:] >= values[:-1]).any().item()):
        return "SEEDS sigma schedule is not strictly descending"
    return None


def _sa_solver_expected_model_calls(sampler: Any, sigmas: Any) -> int | None:
    """Count native SA-Solver denoiser calls, including active PECE evaluations."""
    name = sampler_name(sampler)
    if name not in SA_SOLVER_SAMPLERS:
        return None
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return None
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return None
    corrector_order = options.get("corrector_order", 4)
    if type(corrector_order) is not int or corrector_order < 0:
        return None
    use_pece = name == "sample_sa_solver_pece" or options.get("use_pece", False) is True
    outer_steps = max(0, int(sigmas.numel()) - 1)
    if not use_pece or corrector_order == 0 or outer_steps == 0:
        return outer_steps
    return 2 * outer_steps - 1


def _sa_solver_call_topology(
    sampler: Any,
    sigmas: Any,
) -> tuple[SolverCallDescriptor, ...] | None:
    """Describe every native SA model call without inventing phase time offsets."""
    expected = _sa_solver_expected_model_calls(sampler, sigmas)
    if expected is None:
        return None
    name = sampler_name(sampler)
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return None
    corrector_order = options.get("corrector_order", 4)
    use_pece = name == "sample_sa_solver_pece" or options.get("use_pece", False) is True
    active_pece = bool(use_pece and corrector_order > 0)
    outer_steps = max(0, int(sigmas.numel()) - 1)
    topology: list[SolverCallDescriptor] = []
    for outer_step in range(outer_steps):
        topology.append(
            SolverCallDescriptor(
                outer_step=outer_step,
                stage_index=0,
                phase="predicted",
            )
        )
        if active_pece and outer_step > 0:
            topology.append(
                SolverCallDescriptor(
                    outer_step=outer_step,
                    stage_index=1,
                    phase="corrected",
                )
            )
    if len(topology) != expected:
        return None
    return tuple(topology)


def _sa_solver_prefix_model_calls(
    sampler: Any,
    sigmas: Any,
    outer_prefix: int,
) -> int | None:
    """Translate Continuum's outer-step prefix into native SA model-call space."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return None
    outer_steps = max(0, int(sigmas.numel()) - 1)
    prefix_steps = min(max(0, int(outer_prefix)), outer_steps)
    if prefix_steps == 0:
        return 0
    return _sa_solver_expected_model_calls(sampler, sigmas[: prefix_steps + 1])


def _sa_solver_tau_values(tau_func: Any) -> dict[str, float] | None:
    """Return reviewed native tau closure values without executing the callback."""
    if tau_func is None:
        return None

    try:
        import comfy.k_diffusion.sa_solver as native_sa_solver
    except ImportError:
        return None

    if getattr(tau_func, "__module__", None) != "comfy.k_diffusion.sa_solver":
        return None
    if getattr(tau_func, "__qualname__", None) != (
        "get_tau_interval_func.<locals>.tau_func"
    ):
        return None
    code = getattr(tau_func, "__code__", None)
    if code is None or tuple(code.co_freevars) != (
        "end_sigma",
        "eta",
        "start_sigma",
    ):
        return None
    if getattr(tau_func, "__globals__", None) is not vars(native_sa_solver):
        return None
    outer_code = getattr(native_sa_solver.get_tau_interval_func, "__code__", None)
    if outer_code is None or code not in outer_code.co_consts:
        return None
    closure = getattr(tau_func, "__closure__", None) or ()
    if len(closure) != 3:
        return None

    values: dict[str, float] = {}
    for name, cell in zip(code.co_freevars, closure, strict=True):
        try:
            value = cell.cell_contents
        except ValueError:
            return None
        if type(value) not in (int, float):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        values[name] = numeric
    return values


def _sa_solver_tau_func_is_reviewed(tau_func: Any) -> bool:
    if tau_func is None:
        return True
    values = _sa_solver_tau_values(tau_func)
    return values is not None and values["eta"] >= 0.0


def _sa_solver_is_stochastic(sampler: Any) -> bool:
    """Conservatively detect whether native SA may inject noise during this run."""
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return True
    s_noise = options.get("s_noise", 1.0)
    if type(s_noise) not in (int, float) or not math.isfinite(float(s_noise)):
        return True
    if float(s_noise) <= 0.0:
        return False
    tau_func = options.get("tau_func")
    if tau_func is None:
        return True
    values = _sa_solver_tau_values(tau_func)
    if values is None:
        return True
    return values["eta"] > 0.0


def _sa_solver_is_active_pece(sampler: Any) -> bool:
    """Return whether native SA performs the corrected-state PECE evaluation."""
    name = sampler_name(sampler)
    if name not in SA_SOLVER_SAMPLERS:
        return False
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False
    use_pece = name == "sample_sa_solver_pece" or options.get("use_pece", False) is True
    corrector_order = options.get("corrector_order", 4)
    return bool(use_pece and type(corrector_order) is int and corrector_order > 0)


def _sa_solver_stochastic_protection(
    sampler: Any,
    sigmas: Any,
    model_sampling: Any,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None, str | None]:
    """Resolve native SA stochastic-input steps for telemetry and fail-closed review.

    Spectrum's SA adapter no longer blanket-forces these calls actual. The
    previous policy attacked the symptom by removing most usable forecasts.
    Instead the solver-aware adapter prevents a forecasted denoiser from entering
    persistent Adams history: a forecast is an ephemeral predictor input for its
    current interval only, while exact H3 evaluations are the only values kept as
    multistep history. Fresh stochastic x_pred states can therefore be forecast
    without recursively poisoning later Adams stencils.
    """
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        return None, None, "SA-Solver sigma schedule is not a one-dimensional tensor"
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return None, None, "SA-Solver sampler options are not a dictionary"

    try:
        configured_s_noise = float(options.get("s_noise", 1.0))
        noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
        effective_s_noise = configured_s_noise * noise_scale
    except (TypeError, ValueError):
        return None, None, "SA-Solver effective s_noise is not numeric"
    if not math.isfinite(effective_s_noise) or effective_s_noise < 0.0:
        return None, None, "SA-Solver effective s_noise is not finite and nonnegative"
    if effective_s_noise == 0.0:
        return (), (), None

    tau_func = options.get("tau_func")
    if tau_func is None:
        try:
            start_sigma = float(model_sampling.percent_to_sigma(0.2))
            end_sigma = float(model_sampling.percent_to_sigma(0.8))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return None, None, f"SA-Solver default tau interval is unavailable: {exc}"
        eta = 1.0
    else:
        values = _sa_solver_tau_values(tau_func)
        if values is None:
            return None, None, "SA-Solver tau interval provenance is unavailable"
        start_sigma = values["start_sigma"]
        end_sigma = values["end_sigma"]
        eta = values["eta"]

    if not all(math.isfinite(value) for value in (start_sigma, end_sigma, eta)):
        return None, None, "SA-Solver tau interval contains nonfinite values"
    if eta <= 0.0:
        return (), (), None

    sigma_values = sigmas.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(sigma_values).all().item()):
        return None, None, "SA-Solver sigma schedule contains nonfinite values"
    outer_steps = max(0, int(sigma_values.numel()) - 1)
    stochastic_input_steps = tuple(
        step_id
        for step_id in range(1, outer_steps)
        if start_sigma >= float(sigma_values[step_id].item()) >= end_sigma
    )
    if not stochastic_input_steps:
        return (), (), None

    return (), stochastic_input_steps, None


def _refdelta_backend_sampler_contract(
    name: str,
    function: Any,
    options: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate the versioned RefDelta SEEDS/SA interop surface."""
    try:
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver.sampler_backends import (
            sample_refdelta_sa_solver,
            sample_refdelta_seeds_2,
            sample_refdelta_seeds_3,
        )
        from comfyui_refdelta_solver.spectrum_interop import (
            SPECTRUM_BACKEND_INTEROP_CONTRACT,
        )
    except (ImportError, AttributeError) as exc:
        return False, f"RefDelta backend interop API is unavailable: {exc}"

    expected = {
        "sample_refdelta_seeds_2": sample_refdelta_seeds_2,
        "sample_refdelta_seeds_3": sample_refdelta_seeds_3,
        "sample_refdelta_sa_solver": sample_refdelta_sa_solver,
    }.get(name)
    if expected is None or function is not expected:
        return False, "sampler function is not the installed RefDelta backend implementation"
    if SPECTRUM_BACKEND_INTEROP_CONTRACT != REFDELTA_BACKEND_INTEROP_CONTRACT:
        return False, "RefDelta/Spectrum backend interop contract version does not match"
    if (
        getattr(function, "__spectrum_interop_contract__", None)
        != REFDELTA_BACKEND_INTEROP_CONTRACT
    ):
        return False, "RefDelta backend did not publish the reviewed interop contract"

    config = options.get("config")
    if type(config) is not RefDeltaSamplerConfig:
        return False, "RefDelta backend config has unreviewed provenance"
    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        return False, f"RefDelta backend config is invalid: {exc}"
    if config.calibration_capture:
        return False, "RefDelta calibration capture is ER-SDE-only"
    return True, None


def _with_refdelta_backend_bridge(
    extra_args: dict[str, Any],
    runtime: SpectrumH3Runtime,
) -> dict[str, Any]:
    copied = dict(extra_args or {})
    model_options = dict(copied.get("model_options") or {})
    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options[REFDELTA_BRIDGE_KEY] = RefDeltaBackendInteropBridge(runtime)
    model_options["transformer_options"] = transformer_options
    copied["model_options"] = model_options
    return copied


def _sa_solver_option_contract(name: str, options: Any) -> str | None:
    """Validate the native SA option surface without executing user callbacks."""
    if name not in SA_SOLVER_SAMPLERS:
        return f"{name!r} is not a reviewed SA-Solver sampler"
    if not isinstance(options, dict):
        return "SA-Solver sampler options are not a dictionary"

    unknown = set(options) - SA_SOLVER_TRACKED_OPTIONS[name]
    if unknown:
        return f"SA-Solver sampler has unreviewed options: {sorted(unknown)}"

    s_noise = options.get("s_noise", 1.0)
    if type(s_noise) not in (int, float) or not math.isfinite(float(s_noise)):
        return "SA-Solver s_noise must be a finite numeric value"
    if float(s_noise) < 0.0:
        return "SA-Solver s_noise must be nonnegative"

    for option_name, default, minimum in (
        ("predictor_order", 3, 1),
        ("corrector_order", 4, 0),
    ):
        value = options.get(option_name, default)
        if type(value) is not int or not minimum <= value <= 6:
            return f"SA-Solver {option_name} must be an integer in [{minimum}, 6]"

    if type(options.get("simple_order_2", False)) is not bool:
        return "SA-Solver simple_order_2 must be boolean"

    if name == "sample_sa_solver":
        configured_use_pece = options.get("use_pece", False)
        if type(configured_use_pece) is not bool:
            return "SA-Solver use_pece must be boolean"
    noise_sampler = options.get("noise_sampler")
    if noise_sampler is not None and not callable(noise_sampler):
        return "SA-Solver noise_sampler must be callable or None"
    if not _sa_solver_tau_func_is_reviewed(options.get("tau_func")):
        return (
            "SA-Solver tau_func must be None or the reviewed native "
            "get_tau_interval_func closure"
        )
    return None


def _sa_solver_sampler_contract(sampler: Any) -> tuple[bool, str | None]:
    """Validate native SA-Solver topology, solver helpers, and adapter ownership."""
    import comfy.k_diffusion.sa_solver as native_sa_solver
    import comfy.k_diffusion.sampling as native_sampling
    import comfy.samplers

    name = sampler_name(sampler)
    if name not in SA_SOLVER_SAMPLERS:
        return False, f"{name!r} is not a reviewed SA-Solver sampler"

    function = getattr(sampler, "sampler_function", None)
    if name in REFDELTA_SA_SOLVER_SAMPLERS:
        options = getattr(sampler, "extra_options", {}) or {}
        accepted, reason = _refdelta_backend_sampler_contract(name, function, options)
        if not accepted:
            return False, reason
    else:
        if function is not getattr(native_sampling, name, None):
            return False, f"sampler function is not native ComfyUI {name}"
        if function_ast_digest(function) != SA_SOLVER_NATIVE_FUNCTION_DIGESTS[name]:
            return False, f"native {name} implementation is not a reviewed revision"
    if (
        function_ast_digest(native_sampling.sample_sa_solver)
        != SA_SOLVER_NATIVE_FUNCTION_DIGESTS["sample_sa_solver"]
    ):
        return False, "native sample_sa_solver core is not a reviewed revision"
    for helper_name, expected_digest in SA_SOLVER_HELPER_DIGESTS.items():
        helper = getattr(native_sa_solver, helper_name, None)
        if function_ast_digest(helper) != expected_digest:
            return False, f"native SA-Solver helper {helper_name} is unreviewed"
    if (
        function_ast_digest(native_sampling.default_noise_sampler)
        != ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST
    ):
        return False, "native default_noise_sampler implementation is not reviewed"

    if type(sampler) is not comfy.samplers.KSAMPLER:
        return False, "SA-Solver sampler object is not native ComfyUI KSAMPLER"
    if (
        comfy.samplers.KSAMPLER.__module__ != "comfy.samplers"
        or comfy.samplers.KSAMPLER.__name__ != "KSAMPLER"
    ):
        return False, "native ComfyUI KSAMPLER class provenance is unreviewed"
    if (
        comfy.samplers.KSamplerX0Inpaint.__module__ != "comfy.samplers"
        or comfy.samplers.KSamplerX0Inpaint.__name__ != "KSamplerX0Inpaint"
    ):
        return False, "native KSamplerX0Inpaint adapter provenance is unreviewed"
    sample_contract = _ksampler_sample_contract(sampler)
    if not sample_contract.accepted:
        return False, (
            f"KSAMPLER.sample contract rejected: {sample_contract.failure}; "
            f"{sample_contract.provenance.log_fields()}"
        )

    options = getattr(sampler, "extra_options", {}) or {}
    option_reason = _sa_solver_option_contract(name, options)
    if option_reason is not None:
        return False, option_reason
    return True, None


def _native_sa_solver_preflight_reason(
    sampler: Any,
    model_options: dict[str, Any] | None,
) -> str | None:
    """Resolve reviewed SA failures before Spectrum retains any run state."""
    supported, reason = _sa_solver_sampler_contract(sampler)
    if not supported:
        return reason or "native SA-Solver contract is unproven"

    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        model_options or {},
        is_model_options=True,
    )
    if len(wrappers) != 1 or wrappers[0] is not sampler_sample_wrapper:
        return "another SAMPLER_SAMPLE wrapper makes SA-Solver ordering unproven"
    return None


def _seeds_option_contract(name: str, options: Any) -> str | None:
    """Validate the reviewed native SEEDS option and stage surface."""
    if name not in SEEDS_SAMPLERS:
        return f"{name!r} is not a reviewed SEEDS sampler"
    if not isinstance(options, dict):
        return "SEEDS sampler options are not a dictionary"

    unknown = set(options) - SEEDS_TRACKED_OPTIONS[name]
    if unknown:
        return f"SEEDS sampler has unreviewed options: {sorted(unknown)}"

    for option_name, default in (("eta", 1.0), ("s_noise", 1.0)):
        value = options.get(option_name, default)
        if isinstance(value, bool):
            return f"SEEDS {option_name} must be numeric"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"SEEDS {option_name} must be numeric"
        if not math.isfinite(numeric):
            return f"SEEDS {option_name} must be finite"

    noise_sampler = options.get("noise_sampler")
    if noise_sampler is not None and not callable(noise_sampler):
        return "SEEDS noise_sampler must be callable or None"

    base_name = REFDELTA_SEEDS_BASE_NAMES.get(name, name)
    if base_name == "sample_seeds_2":
        if options.get("solver_type", "phi_1") not in {"phi_1", "phi_2"}:
            return "SEEDS-2 solver_type must be 'phi_1' or 'phi_2'"
        try:
            r = float(options.get("r", 0.5))
        except (TypeError, ValueError):
            return "SEEDS-2 r is not numeric"
        if not math.isfinite(r) or not 0.0 < r < 1.0:
            return "SEEDS-2 requires a strictly interior stage coordinate 0 < r < 1"
    else:
        try:
            r_1 = float(options.get("r_1", 1.0 / 3.0))
            r_2 = float(options.get("r_2", 2.0 / 3.0))
        except (TypeError, ValueError):
            return "SEEDS-3 r_1 and r_2 must be numeric"
        if (
            not math.isfinite(r_1)
            or not math.isfinite(r_2)
            or not 0.0 < r_1 < r_2 < 1.0
        ):
            return "SEEDS-3 requires strictly interior ordered stages 0 < r_1 < r_2 < 1"
    return None


def _seeds_is_stochastic(sampler: Any) -> bool:
    """Return whether native SEEDS is configured to inject stochastic latent state."""
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return True
    try:
        eta = float(options.get("eta", 1.0))
        s_noise = float(options.get("s_noise", 1.0))
    except (TypeError, ValueError):
        return True
    if not math.isfinite(eta) or not math.isfinite(s_noise):
        return True
    # Native SEEDS multiplies s_noise by model_sampling.noise_scale later.
    # Treat configured-positive s_noise conservatively as stochastic: a model
    # with an effective zero noise_scale merely takes the safe native fallback.
    return eta > 0.0 and s_noise > 0.0


def _seeds_sampler_contract(sampler: Any) -> tuple[bool, str | None]:
    """Validate the reviewed native SEEDS multistage contract."""
    import comfy.k_diffusion.sampling as native_sampling
    import comfy.samplers

    name = sampler_name(sampler)
    if name not in SEEDS_SAMPLERS:
        return False, f"{name!r} is not a reviewed SEEDS sampler"

    function = getattr(sampler, "sampler_function", None)
    base_name = REFDELTA_SEEDS_BASE_NAMES.get(name, name)
    if name in REFDELTA_SEEDS_SAMPLERS:
        options = getattr(sampler, "extra_options", {}) or {}
        accepted, reason = _refdelta_backend_sampler_contract(name, function, options)
        if not accepted:
            return False, reason
    else:
        if function is not getattr(native_sampling, name, None):
            return False, f"sampler function is not native ComfyUI {name}"
        if function_ast_digest(function) != SEEDS_NATIVE_FUNCTION_DIGESTS[name]:
            return False, f"native {name} implementation is not a reviewed revision"
    native_base = getattr(native_sampling, base_name, None)
    if function_ast_digest(native_base) != SEEDS_NATIVE_FUNCTION_DIGESTS[base_name]:
        return False, f"native {base_name} implementation is not a reviewed revision"

    if type(sampler) is not comfy.samplers.KSAMPLER:
        return False, "SEEDS sampler object is not native ComfyUI KSAMPLER"
    if (
        comfy.samplers.KSAMPLER.__module__ != "comfy.samplers"
        or comfy.samplers.KSAMPLER.__name__ != "KSAMPLER"
    ):
        return False, "native ComfyUI KSAMPLER class provenance is unreviewed"
    sample_contract = _ksampler_sample_contract(sampler)
    if not sample_contract.accepted:
        return False, (
            f"KSAMPLER.sample contract rejected: {sample_contract.failure}; "
            f"{sample_contract.provenance.log_fields()}"
        )

    options = getattr(sampler, "extra_options", {}) or {}
    option_reason = _seeds_option_contract(name, options)
    if option_reason is not None:
        return False, option_reason
    return True, None


def _native_seeds_preflight_reason(
    sampler: Any,
    model_options: dict[str, Any] | None,
) -> str | None:
    """Fail closed before Spectrum retains any state for a native SEEDS run."""
    supported, reason = _seeds_sampler_contract(sampler)
    if not supported:
        return reason or "native SEEDS contract is unproven"

    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        model_options or {},
        is_model_options=True,
    )
    if len(wrappers) != 1 or wrappers[0] is not sampler_sample_wrapper:
        return "another SAMPLER_SAMPLE wrapper makes SEEDS model-call ordering unproven"
    return None


def _er_sde_noise_scaler_supports_replay(noise_scaler: Any) -> bool:
    """Accept only the reviewed native SamplerER_SDE scaler closures."""
    if noise_scaler is None:
        return True
    if getattr(noise_scaler, "__module__", None) != ER_SDE_NATIVE_SCALER_MODULE:
        return False

    qualname = getattr(noise_scaler, "__qualname__", None)
    expected_freevars = ER_SDE_NATIVE_SCALER_FREEVARS.get(qualname)
    if expected_freevars is None:
        return False

    code = getattr(noise_scaler, "__code__", None)
    if code is None or tuple(code.co_freevars) != expected_freevars:
        return False

    native_module = sys.modules.get(ER_SDE_NATIVE_SCALER_MODULE)
    if native_module is None or getattr(noise_scaler, "__globals__", None) is not vars(native_module):
        return False
    sampler_class = getattr(native_module, "SamplerER_SDE", None)
    execute = getattr(sampler_class, "execute", None)
    execute_function = getattr(execute, "__func__", execute)
    execute_code = getattr(execute_function, "__code__", None)
    if execute_code is None or not any(item is code for item in execute_code.co_consts):
        return False

    closure = getattr(noise_scaler, "__closure__", None) or ()
    if len(closure) != len(expected_freevars):
        return False
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            return False
        if type(value) not in (int, float):
            return False
    return True


def sampler_supports_seeded_replay(sampler: Any) -> bool:
    """Return whether a fresh invocation can reconstruct the sampler's random stream."""
    if not sampler_is_supported(sampler):
        return False

    name = sampler_name(sampler)
    if name in SA_SOLVER_SAMPLERS:
        # SA-Solver is a multistep method: replaying altered denoiser values
        # changes its Adams history even when the RNG itself is reproducible.
        return False
    if name in REFDELTA_SEEDS_SAMPLERS:
        # The backend bridge intentionally classifies only the live causal pass.
        # Do not create an offline replay mode whose provenance cannot be mapped
        # back to RefDelta's actual-only evidence contract.
        return False
    if name in SEEDS_SAMPLERS:
        options = getattr(sampler, "extra_options", {}) or {}
        if _seeds_option_contract(name, options) is not None:
            return False
        return not _seeds_is_stochastic(sampler)
    if name not in ER_SDE_SAMPLERS:
        return True

    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False
    if options.get("noise_sampler") is not None:
        return False
    return _er_sde_noise_scaler_supports_replay(options.get("noise_scaler"))


def _ksampler_sample_contract(sampler: Any) -> KSamplerSampleContract:
    import comfy.samplers

    sample_method = vars(type(sampler)).get("sample")
    return validate_ksampler_sample(
        sample_method,
        expected_adapter=comfy.samplers.KSamplerX0Inpaint,
        expected_reference_digest=ER_SDE_KSAMPLER_SAMPLE_DIGEST,
    )


def _copy_ksampler_with_options(
    sampler: Any,
    options: dict[str, Any],
) -> tuple[Any | None, str | None]:
    """Prove and perform the shallow copy used to isolate tracked options."""
    if not inspect.isfunction(vars(type(sampler)).get("sample")):
        return None, "KSAMPLER.sample is not a plain function descriptor"
    if vars(type(sampler)).get("__copy__") is not None:
        return None, "KSAMPLER defines an unreviewed custom __copy__ method"
    try:
        original_state = vars(sampler)
        original_options = sampler.extra_options
        copied = copy.copy(sampler)
    except Exception as exc:  # noqa: BLE001 - fail closed on custom runtime state
        return None, f"KSAMPLER state/copy inspection failed: {type(exc).__name__}: {exc}"
    if copied is sampler or type(copied) is not type(sampler):
        return None, "KSAMPLER shallow copy did not create a distinct native instance"
    try:
        for name, value in original_state.items():
            if getattr(copied, name, object()) is not value:
                return None, f"KSAMPLER shallow copy changed instance field {name!r}"
        copied.extra_options = options
        if sampler.extra_options is not original_options:
            return None, "KSAMPLER shallow copy mutated the original extra_options"
        if copied.extra_options is not options:
            return None, "KSAMPLER shallow copy did not accept invocation-local options"
        if copied.sampler_function is not sampler.sampler_function:
            return None, "KSAMPLER shallow copy changed sampler_function binding"
        if copied.inpaint_options is not sampler.inpaint_options:
            return None, "KSAMPLER shallow copy changed inpaint_options ownership"
    except Exception as exc:  # noqa: BLE001 - fail closed on custom runtime state
        return None, f"KSAMPLER copied-state validation failed: {type(exc).__name__}: {exc}"
    return copied, None


def _sa_predict_causal_denoised(
    actual_preds: list[torch.Tensor],
    actual_lambdas: list[torch.Tensor],
    actual_steps: list[int],
    *,
    target_lambda: torch.Tensor,
    target_step: int,
    continuum_active: bool = False,
    allow_consecutive_extrapolation: bool = False,
) -> tuple[torch.Tensor, str, tuple[int, ...], torch.Tensor]:
    """Predict SA's denoised variable directly from exact actual anchors.

    Fresh SA predictor noise changes the model input in a direction that cannot be
    reconstructed exactly when the H3 transformer is skipped. Solver-space dense
    output therefore uses exact H3 anchors only. The extrapolation is causal and
    bounded to one previous anchor interval; any untrusted geometry degenerates to
    a latest-actual hold.

    Ordinary stochastic PEC keeps the conservative rule that consecutive exact
    anchors imply a hold. Active PECE may opt into consecutive-anchor secant
    extrapolation because every forecasted predicted phase is immediately followed
    by an exact corrected endpoint before the next predictor.
    """
    if not (
        len(actual_preds) == len(actual_lambdas) == len(actual_steps)
        and actual_preds
    ):
        raise RuntimeError(
            "stochastic SA dense output requires aligned non-empty actual anchors"
        )

    latest_value = actual_preds[-1]
    latest_lambda = actual_lambdas[-1]
    latest_step = int(actual_steps[-1])
    if latest_step >= int(target_step):
        raise RuntimeError(
            "stochastic SA dense-output anchor is not strictly earlier than target"
        )
    if int(target_step) != latest_step + 1:
        raise RuntimeError(
            "stochastic SA dense output requires the latest actual anchor immediately "
            f"before the forecast (anchor={latest_step}, forecast={target_step})"
        )

    zero = target_lambda.new_zeros(())
    if continuum_active:
        return (
            latest_value.clone(memory_format=torch.contiguous_format),
            "latest_actual_hold_continuum",
            (latest_step,),
            zero,
        )
    if len(actual_preds) == 1:
        return (
            latest_value.clone(memory_format=torch.contiguous_format),
            "latest_actual_hold",
            (latest_step,),
            zero,
        )

    previous_value = actual_preds[-2]
    previous_lambda = actual_lambdas[-2]
    previous_step = int(actual_steps[-2])
    if (
        previous_value.shape != latest_value.shape
        or previous_value.device != latest_value.device
        or previous_value.dtype != latest_value.dtype
    ):
        raise RuntimeError("stochastic SA dense-output actual anchors are incompatible")

    # Ordinary stochastic PEC treats consecutive exact anchors as a conservative
    # hold boundary. Active PECE can instead use the exact corrected endpoints as
    # a one-step secant because each endpoint is native persistent Adams evidence.
    if latest_step - previous_step == 1 and not allow_consecutive_extrapolation:
        return (
            latest_value.clone(memory_format=torch.contiguous_format),
            "latest_actual_hold_consecutive_anchors",
            (latest_step,),
            zero,
        )

    denominator = latest_lambda - previous_lambda
    advance = target_lambda - latest_lambda
    alpha = advance / denominator
    scale = torch.maximum(
        torch.maximum(previous_lambda.abs(), latest_lambda.abs()),
        torch.ones_like(latest_lambda),
    )
    valid = (
        torch.isfinite(alpha)
        & torch.isfinite(denominator)
        & (denominator.abs() > 1e-12 * scale)
        & (denominator * advance >= 0.0)
        & (alpha >= -1e-6)
        & (alpha <= 1.0 + 1e-6)
    )
    bounded_alpha = alpha.clamp(0.0, 1.0)
    weight = torch.where(valid, bounded_alpha, torch.zeros_like(bounded_alpha))
    weight = weight.to(device=latest_value.device, dtype=latest_value.dtype)
    predicted = torch.lerp(latest_value, previous_value, -weight)
    return (
        predicted,
        "lambda_bounded_extrapolation",
        (previous_step, latest_step),
        weight,
    )


def _sample_sa_solver_pece_forecast_isolated(
    runtime: SpectrumH3Runtime,
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool = False,
    tau_func: Any = None,
    s_noise: float = 1.0,
    noise_sampler: Any = None,
    predictor_order: int = 3,
    corrector_order: int = 4,
    simple_order_2: bool = False,
    continuum_active: bool = False,
) -> torch.Tensor:
    """Run active PECE with forecastable predicted phases and exact corrected phases.

    ComfyUI evaluates the predicted state first, uses that endpoint in the current
    corrector, then evaluates the corrected state at the same sigma. The corrected
    evaluation replaces the predicted endpoint before the next predictor. Spectrum
    preserves that ownership explicitly:

    * a predicted-phase forecast is ephemeral and can affect only the current
      corrector state;
    * the corrected phase is always an actual H3 evaluation;
    * only the corrected actual becomes the persistent endpoint for that outer
      sigma (the initial predicted actual is retained when no corrector exists).

    Spectrum feature history mirrors native PECE endpoint ownership: the initial
    predicted actual is the first anchor, then each exact corrected evaluation
    replaces the same-coordinate predicted phase as the persistent feature anchor.
    Predicted phases after outer zero never enter persistent feature history, even
    if a hard boundary forces them exact.

    Every predicted-phase forecast is solver-owned. The raw hidden-feature
    forecast completes the cheap Spectrum transaction but is never consumed by
    SA. Instead the corrector receives a causal solver-space estimate built only
    from exact persistent PECE endpoints. Continuum keeps its stricter latest-exact
    hold policy for every forecast coordinate.
    """
    if corrector_order <= 0:
        raise ValueError("active PECE requires corrector_order > 0")
    if len(sigmas) <= 1:
        return x

    import comfy.k_diffusion.sa_solver as native_sa_solver
    import comfy.k_diffusion.sampling as native_sampling
    from tqdm.auto import trange

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = (
        native_sampling.default_noise_sampler(x, seed=seed)
        if noise_sampler is None
        else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    sigmas = native_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = native_sampling.sigma_to_half_log_snr(
        sigmas,
        model_sampling=model_sampling,
    )

    if tau_func is None:
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = native_sa_solver.get_tau_interval_func(
            start_sigma,
            end_sigma,
            eta=1.0,
        )

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x
    h = 0.0
    tau_t = 0.0
    noise = 0.0

    # One exact endpoint per completed outer coordinate. For outer step zero it
    # is the predicted evaluation; for every active corrector step it is the
    # corrected-state evaluation that native PECE assigns to pred_list[-1].
    persistent_preds: list[torch.Tensor] = []
    persistent_lambdas: list[torch.Tensor] = []
    persistent_outer_steps: list[int] = []

    # Dense-output anchors mirror the exact persistent PECE endpoint stream.
    # They are kept separate from the native Adams lists only so a declared
    # transformer discontinuity can reset interpolation geometry without changing
    # native Adams persistence/order.
    dense_endpoint_preds: list[torch.Tensor] = []
    dense_endpoint_lambdas: list[torch.Tensor] = []
    dense_endpoint_outer_steps: list[int] = []

    lower_order_to_end = sigmas[-1].item() == 0

    def append_bounded(
        values: list[torch.Tensor],
        coordinates: list[torch.Tensor],
        outer_steps: list[int],
        value: torch.Tensor,
        coordinate: torch.Tensor,
        outer_step: int,
    ) -> None:
        values.append(value)
        coordinates.append(coordinate)
        outer_steps.append(outer_step)
        if len(values) > max_used_order:
            del values[:-max_used_order]
            del coordinates[:-max_used_order]
            del outer_steps[:-max_used_order]

    for i in trange(len(sigmas) - 1, disable=disable):
        raw_predicted = model(
            x_pred,
            sigmas[i] * s_in,
            **extra_args,
        )
        predicted_mode = runtime.last_completed_mode
        predicted_step_id = 0 if i == 0 else 2 * i - 1
        if runtime.last_completed_step_id not in {None, predicted_step_id}:
            raise RuntimeError(
                "Spectrum PECE predicted-state call arrived at the wrong logical "
                f"coordinate (outer={i}, expected={predicted_step_id}, "
                f"observed={runtime.last_completed_step_id})"
            )
        if predicted_mode not in {"actual", "forecast"}:
            raise RuntimeError(
                "Spectrum PECE adapter could not classify the predicted-state "
                f"evaluation at outer step {i}"
            )
        predicted_actual = predicted_mode == "actual"
        predicted_reason = getattr(runtime, "last_completed_reason", None)
        predicted = raw_predicted
        solver_value_source = "actual_h3" if predicted_actual else "feature_forecast"
        if not predicted_actual:
            predicted, dense_mode, anchor_steps, dense_alpha = (
                _sa_predict_causal_denoised(
                    dense_endpoint_preds,
                    dense_endpoint_lambdas,
                    dense_endpoint_outer_steps,
                    target_lambda=lambdas[i],
                    target_step=i,
                    continuum_active=continuum_active,
                    allow_consecutive_extrapolation=True,
                )
            )
            solver_value_source = dense_mode
            if runtime.config.debug:
                alpha_value = float(dense_alpha.detach().to(device="cpu").item())
                LOG.warning(
                    "Spectrum H3 SA PECE dense output sa_outer=%s sa_phase=predicted "
                    "scope=%s mode=%s anchor_outer_steps=%s alpha=%.8f "
                    "raw_feature_forecast=ignored "
                    "history_lane=persistent_endpoint_actual_only",
                    i,
                    (
                        "continuum_all_forecasts"
                        if continuum_active
                        else "pece_all_forecasts"
                    ),
                    dense_mode,
                    anchor_steps,
                    alpha_value,
                )

        if callback is not None:
            callback(
                {
                    "x": x_pred,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": predicted,
                }
            )

        # The current predicted endpoint is valid input to this corrector even
        # when it is forecast. It is deliberately absent from persistent_preds
        # until native PECE resolves the endpoint through the corrected phase.
        corrector_preds = [*persistent_preds, predicted]
        corrector_lambdas = [*persistent_lambdas, lambdas[i]]
        if len(corrector_preds) > max_used_order:
            corrector_preds = corrector_preds[-max_used_order:]
            corrector_lambdas = corrector_lambdas[-max_used_order:]

        predictor_order_used = min(predictor_order, len(corrector_preds))
        corrector_order_used = 0 if i == 0 else min(
            corrector_order,
            len(corrector_preds),
        )
        if lower_order_to_end:
            predictor_order_used = min(
                predictor_order_used,
                len(sigmas) - 2 - i,
            )
            corrector_order_used = min(
                corrector_order_used,
                len(sigmas) - 1 - i,
            )

        if corrector_order_used == 0:
            x = x_pred
            if not predicted_actual:
                raise RuntimeError(
                    "active PECE cannot begin without an actual initial endpoint"
                )
            append_bounded(
                persistent_preds,
                persistent_lambdas,
                persistent_outer_steps,
                predicted,
                lambdas[i],
                i,
            )
            append_bounded(
                dense_endpoint_preds,
                dense_endpoint_lambdas,
                dense_endpoint_outer_steps,
                predicted,
                lambdas[i],
                i,
            )
            endpoint = predicted
            predicted_persistent = True
        else:
            curr_lambdas = torch.stack(
                corrector_lambdas[-corrector_order_used:]
            )
            b_coeffs = native_sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(
                corrector_preds[-corrector_order_used:],
                dim=1,
            )
            corr_res = torch.tensordot(
                pred_mat,
                b_coeffs,
                dims=([1], [0]),
            )
            x = (
                sigmas[i]
                / sigmas[i - 1]
                * (-(tau_t**2) * h).exp()
                * x
                + corr_res
            )
            if tau_t > 0 and s_noise > 0:
                x = x + noise

            corrected = model(
                x,
                sigmas[i] * s_in,
                **extra_args,
            )
            corrected_mode = runtime.last_completed_mode
            corrected_step_id = 2 * i
            if runtime.last_completed_step_id not in {None, corrected_step_id}:
                raise RuntimeError(
                    "Spectrum PECE corrected-state call arrived at the wrong logical "
                    f"coordinate (outer={i}, expected={corrected_step_id}, "
                    f"observed={runtime.last_completed_step_id})"
                )
            if corrected_mode != "actual":
                raise RuntimeError(
                    "Spectrum PECE corrected-state phase violated its exact reanchor policy "
                    f"at outer step {i} (mode={corrected_mode!r})"
                )
            append_bounded(
                persistent_preds,
                persistent_lambdas,
                persistent_outer_steps,
                corrected,
                lambdas[i],
                i,
            )
            if predicted_reason == "external patch hard sigma transition":
                dense_endpoint_preds.clear()
                dense_endpoint_lambdas.clear()
                dense_endpoint_outer_steps.clear()
                if runtime.config.debug:
                    LOG.warning(
                        "Spectrum H3 SA PECE dense history reset sa_outer=%s "
                        "reason=external_patch_hard_transition "
                        "native_adams_history=preserved",
                        i,
                    )
            append_bounded(
                dense_endpoint_preds,
                dense_endpoint_lambdas,
                dense_endpoint_outer_steps,
                corrected,
                lambdas[i],
                i,
            )
            endpoint = corrected
            predicted_persistent = False
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 SA PECE opportunity logical_step=%s "
                    "sa_outer=%s sa_phase=corrected sigma=%.9g "
                    "actual_model_evaluation=1 forecast=0 "
                    "reason=pece_exact_reanchor adams_persistent=1 "
                    "solver_value_source=actual_h3 continuum=%s",
                    corrected_step_id,
                    i,
                    float(sigmas[i].detach().to(device="cpu").item()),
                    int(continuum_active),
                )

        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 SA PECE opportunity logical_step=%s "
                "sa_outer=%s sa_phase=predicted sigma=%.9g "
                "actual_model_evaluation=%s forecast=%s reason=%s "
                "adams_persistent=%s solver_value_source=%s continuum=%s",
                predicted_step_id,
                i,
                float(sigmas[i].detach().to(device="cpu").item()),
                int(predicted_actual),
                int(not predicted_actual),
                "predicted_actual_h3" if predicted_actual else "predicted_forecast_ephemeral",
                int(predicted_persistent),
                solver_value_source,
                int(continuum_active),
            )

        if sigmas[i + 1] == 0:
            x_pred = endpoint
            continue

        if not persistent_preds:
            raise RuntimeError("active PECE has no exact endpoint for its predictor")
        predictor_order_used = min(predictor_order_used, len(persistent_preds))
        tau_t = tau_func(sigmas[i + 1])
        curr_lambdas = torch.stack(
            persistent_lambdas[-predictor_order_used:]
        )
        b_coeffs = native_sa_solver.compute_stochastic_adams_b_coeffs(
            sigmas[i + 1],
            curr_lambdas,
            lambdas[i],
            lambdas[i + 1],
            tau_t,
            simple_order_2,
            is_corrector_step=False,
        )
        pred_mat = torch.stack(
            persistent_preds[-predictor_order_used:],
            dim=1,
        )
        pred_res = torch.tensordot(
            pred_mat,
            b_coeffs,
            dims=([1], [0]),
        )
        h = lambdas[i + 1] - lambdas[i]
        x_pred = (
            sigmas[i + 1]
            / sigmas[i]
            * (-(tau_t**2) * h).exp()
            * x
            + pred_res
        )
        if tau_t > 0 and s_noise > 0:
            noise = (
                noise_sampler(sigmas[i], sigmas[i + 1])
                * sigmas[i + 1]
                * (-2 * tau_t**2 * h).expm1().neg().sqrt()
                * s_noise
            )
            x_pred = x_pred + noise

    expected_last_step = 2 * (len(sigmas) - 1) - 2
    if runtime.last_completed_step_id not in {None, expected_last_step}:
        raise RuntimeError(
            "Spectrum PECE completed with an unexpected logical model-call count "
            f"(expected_last={expected_last_step}, "
            f"observed={runtime.last_completed_step_id})"
        )
    return x_pred


def _sample_sa_solver_forecast_isolated(
    runtime: SpectrumH3Runtime,
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool = False,
    tau_func: Any = None,
    s_noise: float = 1.0,
    noise_sampler: Any = None,
    predictor_order: int = 3,
    corrector_order: int = 4,
    use_pece: bool = False,
    simple_order_2: bool = False,
    continuum_active: bool = False,
) -> torch.Tensor:
    """Run reviewed SA equations without persisting forecasted denoisers in Adams history.

    Exact H3 evaluations are retained with their real lambda coordinates.
    Forecasted denoisers are ephemeral: they participate in the native current-step
    PEC corrector and predictor, but they are never inserted into persistent
    multistep history. The next exact evaluation therefore re-anchors both the
    Spectrum forecaster and the SA Adams stencil instead of recursively consuming
    old forecasts.

    Stochastic SA owns its predictor noise separately from the denoiser history.
    The exact H3 model can respond to the random increment already present in
    x_pred; a skipped transformer cannot reconstruct that response. During active
    stochastic intervals, a Spectrum feature forecast is therefore used only to
    complete the runtime transaction; SA itself consumes a bounded causal
    solver-space denoised prediction built from exact actual anchors.

    Continuum continuation chunks are a stricter regime because part of the H3
    target is an exact protected prefix while the remaining temporal region is
    generated. The first Continuum hold experiment removed secant extrapolation
    only inside SA's stochastic interval, but real media retained the same
    oscillation/flashing artifact. That run still used raw Spectrum denoisers at
    the non-stochastic forecast endpoints (steps 2 and 17 in the 19-step default
    schedule). Continuum therefore uses the newest exact denoised anchor as a
    zero-order dense output for every forecast coordinate, not only stochastic
    ones. This completely removes hidden-feature forecast values from the SA
    numerical trajectory for continuation chunks while preserving the 11/8 NFE
    budget. Native x_pred, stochastic noise, corrector equations, predictor
    equations, and RNG draws remain unchanged.
    """
    if use_pece and corrector_order > 0:
        return _sample_sa_solver_pece_forecast_isolated(
            runtime,
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            tau_func=tau_func,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            predictor_order=predictor_order,
            corrector_order=corrector_order,
            simple_order_2=simple_order_2,
            continuum_active=continuum_active,
        )
    if len(sigmas) <= 1:
        return x

    import comfy.k_diffusion.sa_solver as native_sa_solver
    import comfy.k_diffusion.sampling as native_sampling
    from tqdm.auto import trange

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = (
        native_sampling.default_noise_sampler(x, seed=seed)
        if noise_sampler is None
        else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    sigmas = native_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = native_sampling.sigma_to_half_log_snr(
        sigmas,
        model_sampling=model_sampling,
    )

    if tau_func is None:
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = native_sa_solver.get_tau_interval_func(
            start_sigma,
            end_sigma,
            eta=1.0,
        )

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x
    h = 0.0
    tau_t = 0.0
    noise = 0.0
    actual_preds: list[torch.Tensor] = []
    actual_lambdas: list[torch.Tensor] = []
    actual_steps: list[int] = []
    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        raw_denoised = model(
            x_pred,
            sigmas[i] * s_in,
            **extra_args,
        )

        mode = runtime.last_completed_mode
        if mode not in {"actual", "forecast"}:
            raise RuntimeError(
                "Spectrum SA adapter could not determine whether the H3 evaluation "
                f"at outer step {i} was actual or forecast"
            )
        actual = mode == "actual"
        denoised = raw_denoised
        solver_space_forecast = bool(
            not actual
            and (
                continuum_active
                or (tau_t > 0 and s_noise > 0)
            )
        )
        if solver_space_forecast:
            denoised, dense_mode, anchor_steps, dense_alpha = _sa_predict_causal_denoised(
                actual_preds,
                actual_lambdas,
                actual_steps,
                target_lambda=lambdas[i],
                target_step=i,
                continuum_active=continuum_active,
            )
            if runtime.config.debug:
                alpha_value = float(dense_alpha.detach().to(device="cpu").item())
                LOG.warning(
                    "Spectrum H3 SA dense output step=%s scope=%s mode=%s "
                    "anchor_steps=%s alpha=%.8f raw_feature_forecast=ignored "
                    "solver_history=actual_only",
                    i,
                    "continuum_all_forecasts" if continuum_active else "stochastic",
                    dense_mode,
                    anchor_steps,
                    alpha_value,
                )

        if callback is not None:
            callback(
                {
                    "x": x_pred,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        if actual:
            actual_preds.append(denoised)
            actual_lambdas.append(lambdas[i])
            actual_steps.append(i)
            if len(actual_preds) > max_used_order:
                actual_preds = actual_preds[-max_used_order:]
                actual_lambdas = actual_lambdas[-max_used_order:]
                actual_steps = actual_steps[-max_used_order:]

        stencil_preds = list(actual_preds)
        stencil_lambdas = list(actual_lambdas)
        if not actual:
            stencil_preds.append(denoised)
            stencil_lambdas.append(lambdas[i])
        if len(stencil_preds) > max_used_order:
            stencil_preds = stencil_preds[-max_used_order:]
            stencil_lambdas = stencil_lambdas[-max_used_order:]
        if not stencil_preds:
            raise RuntimeError("Spectrum SA adapter has no denoiser value for the active predictor")

        predictor_order_used = min(predictor_order, len(stencil_preds))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            # A forecast is not persistent Adams evidence, but it is still the
            # current SA endpoint estimate. Native PEC uses that endpoint in the
            # current corrector before constructing the next predictor state.
            # Dropping this correction causes the forecast error to appear one
            # step later in the next exact model input.
            corrector_order_used = min(corrector_order, len(stencil_preds))

        if lower_order_to_end:
            predictor_order_used = min(
                predictor_order_used,
                len(sigmas) - 2 - i,
            )
            corrector_order_used = min(
                corrector_order_used,
                len(sigmas) - 1 - i,
            )

        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = torch.stack(
                stencil_lambdas[-corrector_order_used:]
            )
            b_coeffs = native_sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(
                stencil_preds[-corrector_order_used:],
                dim=1,
            )
            corr_res = torch.tensordot(
                pred_mat,
                b_coeffs,
                dims=([1], [0]),
            )
            x = (
                sigmas[i]
                / sigmas[i - 1]
                * (-(tau_t**2) * h).exp()
                * x
                + corr_res
            )
            if tau_t > 0 and s_noise > 0:
                x = x + noise

        if sigmas[i + 1] == 0:
            x_pred = denoised
            continue

        tau_t = tau_func(sigmas[i + 1])
        curr_lambdas = torch.stack(
            stencil_lambdas[-predictor_order_used:]
        )
        b_coeffs = native_sa_solver.compute_stochastic_adams_b_coeffs(
            sigmas[i + 1],
            curr_lambdas,
            lambdas[i],
            lambdas[i + 1],
            tau_t,
            simple_order_2,
            is_corrector_step=False,
        )
        pred_mat = torch.stack(
            stencil_preds[-predictor_order_used:],
            dim=1,
        )
        pred_res = torch.tensordot(
            pred_mat,
            b_coeffs,
            dims=([1], [0]),
        )
        h = lambdas[i + 1] - lambdas[i]
        x_pred = (
            sigmas[i + 1]
            / sigmas[i]
            * (-(tau_t**2) * h).exp()
            * x
            + pred_res
        )

        if tau_t > 0 and s_noise > 0:
            noise = (
                noise_sampler(sigmas[i], sigmas[i + 1])
                * sigmas[i + 1]
                * (-2 * tau_t**2 * h).expm1().neg().sqrt()
                * s_noise
            )
            x_pred = x_pred + noise

    return x_pred


def _run_solver_aware_sa(
    executor: Any,
    runtime: SpectrumH3Runtime,
    model_wrap: Any,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any],
    callback: Any,
    noise: torch.Tensor,
    latent_image: Any,
    denoise_mask: Any,
    disable_pbar: bool,
) -> torch.Tensor:
    """Execute SA through the reviewed isolated-history adapter."""
    sampler = executor.class_obj
    name = sampler_name(sampler)
    if len(executor.wrappers) != 1:
        runtime.disable_forecasting_for_run(
            "another SAMPLER_SAMPLE wrapper makes SA isolated-history ordering unproven"
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    options = dict(getattr(sampler, "extra_options", {}) or {})
    copied, copy_failure = _copy_ksampler_with_options(sampler, options)
    if copy_failure is not None or copied is None:
        runtime.disable_forecasting_for_run(
            copy_failure or "SA KSAMPLER copy failed"
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    use_pece = bool(
        name == "sample_sa_solver_pece"
        or options.get("use_pece", False) is True
    )
    continuum_active = _continuum_actual_prefix(
        (extra_args or {}).get("model_options")
    ) > 0
    active_pece = bool(use_pece and int(options.get("corrector_order", 4)) > 0)

    def spectrum_sa_function(
        model,
        x,
        run_sigmas,
        extra_args=None,
        callback=None,
        disable=False,
        **run_options,
    ):
        # Native sample_sa_solver exposes use_pece through KSAMPLER extra_options.
        # Consume that reviewed option exactly once before forwarding solver kwargs.
        run_options = dict(run_options)
        forwarded_use_pece = run_options.pop("use_pece", use_pece)
        if type(forwarded_use_pece) is not bool or forwarded_use_pece != use_pece:
            raise RuntimeError(
                "SA isolated-history adapter observed a changed use_pece option"
            )

        refdelta_state = None
        if name in REFDELTA_SA_SOLVER_SAMPLERS:
            if use_pece:
                raise RuntimeError(
                    "RefDelta SA backend is PEC-only but reached an active PECE adapter"
                )
            config = run_options.pop("config", None)
            if config is None:
                raise RuntimeError("RefDelta SA adapter lost its validated config")
            if not config.is_native_equivalence_mode:
                from comfyui_refdelta_solver.sampler_backends import (
                    prepare_refdelta_backend_adapters,
                )

                refdelta_state, model, adapted_noise = (
                    prepare_refdelta_backend_adapters(
                        model,
                        x,
                        run_sigmas,
                        extra_args or {},
                        config,
                        "sa_solver",
                        1,
                        run_options.get("noise_sampler"),
                    )
                )
                run_options["noise_sampler"] = adapted_noise
        try:
            return _sample_sa_solver_forecast_isolated(
                runtime,
                model,
                x,
                run_sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                use_pece=use_pece,
                continuum_active=continuum_active,
                **run_options,
            )
        finally:
            if refdelta_state is not None:
                refdelta_state.finish()

    spectrum_sa_function.__name__ = f"{name}_spectrum_isolated_history"
    copied.sampler_function = spectrum_sa_function
    if getattr(sampler, "sampler_function", None) is getattr(
        copied,
        "sampler_function",
        None,
    ):
        raise RuntimeError("SA isolated-history adapter mutated the original sampler")

    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 SA adapter mode=isolated_adams_history "
            "topology=%s persistent=actual_only "
            "predicted_forecast=current_corrector_ephemeral "
            "corrected_phase=%s "
            "stochastic_forecast=solver_space_causal_dense_output "
            "continuum_dense_mode=%s continuum_dense_scope=%s",
            "predicted+corrected" if active_pece else "predicted",
            "exact_reanchor" if active_pece else "none",
            "latest_actual_hold" if continuum_active else "lambda_bounded_extrapolation",
            "all_forecasts" if continuum_active else "stochastic_only",
        )
    return copied.sample(
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image,
        denoise_mask,
        disable_pbar,
    )


def _refdelta_sampler_contract(
    function: Any,
    options: dict[str, Any],
) -> tuple[bool, str | None, bool]:
    """Validate RefDelta API v1 and return whether exact-q publication is required."""
    try:
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde
        from comfyui_refdelta_solver.spectrum_interop import (
            SPECTRUM_INTEROP_CONTRACT,
        )
    except (ImportError, AttributeError) as exc:
        return False, f"RefDelta interop API is unavailable: {exc}", False

    if function is not sample_refdelta_er_sde:
        return False, "sampler function is not the installed RefDelta implementation", False
    if SPECTRUM_INTEROP_CONTRACT != REFDELTA_INTEROP_CONTRACT:
        return False, "RefDelta/Spectrum interop contract version does not match", False
    if (
        getattr(function, "__spectrum_interop_contract__", None)
        != REFDELTA_INTEROP_CONTRACT
    ):
        return False, "RefDelta sampler did not publish the reviewed interop contract", False
    unknown = set(options) - REFDELTA_TRACKED_OPTIONS
    if unknown:
        return False, f"RefDelta sampler has unreviewed options: {sorted(unknown)}", False
    config = options.get("config")
    if type(config) is not RefDeltaSamplerConfig:
        return False, "RefDelta sampler config has unreviewed provenance", False
    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        return False, f"RefDelta sampler config is invalid: {exc}", False
    return True, None, not config.is_native_equivalence_mode


def _er_sde_tracking_contract(sampler: Any) -> tuple[bool, str | None]:
    """Accept only the native or versioned RefDelta solver semantics reviewed here."""
    import comfy.k_diffusion.sampling as native_sampling
    import comfy.samplers

    function = getattr(sampler, "sampler_function", None)
    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False, "ER-SDE sampler options are not a dictionary"
    if sampler_name(sampler) == REFDELTA_SAMPLER_NAME:
        accepted, reason, _ = _refdelta_sampler_contract(function, options)
        if not accepted:
            return False, reason
    else:
        if function is not native_sampling.sample_er_sde:
            return False, "sampler function is not native ComfyUI sample_er_sde"
        if function_ast_digest(function) != ER_SDE_NATIVE_FUNCTION_DIGEST:
            return False, "native sample_er_sde implementation is not a reviewed revision"
    if type(sampler) is not comfy.samplers.KSAMPLER:
        return False, "ER-SDE sampler object is not native ComfyUI KSAMPLER"
    if (
        comfy.samplers.KSAMPLER.__module__ != "comfy.samplers"
        or comfy.samplers.KSAMPLER.__name__ != "KSAMPLER"
    ):
        return False, "native ComfyUI KSAMPLER class provenance is unreviewed"
    if (
        comfy.samplers.KSamplerX0Inpaint.__module__ != "comfy.samplers"
        or comfy.samplers.KSamplerX0Inpaint.__name__ != "KSamplerX0Inpaint"
    ):
        return False, "native KSamplerX0Inpaint adapter provenance is unreviewed"
    sample_contract = _ksampler_sample_contract(sampler)
    if not sample_contract.accepted:
        return False, (
            f"KSAMPLER.sample contract rejected: {sample_contract.failure}; "
            f"{sample_contract.provenance.log_fields()}"
        )
    _, copy_failure = _copy_ksampler_with_options(sampler, {})
    if copy_failure is not None:
        return False, f"KSAMPLER copy contract rejected: {copy_failure}"
    if (
        function_ast_digest(native_sampling.default_noise_sampler)
        != ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST
    ):
        return False, "native default_noise_sampler implementation is not a reviewed revision"

    reviewed_options = (
        REFDELTA_TRACKED_OPTIONS
        if sampler_name(sampler) == REFDELTA_SAMPLER_NAME
        else ER_SDE_TRACKED_OPTIONS
    )
    unknown = set(options) - reviewed_options
    if unknown:
        return False, f"ER-SDE sampler has unreviewed options: {sorted(unknown)}"
    try:
        s_noise = float(options.get("s_noise", 1.0))
    except (TypeError, ValueError):
        return False, "ER-SDE s_noise is not numeric"
    if not torch.isfinite(torch.tensor(s_noise)) or s_noise < 0.0:
        return False, "ER-SDE s_noise must be finite and nonnegative"
    max_stage = options.get("max_stage", 3)
    if isinstance(max_stage, bool) or not isinstance(max_stage, int) or not 1 <= max_stage <= 3:
        return False, "ER-SDE max_stage must be an integer in [1, 3]"
    if s_noise > 0.0:
        if options.get("noise_sampler") is not None:
            return False, "custom ER-SDE noise_sampler provenance is unreviewed"
        if not _er_sde_noise_scaler_supports_replay(options.get("noise_scaler")):
            return False, "custom ER-SDE noise_scaler provenance is unreviewed"
    return True, None


def _native_er_sde_preflight_reason(
    sampler: Any,
    model_options: dict[str, Any] | None,
) -> str | None:
    """Resolve reviewed ER-SDE-family failures before retaining Spectrum state."""
    function = getattr(sampler, "sampler_function", None)
    if (
        sampler_name(sampler) != REFDELTA_SAMPLER_NAME
        and getattr(function, "__module__", None) != "comfy.k_diffusion.sampling"
    ):
        return None
    supported, reason = _er_sde_tracking_contract(sampler)
    if not supported:
        return reason or "native ER-SDE compensation contract is unproven"
    options = getattr(sampler, "extra_options", {}) or {}
    if float(options.get("s_noise", 1.0)) == 0.0:
        return None

    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        model_options or {},
        is_model_options=True,
    )
    if len(wrappers) != 1 or wrappers[0] is not sampler_sample_wrapper:
        return "another SAMPLER_SAMPLE wrapper makes ER-SDE ordering unproven"
    return None


def max_consecutive_forecasts(sampler: Any) -> int | None:
    name = sampler_name(sampler)
    if name in SA_SOLVER_SAMPLERS and _sa_solver_is_stochastic(sampler):
        # Native SA stores every denoised result in pred_list and feeds that
        # history back into both its corrector and predictor. Real H3 media
        # validation showed that a multi-forecast tail can therefore become
        # self-referential and catastrophically corrupt x_pred. Keep stochastic
        # SA forecasts isolated so every skipped denoiser is bracketed by exact
        # solver observations instead of allowing a forecast-only Adams window.
        return 1
    return 1 if sampler_is_supported(sampler) else None


def min_actual_steps_after_forecast(sampler: Any) -> int:
    name = sampler_name(sampler)
    if name in SA_SOLVER_SAMPLERS:
        if _sa_solver_is_active_pece(sampler):
            # Every forecastable predicted phase is followed by an exact
            # corrected-state H3 reanchor in the same outer interval. The
            # predicted feature lane still enforces its own one-anchor refresh.
            return 0
        if _sa_solver_is_stochastic(sampler):
            return 0
        options = getattr(sampler, "extra_options", {}) or {}
        if isinstance(options, dict):
            predictor_order = options.get("predictor_order", 3)
            corrector_order = options.get("corrector_order", 4)
            if type(predictor_order) is int and type(corrector_order) is int:
                return max(1, predictor_order, corrector_order)
        return 4
    return 1 if name in SUPPORTED_SAMPLERS else 0


def min_tail_actual_steps(sampler: Any) -> int:
    return 3 if sampler_name(sampler) in RES_MULTISTEP_SAMPLERS else 0


def _binding_from_model_options(
    model_options: dict[str, Any] | None,
) -> SpectrumH3Binding | None:
    binding = (model_options or {}).get(BINDING_KEY)
    return binding if isinstance(binding, SpectrumH3Binding) else None


def _copy_condition_structure(value: Any) -> Any:
    """Copy mutable conditioning containers while sharing tensor/model payloads."""
    if isinstance(value, dict):
        return {key: _copy_condition_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_condition_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_condition_structure(item) for item in value)
    return value


def _offline_progress_callbacks(callback, total_steps: int):
    """Report both passes while keeping previews and callback side effects replay-only."""
    if callback is None or total_steps <= 0:
        return None, callback, None

    import comfy.utils

    total_work = total_steps * 2
    progress = comfy.utils.ProgressBar(total_work)
    replay_finished = False

    def capture_callback(step, _x0, _x, _pass_steps):
        progress.update_absolute(step + 1, total_work)

    def replay_callback(step, x0, x, _pass_steps):
        nonlocal replay_finished
        callback(total_steps + step, x0, x, total_work)
        replay_finished = step + 1 >= total_steps

    def complete_progress():
        if not replay_finished:
            progress.update_absolute(total_work, total_work)

    return capture_callback, replay_callback, complete_progress


def copy_model_options_with_step(
    model_options: dict[str, Any] | None,
    runtime: SpectrumH3Runtime,
    decision: dict[str, Any],
) -> dict[str, Any]:
    copied = dict(model_options or {})
    transformer_options = dict(copied.get("transformer_options") or {})
    copied["transformer_options"] = transformer_options
    transformer_options[RUNTIME_KEY] = runtime
    transformer_options[RUN_ID_KEY] = int(decision["run_id"])
    transformer_options[STEP_ID_KEY] = int(decision["step_id"])
    transformer_options[OUTER_STEP_ID_KEY] = int(decision["policy_step_id"])
    transformer_options[SOLVER_PHASE_KEY] = str(decision["phase"])
    transformer_options[COORDINATE_KEY] = float(decision["coordinate"])
    transformer_options[ACTUAL_KEY] = bool(decision["actual"])
    transformer_options[REASON_KEY] = str(decision["reason"])
    return copied


def outer_sample_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None:
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    transformer_options = (
        (getattr(guider, "model_options", None) or {}).get("transformer_options")
        or {}
    )
    if transformer_options.get("easycache") is not None:
        LOG.warning(
            "Spectrum H3 disabled for this run because EasyCache or LazyCache is active on the same model"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    runtime = binding.runtime
    name = sampler_name(sampler)
    if (
        (name == REFDELTA_SAMPLER_NAME or name in REFDELTA_BACKEND_SAMPLERS)
        and "multigpu_clones" in (getattr(guider, "model_options", None) or {})
    ):
        LOG.warning(
            "Spectrum H3 disabled for this RefDelta backend run because multi-GPU "
            "parallel model calls bypass transactional step finalization; running "
            "the untouched RefDelta sampler"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    expected_model_calls = None
    sa_call_topology = None
    if name in SA_SOLVER_SAMPLERS:
        preflight_reason = _native_sa_solver_preflight_reason(
            sampler,
            getattr(guider, "model_options", None),
        )
        expected_model_calls = _sa_solver_expected_model_calls(sampler, sigmas)
        sa_call_topology = _sa_solver_call_topology(sampler, sigmas)
        if (
            preflight_reason is not None
            or expected_model_calls is None
            or sa_call_topology is None
        ):
            reason = preflight_reason or "SA-Solver model-call topology is unavailable"
            LOG.warning(
                "Spectrum H3 disabled for this SA-Solver run; running the untouched "
                "native sampler because its predictor/corrector contract is unavailable: %s",
                reason,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
    if name in SEEDS_SAMPLERS:
        preflight_reason = _native_seeds_preflight_reason(
            sampler,
            getattr(guider, "model_options", None),
        )
        if preflight_reason is None:
            preflight_reason = _seeds_stage_schedule_reason(sigmas)
        if preflight_reason is not None:
            LOG.warning(
                "Spectrum H3 disabled for this SEEDS run; running the untouched "
                "native sampler: %s",
                preflight_reason,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        expected_model_calls = _seeds_expected_model_calls(sampler, sigmas)
        if expected_model_calls is None:
            LOG.warning(
                "Spectrum H3 disabled for this SEEDS run; running the untouched "
                "native sampler because the model-call topology could not be resolved"
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
    if name in ER_SDE_SAMPLERS:
        preflight_reason = _native_er_sde_preflight_reason(
            sampler,
            getattr(guider, "model_options", None),
        )
        if preflight_reason is not None:
            LOG.warning(
                "Spectrum H3 disabled for this ER-SDE run; running the untouched "
                "native sampler because stochastic compensation is unavailable: %s",
                preflight_reason,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
    continuum_prefix = _continuum_actual_prefix(getattr(guider, "model_options", None))
    continuum_log_emitted = False
    stochastic_seeds = name in SEEDS_SAMPLERS and _seeds_is_stochastic(sampler)
    stochastic_sa = name in SA_SOLVER_SAMPLERS and _sa_solver_is_stochastic(sampler)
    active_pece = name in SA_SOLVER_SAMPLERS and _sa_solver_is_active_pece(sampler)
    state_conditioned_residual = stochastic_seeds or stochastic_sa or active_pece
    sa_forced_actual_steps: tuple[int, ...] = ()
    sa_stochastic_input_steps: tuple[int, ...] = ()
    if name in SA_SOLVER_SAMPLERS:
        try:
            model_sampling = guider.model_patcher.get_model_object("model_sampling")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            LOG.warning(
                "Spectrum H3 disabled for this SA-Solver run; running the untouched "
                "native sampler because model_sampling is unavailable: %s",
                exc,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        protected, stochastic_steps, protection_reason = _sa_solver_stochastic_protection(
            sampler,
            sigmas,
            model_sampling,
        )
        if protection_reason is not None or protected is None or stochastic_steps is None:
            reason = protection_reason or "SA-Solver stochastic protection is unavailable"
            LOG.warning(
                "Spectrum H3 disabled for this SA-Solver run; running the untouched "
                "native sampler because its stochastic-state schedule is unavailable: %s",
                reason,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        sa_stochastic_input_steps = stochastic_steps
        sa_forced_actual_steps = protected
    profile_eligible = sampler_is_supported(sampler) and (
        not runtime.config.offline_smoothing_replay
        or sampler_supports_seeded_replay(sampler)
        or stochastic_seeds
        or name in REFDELTA_SEEDS_SAMPLERS
        or name in SA_SOLVER_SAMPLERS
    )
    if runtime.config.model_aware_mode != "off" and profile_eligible:
        try:
            lookup = get_model_forecastability_profile(guider.model_patcher)
            runtime.set_model_profile(lookup)
            if runtime.config.debug:
                profile = lookup.profile
                LOG.warning(
                    "Spectrum H3 model-aware profile base=%s patches=%s patch_keys=%s "
                    "recognized_lora=%s unknown_patches=%s cache=%s build_s=%.6f "
                    "lookup_s=%.6f memory_bytes=%s sensitivity=%.6f perturbation=%.6f "
                    "final_perturbation=%.6f confidence=%.6f "
                    "profile_payload=compact_scalar_sensitivities retained_head_bytes=0",
                    profile.base_model_identity,
                    profile.active_patch_count,
                    profile.active_patch_keys,
                    profile.recognized_lora_count,
                    profile.unknown_patch_count,
                    "hit" if lookup.cache_hit else "miss",
                    profile.build_seconds,
                    lookup.lookup_seconds,
                    profile.estimated_bytes,
                    profile.aggregate_sensitivity,
                    profile.patch_perturbation,
                    profile.final_block_perturbation,
                    profile.profile_confidence,
                )
        except torch.cuda.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            runtime.disable_model_aware(f"profile construction failed: {exc}")
            LOG.warning(
                "Spectrum H3 model-aware forecasting unavailable for this run: %s",
                exc,
            )

    def execute_run(
        run_noise,
        run_latent,
        run_sigmas,
        run_mask,
        run_callback,
        run_disable_pbar,
        *,
        phase: str,
    ):
        nonlocal continuum_log_emitted
        phase_prefix = _continuum_prefix_for_phase(continuum_prefix, phase)
        if name in SEEDS_SAMPLERS and expected_model_calls is not None and not stochastic_seeds:
            expanded_prefix = _seeds_prefix_model_calls(
                sampler,
                run_sigmas,
                phase_prefix,
            )
            if expanded_prefix is None:
                raise RuntimeError("SEEDS prefix model-call topology became unavailable")
            phase_prefix = expanded_prefix
        pece_history_step_ids = (
            tuple(
                logical_step
                for logical_step, descriptor in enumerate(sa_call_topology or ())
                if descriptor.phase == "corrected"
                or (
                    descriptor.outer_step == 0
                    and descriptor.phase == "predicted"
                )
            )
            if active_pece
            else None
        )
        run_id = runtime.start_run(
            run_sigmas,
            name,
            supported_sampler=sampler_is_supported(sampler),
            max_consecutive_forecasts=max_consecutive_forecasts(sampler),
            min_actual_steps_after_forecast=min_actual_steps_after_forecast(sampler),
            min_tail_actual_steps=min_tail_actual_steps(sampler),
            min_actual_prefix_steps=phase_prefix,
            expected_model_calls=expected_model_calls,
            stage_count=(
                2 if active_pece else SEEDS_STAGE_COUNTS.get(name, 1)
            ),
            logical_call_topology=sa_call_topology,
            state_conditioned_residual=state_conditioned_residual,
            separate_stage_histories=(
                False if active_pece or stochastic_seeds else None
            ),
            forecastable_stage_indices=(
                (0,)
                if active_pece
                else tuple(range(1, SEEDS_STAGE_COUNTS[name]))
                if stochastic_seeds
                else None
            ),
            history_stage_indices=(0, 1) if active_pece else None,
            history_step_ids=pece_history_step_ids,
            tail_actual_stage_indices=(1,) if active_pece else None,
            allow_state_conditioned_bootstrap=active_pece,
            forced_actual_step_ids=(
                sa_forced_actual_steps if name in SA_SOLVER_SAMPLERS else None
            ),
            forced_actual_steps_advance_window=bool(
                name in SA_SOLVER_SAMPLERS and stochastic_sa
            ),
            model_aware_can_force_actual=not (
                stochastic_seeds or stochastic_sa or active_pece
            ),
        )
        if phase_prefix > 0 and runtime.supported_sampler and not continuum_log_emitted:
            LOG.warning(
                "Spectrum H3: accepted H3 Continuum API v1, actual prefix=%s",
                min(phase_prefix, runtime.stats.total_steps),
            )
            continuum_log_emitted = True
        if name in ER_SDE_SAMPLERS and (
            runtime.config.anchor_residual_feedback
            or runtime.config.selective_rollback_correction
        ):
            runtime.disable_experiment(
                "ER-SDE is reviewed only for ordinary Spectrum and offline smoothing replay"
            )
        if state_conditioned_residual and (
            runtime.config.anchor_residual_feedback
            or runtime.config.selective_rollback_correction
        ):
            runtime.disable_experiment(
                "state-conditioned sampler forecasting is not reviewed with "
                "rollback/residual experiments"
            )
        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 run start phase=%s run_id=%s sampler=%s steps=%s supported=%s "
                "seeds_stochastic=%s sa_stochastic=%s sa_active_pece=%s "
                "stage_count=%s feature_geometry=%s "
                "stage_histories=%s sa_stochastic_input_steps=%s "
                "sa_forced_actual_steps=%s sa_post_forecast_refresh=%s "
                "sa_max_forecast_streak=%s sa_exact_window_credit=%s "
                "sa_adams_history=%s sa_feature_history=%s "
                "model_aware_force_actual=%s",
                phase,
                run_id,
                name,
                runtime.stats.total_steps,
                runtime.supported_sampler,
                stochastic_seeds,
                stochastic_sa,
                active_pece,
                2 if active_pece else SEEDS_STAGE_COUNTS.get(name, 1),
                (
                    "state_conditioned_residual_endpoint_history"
                    if active_pece
                    else "state_conditioned_residual_shared_history"
                    if stochastic_seeds
                    else "state_conditioned_residual"
                    if state_conditioned_residual
                    else "absolute_hidden"
                ),
                "shared_endpoint"
                if active_pece
                else "shared"
                if stochastic_seeds
                else "single",
                sa_stochastic_input_steps,
                sa_forced_actual_steps,
                min_actual_steps_after_forecast(sampler) if name in SA_SOLVER_SAMPLERS else 0,
                max_consecutive_forecasts(sampler) if name in SA_SOLVER_SAMPLERS else 0,
                bool(name in SA_SOLVER_SAMPLERS and stochastic_sa),
                (
                    "actual_only_persistent+ephemeral_forecast"
                    if name in SA_SOLVER_SAMPLERS
                    else "-"
                ),
                (
                    "initial_predicted+corrected_actual_endpoints"
                    if active_pece
                    else "-"
                ),
                not (stochastic_seeds or stochastic_sa or active_pece),
            )
        started = time.perf_counter()
        try:
            result = executor(
                run_noise,
                run_latent,
                sampler,
                run_sigmas,
                run_mask,
                run_callback,
                run_disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
            runtime.log_offline_transition(
                "first_pass_executor_return" if phase == "offline_first_pass" else "executor_return",
                phase=phase,
                run_id=run_id,
                elapsed_s=f"{time.perf_counter() - started:.6f}",
            )
            return result
        finally:
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 run summary phase=%s wall_s=%.3f %s",
                    phase,
                    time.perf_counter() - started,
                    runtime.debug_summary(),
                )
            runtime.end_run(run_id)
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 run teardown phase=%s run_id=%s", phase, run_id
                )

    if not runtime.config.offline_smoothing_replay:
        result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass",
        )
        return result

    if not sampler_is_supported(sampler):
        LOG.warning(
            "Spectrum H3 offline smoothing replay is unsupported for sampler %s; running one native pass",
            name,
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not sampler_supports_seeded_replay(sampler):
        if name in REFDELTA_SEEDS_SAMPLERS:
            LOG.warning(
                "Spectrum H3 offline smoothing replay is disabled for RefDelta SEEDS "
                "because actual-only RefDelta evidence is defined on the live causal pass; "
                "running one causal state-conditioned Spectrum pass"
            )
            return execute_run(
                noise,
                latent_image,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                phase="single_pass_replay_fallback",
            )
        if stochastic_seeds:
            LOG.warning(
                "Spectrum H3 offline smoothing replay is disabled for stochastic SEEDS "
                "because replay must not replace its Markov-preserving noise-conditioned "
                "stage evaluations; running one causal state-conditioned Spectrum pass"
            )
            return execute_run(
                noise,
                latent_image,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                phase="single_pass_replay_fallback",
            )
        if name in SA_SOLVER_SAMPLERS:
            LOG.warning(
                "Spectrum H3 offline smoothing replay is disabled for SA-Solver because "
                "replaying changed denoiser values would change native Adams history; "
                "running one causal Spectrum pass"
            )
            return execute_run(
                noise,
                latent_image,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                phase="single_pass_replay_fallback",
            )
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires ER-SDE's native seeded "
            "noise_sampler and noise_scaler; running one native pass"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not all(torch.is_tensor(value) for value in (noise, latent_image, sigmas)):
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires tensor sampling inputs; running one ordinary pass"
        )
        result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass_fallback",
        )
        return result

    replay_noise = noise.detach().clone()
    replay_latent = latent_image.detach().clone()
    replay_sigmas = sigmas.detach().clone()
    replay_mask = (
        denoise_mask.detach().clone()
        if torch.is_tensor(denoise_mask)
        else denoise_mask
    )
    initial_conds = (
        _copy_condition_structure(guider.conds) if hasattr(guider, "conds") else None
    )
    offline_steps = max(0, sigmas.numel() - 1)
    offline_logical_steps = (
        offline_steps if expected_model_calls is None else expected_model_calls
    )
    capture_callback, replay_callback, complete_progress = _offline_progress_callbacks(
        callback,
        offline_steps,
    )
    runtime.begin_offline_capture(
        total_steps=offline_logical_steps,
        sampler_name=name,
    )
    try:
        first_result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            capture_callback,
            disable_pbar,
            phase="offline_first_pass",
        )
        try:
            capture_complete = runtime.complete_offline_capture()
        except Exception as exc:  # noqa: BLE001 - preserve completed first-pass output
            archive = runtime.offline_archive
            if archive is not None:
                archive.invalidate(f"offline capture completion failed: {exc}")
            capture_complete = False
            runtime.log_offline_transition(
                "complete_offline_capture_failed",
                error_type=type(exc).__name__,
                reason=exc,
            )
            LOG.warning(
                "Spectrum H3 offline capture completion failed; preserving the valid "
                "first-pass result: %s",
                exc,
            )
        if not capture_complete:
            if complete_progress is not None:
                complete_progress()
            reason = (
                runtime.offline_archive.failure_reason
                if runtime.offline_archive is not None
                else "offline archive was not retained"
            )
            LOG.warning(
                "Spectrum H3 offline replay skipped; returning the valid first-pass result: %s",
                reason,
            )
            return first_result

        try:
            runtime.begin_offline_replay()
        except Exception as exc:  # noqa: BLE001 - preserve completed first-pass output
            runtime.log_offline_transition(
                "begin_offline_replay_failed",
                error_type=type(exc).__name__,
                reason=exc,
            )
            if complete_progress is not None:
                complete_progress()
            LOG.warning(
                "Spectrum H3 offline replay setup failed; returning the valid first-pass "
                "result: %s",
                exc,
            )
            return first_result
        if initial_conds is not None:
            guider.conds = _copy_condition_structure(initial_conds)
        try:
            replay_result = execute_run(
                replay_noise,
                replay_latent,
                replay_sigmas,
                replay_mask,
                replay_callback,
                True,
                phase="offline_replay",
            )
            if complete_progress is not None:
                complete_progress()
        except OfflineReplayAbort as exc:
            if complete_progress is not None:
                complete_progress()
            LOG.warning(
                "Spectrum H3 offline replay aborted; returning the valid first-pass result: %s",
                exc,
            )
            return first_result
        return replay_result
    finally:
        runtime.release_offline_archive()


def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if (
        binding is None
        or binding.runtime.active_run_id is None
        or not binding.runtime.supported_sampler
    ):
        return executor(x, timestep, model_options or {}, seed)

    if "multigpu_clones" in (model_options or {}):
        if binding.runtime.config.debug:
            LOG.warning(
                "Spectrum H3 native fallback: multi-GPU parallel model calls are not transactionally supported"
            )
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    decision = runtime.begin_step(timestep)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 step run_id=%s step=%s outer_step=%s stage=%s phase=%s "
            "coordinate=%.6f decision=%s reason=%s history=%s window=%.3f",
            decision["run_id"],
            decision["step_id"],
            decision["policy_step_id"],
            decision["stage_index"],
            decision["phase"],
            decision["coordinate"],
            "actual" if decision["actual"] else "forecast",
            decision["reason"],
            runtime.prediction_history_length,
            runtime.stats.current_window,
        )
        model_decision = runtime.active_model_aware_decision
        if model_decision is not None:
            audio_gain = model_decision.audio_correction_telemetry
            video_gain = model_decision.video_correction_telemetry
            LOG.warning(
                "Spectrum H3 model-aware step=%s trajectory_risk=%.6f model_risk=%.6f "
                "patch_risk=%.6f combined_risk=%.6f confidence=%.6f horizon=%.3f "
                "degree=%s ridge=%.8f audio_blend=%.6f video_blend=%.6f "
                "audio_generic_projection=%.6f audio_raw_generic_gain=%.6f "
                "audio_generic_gain=%.6f audio_applied_gain=%.6f "
                "audio_generic_bound_active=%s "
                "video_generic_projection=%.6f video_raw_generic_gain=%.6f "
                "video_generic_gain=%.6f video_applied_gain=%.6f "
                "video_generic_bound_active=%s model_informed_correction=retired decision=%s",
                decision["step_id"],
                model_decision.trajectory_risk,
                model_decision.model_risk,
                model_decision.patch_risk,
                model_decision.combined_risk,
                model_decision.confidence,
                model_decision.forecast_horizon,
                model_decision.degree,
                model_decision.ridge_lambda,
                model_decision.audio_blend_weight,
                model_decision.video_blend_weight,
                audio_gain.residual_projection,
                audio_gain.raw_generic_gain,
                audio_gain.generic_gain,
                model_decision.audio_correction_gain,
                audio_gain.generic_bound_active,
                video_gain.residual_projection,
                video_gain.raw_generic_gain,
                video_gain.generic_gain,
                model_decision.video_correction_gain,
                video_gain.generic_bound_active,
                "ACTUAL" if decision["actual"] else "FORECAST",
            )

    def execute_attempt(attempt_decision: dict[str, Any]):
        patched = copy_model_options_with_step(model_options, runtime, attempt_decision)
        return executor(x, timestep, patched, seed)

    def consume_er_sde_increment(
        result: torch.Tensor,
        attempt_decision: dict[str, Any],
    ) -> torch.Tensor:
        transformer_options = (model_options or {}).get("transformer_options") or {}
        tracker = transformer_options.get(ER_SDE_TRACKER_KEY)
        bridge = transformer_options.get(REFDELTA_BRIDGE_KEY)
        if not isinstance(tracker, ERSDEStochasticTracker) and not isinstance(
            bridge, RefDeltaInteropBridge
        ):
            return result
        descriptor = None
        try:
            descriptor = runtime.describe_current_er_sde_step(
                int(attempt_decision["run_id"]),
                int(attempt_decision["step_id"]),
            )
            if isinstance(tracker, ERSDEStochasticTracker):
                result = tracker.consume(result, descriptor)
            if isinstance(bridge, RefDeltaInteropBridge):
                bridge.note_model_result(descriptor)
            return result
        except (ERSDETrackingError, RefDeltaInteropError) as exc:
            if isinstance(tracker, ERSDEStochasticTracker):
                tracker.clear()
            if isinstance(bridge, RefDeltaInteropBridge):
                bridge.clear()
            if descriptor is not None and descriptor.mode == "replay":
                raise OfflineReplayAbort(
                    f"ER-SDE stochastic-state compensation failed during replay: {exc}"
                ) from exc
            if not bool(attempt_decision["actual"]):
                raise ForecastRetryActual(
                    f"ER-SDE stochastic-state compensation failed: {exc}"
                ) from exc
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 ER-SDE tracker discarded on state-aware actual "
                    "step=%s reason=%s",
                    attempt_decision["step_id"],
                    exc,
                )
            return result

    try:
        try:
            result = execute_attempt(decision)
            runtime.log_offline_transition(
                "actual_executor_return" if decision["actual"] else "forecast_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
            )
            result = consume_er_sde_increment(result, decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
        except ForecastRetryActual as retry:
            runtime.prepare_actual_retry(
                decision["run_id"], decision["step_id"], str(retry)
            )
            retry_decision = dict(decision)
            retry_decision["actual"] = True
            retry_decision["reason"] = f"forecast transaction retry: {retry}"
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 forecast retry run_id=%s step=%s reason=%s",
                    decision["run_id"],
                    decision["step_id"],
                    retry,
                )
            result = execute_attempt(retry_decision)
            runtime.log_offline_transition(
                "actual_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
                retry=True,
            )
            result = consume_er_sde_increment(result, retry_decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


def _run_tracked_er_sde(
    executor,
    runtime: SpectrumH3Runtime,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image,
    denoise_mask,
    disable_pbar,
):
    supported, reason = _er_sde_tracking_contract(executor.class_obj)
    if len(executor.wrappers) != 1:
        supported = False
        reason = "another SAMPLER_SAMPLE wrapper makes ER-SDE ordering unproven"
    if not supported:
        assert reason is not None
        runtime.disable_forecasting_for_run(reason)
        fallback = (
            "RefDelta all-actual"
            if sampler_name(executor.class_obj) == REFDELTA_SAMPLER_NAME
            else "native all-actual"
        )
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving %s "
            "sampling because stochastic compensation is unavailable: %s",
            fallback,
            reason,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    sampler = executor.class_obj
    options = dict(getattr(sampler, "extra_options", {}) or {})
    refdelta_external_increment = False
    if sampler_name(sampler) == REFDELTA_SAMPLER_NAME:
        accepted, contract_reason, refdelta_external_increment = (
            _refdelta_sampler_contract(sampler.sampler_function, options)
        )
        if not accepted:
            raise AssertionError(contract_reason or "RefDelta contract changed after preflight")
    configured_s_noise = float(options.get("s_noise", 1.0))
    model_sampling = model_wrap.model_patcher.get_model_object("model_sampling")
    noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
    effective_s_noise = configured_s_noise * noise_scale
    if not torch.isfinite(torch.tensor(effective_s_noise)) or effective_s_noise < 0.0:
        reason = "effective ER-SDE s_noise is not finite and nonnegative"
        runtime.disable_forecasting_for_run(reason)
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving all-actual "
            "sampling because %s",
            reason,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if effective_s_noise == 0.0 and not refdelta_external_increment:
        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 ER-SDE compensation q_pending=false applied=false "
                "reason=no_stochastic_increment max_stage=%s stochastic=false",
                options.get("max_stage", 3),
            )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if effective_s_noise > 0.0 and not torch.is_tensor(noise):
        reason = "ER-SDE sampler noise is not a packed tensor"
        runtime.disable_forecasting_for_run(reason)
        LOG.warning("Spectrum H3 disabled for this ER-SDE run: %s", reason)
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    tracker = None
    tracked_sampler = sampler
    copy_failure = None
    if effective_s_noise > 0.0:
        import comfy.k_diffusion.sampling as native_sampling

        base_noise_sampler = native_sampling.default_noise_sampler(
            noise,
            seed=extra_args.get("seed"),
        )
        base_noise_scaler = options.get("noise_scaler")
        if base_noise_scaler is None:
            base_noise_scaler = native_default_er_sde_noise_scaler
        tracker = ERSDEStochasticTracker(
            noise_sampler=base_noise_sampler,
            noise_scaler=base_noise_scaler,
            effective_s_noise=effective_s_noise,
            max_stage=int(options.get("max_stage", 3)),
            debug=runtime.config.debug,
            run_id=int(runtime.active_run_id),
            external_increment=refdelta_external_increment,
        )
        tracked_options = dict(options)
        tracked_options["noise_sampler"] = tracker.noise_sampler
        tracked_options["noise_scaler"] = tracker.noise_scaler
        tracked_sampler, copy_failure = _copy_ksampler_with_options(
            sampler,
            tracked_options,
        )
    if copy_failure is not None:
        if tracker is not None:
            tracker.clear()
        runtime.disable_forecasting_for_run(copy_failure)
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving all-actual "
            "sampling because KSAMPLER copy validation failed: %s",
            copy_failure,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    assert tracked_sampler is not None

    if runtime.config.debug:
        sample_contract = _ksampler_sample_contract(sampler)
        LOG.warning(
            "Spectrum H3 ER-SDE stochastic tracking active run_id=%s sampler=%s "
            "max_stage=%s configured_s_noise=%.8f effective_s_noise=%.8f "
            "external_increment=%s stochastic=%s "
            "ksampler_contract=accepted %s",
            runtime.active_run_id,
            sampler_name(sampler),
            options.get("max_stage", 3),
            configured_s_noise,
            effective_s_noise,
            refdelta_external_increment,
            effective_s_noise > 0.0,
            sample_contract.provenance.log_fields(),
        )

    tracked_extra_args = dict(extra_args)
    tracked_model_options = dict(tracked_extra_args.get("model_options") or {})
    tracked_transformer_options = dict(
        tracked_model_options.get("transformer_options") or {}
    )
    if tracker is not None:
        tracked_transformer_options[ER_SDE_TRACKER_KEY] = tracker
    bridge = None
    if refdelta_external_increment:
        bridge = RefDeltaInteropBridge(
            run_id=int(runtime.active_run_id),
            tracker=tracker,
        )
        tracked_transformer_options[REFDELTA_BRIDGE_KEY] = bridge
    tracked_model_options["transformer_options"] = tracked_transformer_options
    tracked_extra_args["model_options"] = tracked_model_options
    try:
        try:
            return tracked_sampler.sample(
                model_wrap,
                sigmas,
                tracked_extra_args,
                callback,
                noise,
                latent_image,
                denoise_mask,
                disable_pbar,
            )
        except (ERSDETrackingError, RefDeltaInteropError) as exc:
            if bridge is not None and bridge.is_replay_step:
                raise OfflineReplayAbort(
                    f"RefDelta stochastic-state interop failed during replay: {exc}"
                ) from exc
            raise
    finally:
        if bridge is not None:
            bridge.clear()
        if tracker is not None:
            tracker.clear()


def sampler_sample_wrapper(
    executor,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image=None,
    denoise_mask=None,
    disable_pbar=False,
):
    binding = _binding_from_model_options(
        getattr(model_wrap, "model_options", None)
    )
    if binding is None or binding.runtime.active_run_id is None:
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    runtime = binding.runtime
    sampler = executor.class_obj
    name = sampler_name(sampler)
    if (
        (name == REFDELTA_SAMPLER_NAME or name in REFDELTA_BACKEND_SAMPLERS)
        and "multigpu_clones" in ((extra_args or {}).get("model_options") or {})
    ):
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if name in REFDELTA_BACKEND_SAMPLERS:
        extra_args = _with_refdelta_backend_bridge(extra_args, runtime)
    if name in SA_SOLVER_SAMPLERS:
        return _run_solver_aware_sa(
            executor,
            runtime,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if name in ER_SDE_SAMPLERS:
        return _run_tracked_er_sde(
            executor,
            runtime,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if (
        not runtime.config.selective_rollback_correction
        or runtime.experiment_disabled_reason is not None
    ):
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    import comfy.k_diffusion.sampling as native_sampling

    function = getattr(sampler, "sampler_function", None)
    options = dict(getattr(sampler, "extra_options", {}) or {})
    unsupported_reason = None
    if function is not native_sampling.sample_euler:
        unsupported_reason = (
            "selective rollback supports only the exact reviewed sample_euler "
            f"contract; got {sampler_name(sampler)}"
        )
    elif set(options) - {"s_churn", "s_tmin", "s_tmax", "s_noise"}:
        unsupported_reason = "selective rollback received unknown Euler sampler options"
    elif float(options.get("s_churn", 0.0)) != 0.0:
        unsupported_reason = "selective rollback does not support Euler churn"
    elif "multigpu_clones" in (extra_args.get("model_options") or {}):
        unsupported_reason = "selective rollback does not support multi-GPU parallel sampling"
    elif len(executor.wrappers) != 1:
        unsupported_reason = "selective rollback does not support another SAMPLER_SAMPLE wrapper"
    else:
        import comfy.patcher_extension

        predict_wrappers = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            getattr(model_wrap, "model_options", {}) or {},
            is_model_options=True,
        )
        if any(wrapper is not predict_noise_wrapper for wrapper in predict_wrappers):
            unsupported_reason = (
                "selective rollback does not support another PREDICT_NOISE wrapper"
            )

    if unsupported_reason is not None:
        runtime.disable_experiment(unsupported_reason)
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    return run_selective_rollback_euler(
        sampler,
        runtime,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image,
        denoise_mask,
        disable_pbar,
    )


def model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_binding = _binding_from_model_options(
        getattr(source_model, "model_options", None)
    )
    if source_binding is None:
        return
    if not hasattr(cloned_model, "model_options") or cloned_model.model_options is None:
        cloned_model.model_options = {}
    cloned_model.model_options[BINDING_KEY] = SpectrumH3Binding(
        SpectrumH3Runtime(source_binding.runtime.config)
    )


def _place_kj_preview_inside_offline_wrapper(
    model: Any,
    outer_wrapper_type: str,
) -> None:
    """Ensure KJ's observational preview wrapper is entered once for each offline pass."""
    outer_wrappers = (getattr(model, "wrappers", None) or {}).get(outer_wrapper_type)
    if not isinstance(outer_wrappers, dict):
        return
    keys = list(outer_wrappers)
    if (
        KJ_PREVIEW_WRAPPER_KEY not in outer_wrappers
        or WRAPPER_KEY not in outer_wrappers
    ):
        return
    if keys.index(KJ_PREVIEW_WRAPPER_KEY) > keys.index(WRAPPER_KEY):
        return

    preview_wrappers = outer_wrappers.pop(KJ_PREVIEW_WRAPPER_KEY)
    reordered = {}
    for key, wrappers in outer_wrappers.items():
        reordered[key] = wrappers
        if key == WRAPPER_KEY:
            reordered[KJ_PREVIEW_WRAPPER_KEY] = preview_wrappers
    outer_wrappers.clear()
    outer_wrappers.update(reordered)


def install_sampler_wrappers(model: Any, runtime: SpectrumH3Runtime) -> None:
    import comfy.patcher_extension

    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    model.model_options[BINDING_KEY] = SpectrumH3Binding(runtime)
    model.model_options.setdefault("transformer_options", {})

    wrapper_types = comfy.patcher_extension.WrappersMP
    existing_outer = model.get_wrappers(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY)
    if not existing_outer:
        model.add_wrapper_with_key(
            wrapper_types.OUTER_SAMPLE,
            WRAPPER_KEY,
            outer_sample_wrapper,
        )
    if runtime.config.offline_smoothing_replay:
        _place_kj_preview_inside_offline_wrapper(model, wrapper_types.OUTER_SAMPLE)
    existing_predict = model.get_wrappers(wrapper_types.PREDICT_NOISE, WRAPPER_KEY)
    if not existing_predict:
        model.add_wrapper_with_key(
            wrapper_types.PREDICT_NOISE,
            WRAPPER_KEY,
            predict_noise_wrapper,
        )
    existing_sampler = model.get_wrappers(wrapper_types.SAMPLER_SAMPLE, WRAPPER_KEY)
    if not existing_sampler:
        model.add_wrapper_with_key(
            wrapper_types.SAMPLER_SAMPLE,
            WRAPPER_KEY,
            sampler_sample_wrapper,
        )
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLONE
    if not model.get_callbacks(callback_type, WRAPPER_KEY):
        model.add_callback_with_key(callback_type, WRAPPER_KEY, model_clone_callback)
