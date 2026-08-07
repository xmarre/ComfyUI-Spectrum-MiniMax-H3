from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .runtime import ForecastRetryActual, SpectrumH3Runtime

LOG = logging.getLogger(__name__)

BINDING_KEY = "spectrum_h3_binding"
RUNTIME_KEY = "spectrum_h3_runtime"
RUN_ID_KEY = "spectrum_h3_run_id"
STEP_ID_KEY = "spectrum_h3_step_id"
COORDINATE_KEY = "spectrum_h3_coordinate"
ACTUAL_KEY = "spectrum_h3_actual"
REASON_KEY = "spectrum_h3_reason"
WRAPPER_KEY = "spectrum_minimax_h3"

SUPPORTED_SINGLE_CALL_SAMPLERS = frozenset(
    {
        "sample_euler",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

RES_MULTISTEP_SAMPLERS = frozenset(
    {
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)


@dataclass(slots=True)
class SpectrumH3Binding:
    runtime: SpectrumH3Runtime


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def sampler_is_supported(sampler: Any) -> bool:
    return sampler_name(sampler) in SUPPORTED_SINGLE_CALL_SAMPLERS


def max_consecutive_forecasts(sampler: Any) -> int | None:
    return 1 if sampler_is_supported(sampler) else None


def min_actual_steps_after_forecast(sampler: Any) -> int:
    name = sampler_name(sampler)
    return 1 if name in SUPPORTED_SINGLE_CALL_SAMPLERS else 0


def min_tail_actual_steps(sampler: Any) -> int:
    return 3 if sampler_name(sampler) in RES_MULTISTEP_SAMPLERS else 0


def _binding_from_model_options(model_options: dict[str, Any] | None) -> SpectrumH3Binding | None:
    binding = (model_options or {}).get(BINDING_KEY)
    return binding if isinstance(binding, SpectrumH3Binding) else None


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

    transformer_options = (getattr(guider, "model_options", None) or {}).get("transformer_options") or {}
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
    run_id = runtime.start_run(
        sigmas,
        name,
        supported_sampler=sampler_is_supported(sampler),
        max_consecutive_forecasts=max_consecutive_forecasts(sampler),
        min_actual_steps_after_forecast=min_actual_steps_after_forecast(sampler),
        min_tail_actual_steps=min_tail_actual_steps(sampler),
    )
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 run start run_id=%s sampler=%s steps=%s supported=%s",
            run_id,
            name,
            runtime.stats.total_steps,
            runtime.supported_sampler,
        )
    try:
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
    finally:
        if runtime.config.debug:
            LOG.warning("Spectrum H3 run summary %s", runtime.debug_summary())
        runtime.end_run(run_id)
        if runtime.config.debug:
            LOG.warning("Spectrum H3 run teardown run_id=%s", run_id)


def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None or binding.runtime.active_run_id is None or not binding.runtime.supported_sampler:
        return executor(x, timestep, model_options or {}, seed)

    if "multigpu_clones" in (model_options or {}):
        if binding.runtime.config.debug:
            LOG.warning("Spectrum H3 native fallback: multi-GPU parallel model calls are not transactionally supported")
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    decision = runtime.begin_step(timestep)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 step run_id=%s step=%s coordinate=%.6f decision=%s reason=%s history=%s window=%.3f",
            decision["run_id"],
            decision["step_id"],
            decision["coordinate"],
            "actual" if decision["actual"] else "forecast",
            decision["reason"],
            runtime.forecaster.history_length,
            runtime.stats.current_window,
        )

    def execute_attempt(attempt_decision: dict[str, Any]):
        patched = copy_model_options_with_step(model_options, runtime, attempt_decision)
        return executor(x, timestep, patched, seed)

    try:
        try:
            result = execute_attempt(decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
        except ForecastRetryActual as retry:
            runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], str(retry))
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
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


def model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_binding = _binding_from_model_options(getattr(source_model, "model_options", None))
    if source_binding is None:
        return
    if not hasattr(cloned_model, "model_options") or cloned_model.model_options is None:
        cloned_model.model_options = {}
    cloned_model.model_options[BINDING_KEY] = SpectrumH3Binding(
        SpectrumH3Runtime(source_binding.runtime.config)
    )


def install_sampler_wrappers(model: Any, runtime: SpectrumH3Runtime) -> None:
    import comfy.patcher_extension

    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    model.model_options[BINDING_KEY] = SpectrumH3Binding(runtime)
    model.model_options.setdefault("transformer_options", {})

    wrapper_types = comfy.patcher_extension.WrappersMP
    existing_outer = model.get_wrappers(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY)
    if not existing_outer:
        model.add_wrapper_with_key(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY, outer_sample_wrapper)
    existing_predict = model.get_wrappers(wrapper_types.PREDICT_NOISE, WRAPPER_KEY)
    if not existing_predict:
        model.add_wrapper_with_key(wrapper_types.PREDICT_NOISE, WRAPPER_KEY, predict_noise_wrapper)
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLONE
    if not model.get_callbacks(callback_type, WRAPPER_KEY):
        model.add_callback_with_key(callback_type, WRAPPER_KEY, model_clone_callback)
