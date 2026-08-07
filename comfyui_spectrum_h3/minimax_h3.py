from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch

from .runtime import SpectrumH3Runtime
from .sampling import (
    RUN_ID_KEY,
    RUNTIME_KEY,
    STEP_ID_KEY,
    WRAPPER_KEY,
)

LOG = logging.getLogger(__name__)


def locate_minimax_h3_inner(model: Any) -> tuple[Any | None, str | None]:
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None)
    if inner is not None:
        return inner, "model.diffusion_model"
    inner = getattr(model, "diffusion_model", None)
    if inner is not None:
        return inner, "diffusion_model"
    return None, None


def is_native_minimax_h3(inner: Any) -> bool:
    if inner is None:
        return False
    class_match = (
        type(inner).__name__ == "MiniMaxH3Model"
        and type(inner).__module__ == "comfy.ldm.minimax.model"
    )
    required = (
        "blocks",
        "final_layer",
        "hidden_size",
        "patch_size",
        "latents_dim",
        "audio_latents_dim",
        "sigma_shift_video",
        "sigma_shift_audio",
        "use_adaln_curves",
    )
    if not class_match or not all(hasattr(inner, name) for name in required):
        return False
    if not isinstance(inner.use_adaln_curves, bool):
        return False
    timestep_attribute = "adaln_t_table" if inner.use_adaln_curves else "time_embedder"
    return hasattr(inner, timestep_attribute)


def require_native_minimax_h3(model: Any) -> tuple[Any, str]:
    inner, path = locate_minimax_h3_inner(model)
    if not is_native_minimax_h3(inner):
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        if actual == "comfy.ldm.minimax.model.MiniMaxH3Model":
            actual += " with an incompatible native attribute contract"
        raise TypeError(
            "Spectrum Apply MiniMax H3 requires ComfyUI's native "
            f"comfy.ldm.minimax.model.MiniMaxH3Model; discovered {actual}"
        )
    assert path is not None
    return inner, path


def branch_labels(transformer_options: dict[str, Any]) -> tuple[Any, ...] | None:
    conds = transformer_options.get("cond_or_uncond")
    uuids = transformer_options.get("uuids")
    if conds is None or uuids is None:
        return None
    try:
        cond_values = tuple(int(value) for value in conds)
        uuid_values = tuple(str(value) for value in uuids)
    except (TypeError, ValueError):
        return None
    if not cond_values or len(cond_values) != len(uuid_values):
        return None
    return tuple((cond_values[index], uuid_values[index]) for index in range(len(cond_values)))


def target_segments(layout: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    audio = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "audio"]
    video = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "video"]
    if len(audio) != 1 or len(video) != 1:
        raise RuntimeError("native MiniMax H3 layout must contain one target audio and one target video segment")
    aa, ab = audio[0]
    va, vb = video[0]
    if not (0 <= aa < ab <= va < vb <= int(layout.seq_len)):
        raise RuntimeError("native MiniMax H3 target order is not [audio | video] at the packed tail")
    if ab != va or vb != int(layout.seq_len):
        raise RuntimeError("native MiniMax H3 target audio/video rows are not contiguous tail segments")
    return (aa, ab), (va, vb)


def _native_module(inner: Any):
    module = importlib.import_module(type(inner).__module__)
    required = ("PackedLayout", "unpatchify_video", "unpack_audio", "time_shift_sigma")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"native MiniMax H3 module is missing required helpers: {', '.join(missing)}")
    return module


def _padded_shape(shape: tuple[int, ...], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    t, h, w = shape[-3:]
    pt, ph, pw = patch_size
    return (
        ((t + pt - 1) // pt) * pt,
        ((h + ph - 1) // ph) * ph,
        ((w + pw - 1) // pw) * pw,
    )


def _resolve_layout(inner: Any, context: torch.Tensor, video_x: torch.Tensor, audio_x: torch.Tensor, payload: dict[str, Any]):
    module = _native_module(inner)
    latent_t, latent_h, latent_w = _padded_shape(tuple(video_x.shape), tuple(inner.patch_size))
    signature = (int(context.shape[1]), latent_t, latent_h, latent_w, int(audio_x.shape[-1]))
    layout = payload.get("layout")
    if layout is None or tuple(getattr(layout, "signature", ())) != signature:
        layout = module.PackedLayout(
            signature[0],
            signature[1],
            signature[2],
            signature[3],
            signature[4],
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )
    return layout


def topology_signature(
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    context: torch.Tensor,
    layout: Any,
    transformer_options: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    (aa, ab), (va, vb) = target_segments(layout)
    segments = tuple((kind, int(b) - int(a)) for a, b, kind in layout.segments)
    refs = tuple(
        (
            ref.get("kind"),
            ref.get("latent_t"),
            ref.get("latent_h"),
            ref.get("latent_w"),
            ref.get("ref_audio_t"),
        )
        for ref in (payload.get("refs") or ())
    )
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", inner.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", inner.sigma_shift_audio))
    return (
        ("video_shape", tuple(int(v) for v in video_x.shape)),
        ("video_padded", _padded_shape(tuple(video_x.shape), tuple(inner.patch_size))),
        ("audio_shape", tuple(int(v) for v in audio_x.shape)),
        ("text_length", int(context.shape[1])),
        ("hidden_width", int(inner.hidden_size)),
        ("target_audio_rows", ab - aa),
        ("target_video_rows", vb - va),
        ("segments", segments),
        ("patch_size", tuple(int(v) for v in inner.patch_size)),
        ("refs", refs),
        ("keyframes", tuple(kf.get("resolved_frame_index") for kf in (payload.get("keyframes") or ()))),
        ("sigma_shifts", (shift_v, shift_a)),
        ("adaln_curves", bool(inner.use_adaln_curves)),
    )


@dataclass(slots=True)
class _OutputState:
    layout: Any
    t_emb: torch.Tensor
    video_timestep_row: int
    audio_timestep_row: int
    sigma_v: torch.Tensor
    shift_v: float
    shift_a: float
    original_video_shape: tuple[int, int, int]
    padded_video_shape: tuple[int, int, int]


def _prepare_output_state(
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any],
    payload: dict[str, Any],
    layout: Any,
) -> _OutputState:
    import comfy.model_management

    module = _native_module(inner)
    device = video_x.device
    dtype = context.dtype
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", inner.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", inner.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - module.time_shift_sigma(sigma_v, shift_v, shift_a))
    visual_aug = float(payload.get("visual_cond_noise_aug", getattr(module, "VISUAL_COND_TIMESTEP", 0.999)))
    audio_aug = float(payload.get("audio_cond_noise_aug", getattr(module, "AUDIO_COND_TIMESTEP", 1.0)))
    has_visual_condition = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_audio_condition = any(kind == "ref_audio" for _, _, kind in layout.segments)
    unique_t = sorted(
        {t_v, t_a}
        | ({max(t_v, visual_aug)} if has_visual_condition else set())
        | ({max(t_a, audio_aug)} if has_audio_condition else set())
    )
    timestep_row = {value: index for index, value in enumerate(unique_t)}
    values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if inner.use_adaln_curves:
        table = comfy.model_management.cast_to(inner.adaln_t_table, device=device)
        position = values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    else:
        t_emb = inner.time_embedder(values).to(dtype)
    return _OutputState(
        layout=layout,
        t_emb=t_emb,
        video_timestep_row=timestep_row[t_v],
        audio_timestep_row=timestep_row[t_a],
        sigma_v=sigma_v,
        shift_v=shift_v,
        shift_a=shift_a,
        original_video_shape=tuple(int(v) for v in video_x.shape[-3:]),
        padded_video_shape=_padded_shape(tuple(video_x.shape), tuple(inner.patch_size)),
    )


def _sanitize_prediction(feature: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if not dtype.is_floating_point:
        return None, {"reason": "target dtype is not floating point"}
    fp32 = feature.to(torch.float32)
    finite = torch.isfinite(fp32)
    if not bool(finite.any().item()):
        return None, {"reason": "forecast contains no finite values"}
    info = None
    finfo = torch.finfo(dtype)
    if not bool(finite.all().item()) or bool(((fp32 < finfo.min) | (fp32 > finfo.max)).any().item()):
        info = {
            "nonfinite": int((~finite).sum().item()),
            "below": int((fp32 < finfo.min).sum().item()),
            "above": int((fp32 > finfo.max).sum().item()),
        }
    sanitized = torch.nan_to_num(fp32, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return sanitized.clamp_(min=finfo.min, max=finfo.max).to(dtype), info


def _execute_actual(
    executor,
    inner: Any,
    runtime: SpectrumH3Runtime,
    run_id: int,
    step_id: int,
    call_id: int,
    layout: Any,
    x,
    timestep,
    context,
    transformer_options,
    minimax_payload,
    kwargs,
):
    if len(inner.blocks) == 0:
        raise RuntimeError("native MiniMax H3 has no transformer blocks to observe")
    last_index = len(inner.blocks) - 1
    local_options = dict(transformer_options)
    patches_replace = dict(local_options.get("patches_replace") or {})
    dit_replacements = dict(patches_replace.get("dit") or {})
    patches_replace["dit"] = dit_replacements
    local_options["patches_replace"] = patches_replace
    existing = dit_replacements.get(("double_block", last_index))
    observed = False

    def capture_replacement(args, replacement_context):
        nonlocal observed
        output = existing(args, replacement_context) if existing is not None else replacement_context["original_block"](args)
        if not isinstance(output, dict) or "img" not in output or not torch.is_tensor(output["img"]):
            raise RuntimeError("final MiniMax H3 block replacement did not return {'img': tensor}")
        (aa, _), (_, vb) = target_segments(layout)
        hidden = output["img"]
        if hidden.ndim != 2 or hidden.shape[0] < vb:
            raise RuntimeError("final MiniMax H3 hidden feature is incompatible with the packed layout")
        # Native H3 guarantees contiguous [audio | video] target rows at the
        # packed tail. Keep a view here; materializing torch.cat would create a
        # second full target tensor on the GPU before the required CPU archive.
        target = hidden[aa:vb].unsqueeze(0)
        runtime.observe_actual(run_id, step_id, call_id, target)
        observed = True
        return output

    dit_replacements[("double_block", last_index)] = capture_replacement
    result = executor(
        x,
        timestep,
        context,
        local_options,
        minimax_payload=minimax_payload,
        **kwargs,
    )
    if not observed:
        raise RuntimeError("native MiniMax H3 final transformer block was not executed")
    return result


def _execute_forecast(
    inner: Any,
    predicted: torch.Tensor,
    state: _OutputState,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
):
    module = _native_module(inner)
    (aa, ab), (va, vb) = target_segments(state.layout)
    audio_rows = ab - aa
    video_rows = vb - va
    compact = predicted[0]
    if compact.shape != (audio_rows + video_rows, inner.hidden_size):
        raise RuntimeError("forecasted MiniMax H3 target feature has an invalid compact shape")
    audio_segment = (0, audio_rows, state.audio_timestep_row)
    video_segment = (audio_rows, audio_rows + video_rows, state.video_timestep_row)
    video_projected, audio_projected = inner.final_layer(
        compact, 
        state.t_emb, 
        video_segment, 
        audio_segment,
    )
    latent_t, latent_h, latent_w = state.padded_video_shape
    video_out = module.unpatchify_video(
        video_projected, 
        latent_t, 
        latent_h // inner.patch_size[1], 
        latent_w // inner.patch_size[2],
        inner.latents_dim, 
        inner.patch_size,
    )
    original_t, original_h, original_w = state.original_video_shape
    video_out = video_out[:, :, :original_t, :original_h, :original_w]
    audio_out = module.unpack_audio(audio_projected)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def diffusion_model_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    runtime = options.get(RUNTIME_KEY)
    run_id = options.get(RUN_ID_KEY)
    step_id = options.get(STEP_ID_KEY)
    if not isinstance(runtime, SpectrumH3Runtime) or run_id is None or step_id is None:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    inner = executor.class_obj
    if not is_native_minimax_h3(inner):
        runtime.fallback_current_step(
            int(run_id),
            int(step_id),
            "diffusion model is not ComfyUI's native MiniMax H3",
        )
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        runtime.fallback_current_step(int(run_id), int(step_id), "native H3 input is not a video/audio latent pair")
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    video_x, audio_x = x
    if video_x.shape[0] != 1 or audio_x.shape[0] != 1:
        runtime.fallback_current_step(int(run_id), int(step_id), "native MiniMax H3 batch size is not one")
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    payload = minimax_payload or {}
    layout = _resolve_layout(inner, context, video_x, audio_x, payload)
    (aa, ab), (va, vb) = target_segments(layout)
    labels = branch_labels(options)
    expected_shape = (1, (ab - aa) + (vb - va), int(inner.hidden_size))
    topology = topology_signature(inner, video_x, audio_x, context, layout, options, payload)
    call_id, actual = runtime.begin_model_call(
        int(run_id),
        int(step_id),
        topology=topology,
        labels=labels,
        expected_shape=expected_shape,
    )
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 model call run_id=%s step=%s call=%s path=%s target_audio=%s target_video=%s topology=%s",
            run_id,
            step_id,
            call_id,
            "actual" if actual else "forecast",
            ab - aa,
            vb - va,
            topology,
        )

    if actual:
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
        )

    predicted = runtime.predict(
        int(run_id),
        int(step_id),
        call_id,
        device=video_x.device,
        dtype=context.dtype,
    )
    if predicted is None:
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
        )

    sanitized, event = _sanitize_prediction(predicted, context.dtype)
    if sanitized is None:
        runtime.fallback_current_step(int(run_id), int(step_id), event["reason"] if event else "forecast sanitization failed")
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
        )
    if event is not None and runtime.config.debug:
        LOG.warning("Spectrum H3 forecast sanitized run_id=%s step=%s event=%s", run_id, step_id, event)
    state = _prepare_output_state(inner, video_x, audio_x, timestep, context, options, payload, layout)
    output = _execute_forecast(inner, sanitized, state, video_x, audio_x)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 forecast complete run_id=%s step=%s chunks=%s history=%s",
            run_id,
            step_id,
            runtime.forecaster.last_prediction_chunk_count,
            runtime.forecaster.history_length,
        )
    return output


def install_h3_wrapper(model: Any) -> None:
    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    if not model.get_wrappers(wrapper_type, WRAPPER_KEY):
        model.add_wrapper_with_key(wrapper_type, WRAPPER_KEY, diffusion_model_wrapper)
