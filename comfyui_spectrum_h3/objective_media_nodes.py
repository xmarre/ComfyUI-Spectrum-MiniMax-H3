from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from .objective_media import (
    ObjectiveMediaError,
    evaluate_objective_media,
    persist_objective_report,
)
from .objective_media_bounded import (
    PROFILE_NAME as BOUNDED_PROFILE_NAME,
    evaluate_objective_media_bounded,
)

LOG = logging.getLogger(__name__)

OBJECTIVE_MEDIA_TYPE = "SPECTRUM_H3_OBJECTIVE_MEDIA"
MAX_PENDING_BENCHMARKS = 2
MAX_PENDING_BYTES = 4 * 1024**3
SEQUENTIAL_MAX_ANALYSIS_PIXELS = 393_216
SEQUENTIAL_STAGE_CHUNK_FRAMES = 4

_ROLE_PROVENANCE = {
    "R": {"spectrum": "bypassed", "role": "native_full_compute_reference"},
    "A": {
        "spectrum": "enabled",
        "generic_correction_mode": "legacy",
        "generic_correction_attenuation": "mode_default",
        "generic_correction_limiter": "rational",
        "generic_correction_limit": 0.25,
    },
    "B": {
        "spectrum": "enabled",
        "generic_correction_mode": "coordinate_rls",
        "generic_correction_attenuation": "no_attenuation",
        "generic_correction_limiter": "hard_clip",
        "generic_correction_limit": 0.40,
    },
}

DEFAULT_PROVENANCE_JSON = json.dumps(
    {
        "compatibility": {
            "model": "MiniMax-H3",
            "model_weights": "same-workflow",
            "precision": "same-workflow",
            "sampler": "er_sde",
            "scheduler": "same-workflow",
            "steps": 20,
            "conditioning": "same-workflow",
            "video_vae": "same-workflow",
            "audio_decoder": "same-workflow",
            "generation_settings": {
                "provenance_status": "user-unverified-same-workflow",
            },
        },
        **_ROLE_PROVENANCE,
    },
    separators=(",", ":"),
)

_ROLE_OPTIONS = (
    "R - native reference",
    "A - legacy Spectrum",
    "B - candidate",
)
_ROLE_KEYS = {
    _ROLE_OPTIONS[0]: "R",
    _ROLE_OPTIONS[1]: "A",
    _ROLE_OPTIONS[2]: "B",
    "R": "R",
    "A": "A",
    "B": "B",
}

_PENDING_LOCK = threading.RLock()
_PENDING_CAPTURES: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _parse_provenance(provenance_json: str) -> dict[str, Any]:
    try:
        provenance = json.loads(provenance_json)
    except json.JSONDecodeError as exc:
        raise ObjectiveMediaError(f"provenance_json is invalid: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ObjectiveMediaError("provenance_json must decode to a JSON object")
    return provenance


def _canonical_provenance(provenance: dict[str, Any]) -> str:
    return json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_generation_seed(value: Any) -> int:
    text = str(value).strip()
    try:
        seed = int(text, 10)
    except ValueError as exc:
        raise ObjectiveMediaError("generation_seed must be a decimal integer") from exc
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ObjectiveMediaError(
            "generation_seed must be in the unsigned 64-bit range"
        )
    return seed


def _sequential_provenance(
    compatibility_tag: str,
    steps: int,
) -> dict[str, Any]:
    tag = str(compatibility_tag).strip()
    if not tag:
        raise ObjectiveMediaError("compatibility_tag must be non-empty")
    resolved_steps = int(steps)
    if resolved_steps < 1:
        raise ObjectiveMediaError("steps must be positive")
    asserted = f"user-asserted:{tag}"
    return {
        "compatibility": {
            "model": "MiniMax-H3",
            "model_weights": asserted,
            "precision": asserted,
            "sampler": "er_sde",
            "scheduler": asserted,
            "steps": resolved_steps,
            "conditioning": asserted,
            "video_vae": asserted,
            "audio_decoder": asserted,
            "generation_settings": {
                "compatibility_tag": tag,
                "provenance_status": "user-asserted-same-workflow",
                "objective_metric_profile": BOUNDED_PROFILE_NAME,
                "objective_analysis_max_pixels": SEQUENTIAL_MAX_ANALYSIS_PIXELS,
            },
        },
        **_ROLE_PROVENANCE,
    }


def _stage_audio(audio: Any = None) -> dict[str, Any] | None:
    if audio is None:
        return None
    if (
        not isinstance(audio, dict)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ObjectiveMediaError(
            "audio must be a ComfyUI AUDIO object with waveform and sample_rate"
        )
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise ObjectiveMediaError("audio.waveform must be a torch tensor")
    if waveform.ndim != 3:
        raise ObjectiveMediaError(
            "audio.waveform must have shape [batch, channels, samples]"
        )
    if not torch.isfinite(waveform).all():
        raise ObjectiveMediaError("audio.waveform contains NaN or infinite values")
    return {
        "waveform": waveform.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous(),
        "sample_rate": int(audio["sample_rate"]),
    }


def _stage_media(video: Any, audio: Any = None) -> dict[str, Any]:
    if not isinstance(video, torch.Tensor):
        raise ObjectiveMediaError("video must be a torch IMAGE tensor")
    staged_video = video.detach().to(device="cpu").contiguous()
    return {
        "video": staged_video,
        "audio": _stage_audio(audio),
    }


def _bounded_analysis_size(height: int, width: int) -> tuple[int, int]:
    pixels = int(height) * int(width)
    if pixels <= SEQUENTIAL_MAX_ANALYSIS_PIXELS:
        return int(height), int(width)
    scale = math.sqrt(SEQUENTIAL_MAX_ANALYSIS_PIXELS / pixels)
    target_height = max(1, round(height * scale))
    target_width = max(1, round(width * scale))
    while target_height * target_width > SEQUENTIAL_MAX_ANALYSIS_PIXELS:
        if target_width >= target_height and target_width > 1:
            target_width -= 1
        elif target_height > 1:
            target_height -= 1
        else:
            break
    return target_height, target_width


def _source_topology(video: Any, audio: Any = None) -> tuple[Any, ...]:
    if not isinstance(video, torch.Tensor):
        raise ObjectiveMediaError("video must be a torch IMAGE tensor")
    if video.ndim != 4:
        raise ObjectiveMediaError(
            "video must have shape [frames, height, width, channels]"
        )
    if video.shape[0] < 2:
        raise ObjectiveMediaError("video must contain at least two frames")
    if video.shape[-1] < 3:
        raise ObjectiveMediaError("video must contain at least three channels")
    if not video.is_floating_point():
        raise ObjectiveMediaError("video must be floating point")
    audio_topology: tuple[Any, ...] | None = None
    if audio is not None:
        if (
            not isinstance(audio, dict)
            or "waveform" not in audio
            or "sample_rate" not in audio
        ):
            raise ObjectiveMediaError(
                "audio must be a ComfyUI AUDIO object with waveform and sample_rate"
            )
        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ObjectiveMediaError(
                "audio.waveform must have shape [batch, channels, samples]"
            )
        audio_topology = (
            tuple(int(item) for item in waveform.shape),
            str(waveform.dtype),
            int(audio["sample_rate"]),
        )
    return (
        tuple(int(item) for item in video.shape),
        str(video.dtype),
        audio_topology,
    )


def _source_video_metadata(video: torch.Tensor) -> dict[str, Any]:
    return {
        "frame_count": int(video.shape[0]),
        "height": int(video.shape[1]),
        "width": int(video.shape[2]),
        "channels": int(video.shape[3]),
        "dtype": str(video.dtype),
    }


def _stage_media_sequential(
    video: Any,
    audio: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = _source_topology(video, audio)
    source_metadata = _source_video_metadata(video)
    frames = source_metadata["frame_count"]
    target_height, target_width = _bounded_analysis_size(
        source_metadata["height"],
        source_metadata["width"],
    )
    staged_video = torch.empty(
        (frames, target_height, target_width, 3),
        dtype=torch.float16,
        device="cpu",
    )
    started = time.perf_counter()
    for start in range(0, frames, SEQUENTIAL_STAGE_CHUNK_FRAMES):
        end = min(frames, start + SEQUENTIAL_STAGE_CHUNK_FRAMES)
        chunk = video[start:end, ..., :3].detach()
        if not torch.isfinite(chunk).all():
            raise ObjectiveMediaError("video contains NaN or infinite values")
        minimum = float(chunk.min().detach().cpu().item())
        maximum = float(chunk.max().detach().cpu().item())
        if minimum < -1.0e-4 or maximum > 1.0001:
            raise ObjectiveMediaError(
                "IMAGE values must be in the decoded ComfyUI [0, 1] range"
            )
        nchw = chunk.clamp(0.0, 1.0).movedim(-1, 1)
        if (
            int(nchw.shape[-2]) != target_height
            or int(nchw.shape[-1]) != target_width
        ):
            nchw = F.interpolate(
                nchw.to(dtype=torch.float32),
                size=(target_height, target_width),
                mode="area",
            )
        staged_video[start:end].copy_(
            nchw.movedim(1, -1).to(
                device="cpu",
                dtype=torch.float16,
            )
        )
    staged = {
        "video": staged_video,
        "audio": _stage_audio(audio),
    }
    analysis_bytes = _media_nbytes(staged)
    source_metadata.update(
        {
            "analysis_height": target_height,
            "analysis_width": target_width,
            "analysis_dtype": str(staged_video.dtype),
            "analysis_bytes": analysis_bytes,
            "analysis_profile": BOUNDED_PROFILE_NAME,
            "stage_seconds": time.perf_counter() - started,
            "source_topology": topology,
        }
    )
    return staged, source_metadata


def _media_nbytes(media: dict[str, Any]) -> int:
    video = media["video"]
    total = int(video.numel() * video.element_size())
    audio = media.get("audio")
    if audio is not None:
        waveform = audio["waveform"]
        total += int(waveform.numel() * waveform.element_size())
    return total


def _pending_bytes_locked() -> int:
    return sum(int(entry["bytes"]) for entry in _PENDING_CAPTURES.values())


def clear_pending_objective_media(
    benchmark_id: str | None = None,
) -> tuple[str, ...]:
    with _PENDING_LOCK:
        if benchmark_id is None:
            released = tuple(_PENDING_CAPTURES.keys())
            _PENDING_CAPTURES.clear()
            return released
        key = str(benchmark_id).strip()
        if key in _PENDING_CAPTURES:
            del _PENDING_CAPTURES[key]
            return (key,)
        return ()


def pending_objective_media_state() -> dict[str, Any]:
    with _PENDING_LOCK:
        return {
            "benchmark_count": len(_PENDING_CAPTURES),
            "total_bytes": _pending_bytes_locked(),
            "benchmarks": {
                benchmark_id: {
                    "roles": sorted(entry["roles"]),
                    "bytes": int(entry["bytes"]),
                    "source_video_metadata": dict(
                        entry["source_video_metadata"]
                    ),
                }
                for benchmark_id, entry in _PENDING_CAPTURES.items()
            },
        }


def _evict_for_capture_locked(
    benchmark_id: str,
    extra_bytes: int,
    *,
    creating: bool,
) -> list[str]:
    if extra_bytes > MAX_PENDING_BYTES:
        raise ObjectiveMediaError(
            "one objective-media analysis capture exceeds the sequential RAM limit "
            f"({extra_bytes / 1024**3:.2f} GiB > "
            f"{MAX_PENDING_BYTES / 1024**3:.2f} GiB)"
        )
    evicted: list[str] = []
    while True:
        benchmark_overflow = (
            len(_PENDING_CAPTURES) + int(creating) > MAX_PENDING_BENCHMARKS
        )
        byte_overflow = _pending_bytes_locked() + extra_bytes > MAX_PENDING_BYTES
        if not benchmark_overflow and not byte_overflow:
            break
        victim = next(
            (key for key in _PENDING_CAPTURES if key != benchmark_id),
            None,
        )
        if victim is None:
            raise ObjectiveMediaError(
                "sequential objective-media capture would exceed its bounded RAM "
                "budget; clear/restart the pending benchmark or reduce media size"
            )
        del _PENDING_CAPTURES[victim]
        evicted.append(victim)
    return evicted


def _summary_from_report(report: dict[str, Any], persisted) -> str:
    rows = {row["metric"]: row for row in report["comparisons"]}
    summary_lines = [
        f"Spectrum H3 objective media: {report['verdict']['value']}",
        (
            f"benchmark={report['benchmark_id']} seed={report['seed']} "
            f"group={persisted.group_id}"
        ),
        f"independent compatible triads={persisted.run_count}",
        (
            "VIDEO: MS-SSIM advantage="
            f"{rows['video_ms_ssim']['candidate_relative_advantage']:+.3%}; "
            "temporal advantage="
            f"{rows['video_temporal_derivative_error']['candidate_relative_advantage']:+.3%}; "
            "motion-detail advantage="
            f"{rows['video_motion_weighted_detail_error']['candidate_relative_advantage']:+.3%}; "
            "PSNR diagnostic delta="
            f"{rows['video_psnr_db']['absolute_candidate_delta']:+.3f} dB"
        ),
    ]
    if "audio_mrstft_log_magnitude_error" in rows:
        summary_lines.append(
            "AUDIO: MR-STFT advantage="
            f"{rows['audio_mrstft_log_magnitude_error']['candidate_relative_advantage']:+.3%}; "
            "normalized-correlation diagnostic delta="
            f"{rows['audio_normalized_correlation']['absolute_candidate_delta']:+.5f} points; "
            "SI-SDR diagnostic delta="
            f"{rows['audio_si_sdr_db']['absolute_candidate_delta']:+.3f} dB; "
            "bounded-lag diagnostic delta="
            f"{rows['audio_absolute_bounded_lag_ms']['absolute_candidate_delta']:+.3f} ms"
        )
    summary_lines.append(f"report={persisted.markdown_path}")
    return "\n".join(summary_lines)


def _persist_report(report: dict[str, Any]) -> tuple[str, str, str, str, str]:
    persisted = persist_objective_report(report)
    summary = _summary_from_report(report, persisted)
    print(summary)
    return (
        summary,
        str(persisted.json_path),
        str(persisted.markdown_path),
        str(persisted.aggregate_json_path),
        str(persisted.aggregate_markdown_path),
    )


def _evaluate_and_persist(
    reference_video,
    legacy_video,
    candidate_video,
    *,
    fps: float,
    benchmark_id: str,
    seed: int,
    provenance: dict[str, Any],
    frame_chunk_size: int,
    reference_audio=None,
    legacy_audio=None,
    candidate_audio=None,
) -> tuple[str, str, str, str, str]:
    report = evaluate_objective_media(
        reference_video,
        legacy_video,
        candidate_video,
        fps=float(fps),
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        provenance=provenance,
        reference_audio=reference_audio,
        legacy_audio=legacy_audio,
        candidate_audio=candidate_audio,
        chunk_size=int(frame_chunk_size),
    )
    return _persist_report(report)


def _evaluate_and_persist_sequential(
    reference_video,
    legacy_video,
    candidate_video,
    *,
    fps: float,
    benchmark_id: str,
    seed: int,
    provenance: dict[str, Any],
    frame_chunk_size: int,
    source_video_metadata: dict[str, Any],
    reference_audio=None,
    legacy_audio=None,
    candidate_audio=None,
) -> tuple[str, str, str, str, str]:
    report = evaluate_objective_media_bounded(
        reference_video,
        legacy_video,
        candidate_video,
        fps=float(fps),
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        provenance=provenance,
        source_video_metadata=source_video_metadata,
        reference_audio=reference_audio,
        legacy_audio=legacy_audio,
        candidate_audio=candidate_audio,
        chunk_size=int(frame_chunk_size),
    )
    return _persist_report(report)


class SpectrumH3ObjectiveMediaStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video": ("IMAGE",)},
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = (OBJECTIVE_MEDIA_TYPE,)
    RETURN_NAMES = ("staged_media",)
    FUNCTION = "stage"
    CATEGORY = "sampling/spectrum/research"
    DESCRIPTION = (
        "One-shot three-branch helper only: moves one decoded result to CPU. "
        "This path retains full decoded media and is not recommended for normal "
        "sequential R/A/B testing."
    )

    def stage(self, video, audio=None):
        return (_stage_media(video, audio),)


class SpectrumH3ObjectiveQualityCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "R: decoded native H3 IMAGE batch with Spectrum bypassed."
                        )
                    },
                ),
                "legacy_video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "A: decoded accelerated legacy Spectrum IMAGE batch."
                        )
                    },
                ),
                "candidate_video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "B: decoded accelerated correction-candidate IMAGE batch."
                        )
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.01,
                        "max": 240.0,
                        "step": 0.01,
                    },
                ),
                "benchmark_id": (
                    "STRING",
                    {
                        "default": "h3-objective-seed-1",
                        "tooltip": "Unique ID for one same-input R/A/B triad.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "provenance_json": (
                    "STRING",
                    {
                        "default": DEFAULT_PROVENANCE_JSON,
                        "multiline": True,
                        "tooltip": (
                            "R/A/B generation provenance plus compatibility metadata. "
                            "The default is valid but marked user-unverified; fill exact "
                            "values before using cross-seed aggregate evidence."
                        ),
                    },
                ),
                "frame_chunk_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "legacy_audio": ("AUDIO",),
                "candidate_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "summary",
        "report_json_path",
        "report_markdown_path",
        "aggregate_json_path",
        "aggregate_markdown_path",
    )
    FUNCTION = "compare"
    CATEGORY = "sampling/spectrum/research"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "One-shot full-reference decoded-media comparison requiring native R, "
        "legacy A, and candidate B in the same execution. It retains full media; "
        "use the bounded Sequential capture for ordinary testing."
    )

    def compare(
        self,
        reference_video,
        legacy_video,
        candidate_video,
        fps,
        benchmark_id,
        seed,
        provenance_json,
        frame_chunk_size,
        reference_audio=None,
        legacy_audio=None,
        candidate_audio=None,
    ):
        provenance = _parse_provenance(provenance_json)
        return _evaluate_and_persist(
            reference_video,
            legacy_video,
            candidate_video,
            fps=float(fps),
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            provenance=provenance,
            frame_chunk_size=int(frame_chunk_size),
            reference_audio=reference_audio,
            legacy_audio=legacy_audio,
            candidate_audio=candidate_audio,
        )


class SpectrumH3ObjectiveStagedQualityCompare:
    @classmethod
    def INPUT_TYPES(cls):
        direct = SpectrumH3ObjectiveQualityCompare.INPUT_TYPES()["required"]
        return {
            "required": {
                "reference_media": (OBJECTIVE_MEDIA_TYPE,),
                "legacy_media": (OBJECTIVE_MEDIA_TYPE,),
                "candidate_media": (OBJECTIVE_MEDIA_TYPE,),
                "fps": direct["fps"],
                "benchmark_id": direct["benchmark_id"],
                "seed": direct["seed"],
                "provenance_json": direct["provenance_json"],
                "frame_chunk_size": direct["frame_chunk_size"],
            }
        }

    RETURN_TYPES = SpectrumH3ObjectiveQualityCompare.RETURN_TYPES
    RETURN_NAMES = SpectrumH3ObjectiveQualityCompare.RETURN_NAMES
    FUNCTION = "compare"
    CATEGORY = "sampling/spectrum/research"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "One-shot CPU-staged comparison. Requires three separate Media Stage "
        "nodes in one execution and retains full decoded media. Use the bounded "
        "Sequential capture for ordinary testing."
    )

    def compare(
        self,
        reference_media,
        legacy_media,
        candidate_media,
        fps,
        benchmark_id,
        seed,
        provenance_json,
        frame_chunk_size,
    ):
        return SpectrumH3ObjectiveQualityCompare().compare(
            reference_media["video"],
            legacy_media["video"],
            candidate_media["video"],
            fps,
            benchmark_id,
            seed,
            provenance_json,
            frame_chunk_size,
            reference_audio=reference_media.get("audio"),
            legacy_audio=legacy_media.get("audio"),
            candidate_audio=candidate_media.get("audio"),
        )


class SpectrumH3ObjectiveSequentialCapture:
    @classmethod
    def INPUT_TYPES(cls):
        direct = SpectrumH3ObjectiveQualityCompare.INPUT_TYPES()["required"]
        return {
            "required": {
                "video": ("IMAGE",),
                "role": (
                    list(_ROLE_OPTIONS),
                    {
                        "default": _ROLE_OPTIONS[0],
                        "tooltip": (
                            "Run the same workflow three times with one "
                            "benchmark_id: R native reference, A legacy Spectrum, "
                            "then B candidate. Order does not matter."
                        ),
                    },
                ),
                "fps": direct["fps"],
                "benchmark_id": direct["benchmark_id"],
                "generation_seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": (
                            "Connect the exact same fixed INT seed output that "
                            "drives the generation workflow. The benchmark owns no "
                            "separate seed widget or randomizer."
                        ),
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                    },
                ),
                "compatibility_tag": (
                    "STRING",
                    {
                        "default": "minimax-h3-er-sde-current-workflow",
                        "tooltip": (
                            "Short user assertion identifying the unchanged "
                            "model/precision/scheduler/conditioning/decoder setup. "
                            "Keep it identical across R/A/B."
                        ),
                    },
                ),
                "frame_chunk_size": direct["frame_chunk_size"],
                "reset_before_capture": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Clear an incomplete triad with this benchmark ID "
                            "before storing the current role."
                        ),
                    },
                ),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = SpectrumH3ObjectiveQualityCompare.RETURN_TYPES
    RETURN_NAMES = SpectrumH3ObjectiveQualityCompare.RETURN_NAMES
    FUNCTION = "capture"
    CATEGORY = "sampling/spectrum/research"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Recommended objective benchmark workflow. Each decoded role is "
        "deterministically reduced to a bounded float16 CPU analysis surface "
        "before retention. Full-resolution decoded video is never retained by "
        "the sequential benchmark. The third role automatically evaluates and "
        "writes reports."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def capture(
        self,
        video,
        role,
        fps,
        benchmark_id,
        generation_seed,
        steps,
        compatibility_tag,
        frame_chunk_size,
        reset_before_capture=False,
        audio=None,
    ):
        benchmark_key = str(benchmark_id).strip()
        if not benchmark_key:
            raise ObjectiveMediaError("benchmark_id must be non-empty")
        role_key = _ROLE_KEYS.get(str(role))
        if role_key is None:
            raise ObjectiveMediaError(f"unknown objective-media role: {role}")
        seed_value = _parse_generation_seed(generation_seed)
        provenance = _sequential_provenance(
            compatibility_tag,
            int(steps),
        )
        canonical_provenance = _canonical_provenance(provenance)
        source_topology = _source_topology(video, audio)
        staged, source_metadata = _stage_media_sequential(video, audio)
        staged_bytes = _media_nbytes(staged)
        fps_value = float(fps)
        chunk_value = int(frame_chunk_size)

        completed: dict[str, Any] | None = None
        evicted: list[str] = []
        with _PENDING_LOCK:
            if bool(reset_before_capture):
                _PENDING_CAPTURES.pop(benchmark_key, None)
            existing = _PENDING_CAPTURES.get(benchmark_key)
            if existing is not None:
                if role_key in existing["roles"]:
                    raise ObjectiveMediaError(
                        f"benchmark {benchmark_key!r} already contains role "
                        f"{role_key}; set reset_before_capture=true to restart "
                        "the triad"
                    )
                if existing["fps"] != fps_value:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical fps"
                    )
                if existing["seed"] != seed_value:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical "
                        "generation_seed"
                    )
                if existing["canonical_provenance"] != canonical_provenance:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical "
                        "steps and compatibility_tag"
                    )
                if existing["frame_chunk_size"] != chunk_value:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical "
                        "frame_chunk_size"
                    )
                if existing["source_topology"] != source_topology:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must have matching "
                        "decoded source video/audio topology"
                    )
            creating = existing is None
            evicted = _evict_for_capture_locked(
                benchmark_key,
                staged_bytes,
                creating=creating,
            )
            if existing is None:
                existing = {
                    "fps": fps_value,
                    "seed": seed_value,
                    "provenance": provenance,
                    "canonical_provenance": canonical_provenance,
                    "frame_chunk_size": chunk_value,
                    "source_topology": source_topology,
                    "source_video_metadata": source_metadata,
                    "roles": {},
                    "bytes": 0,
                }
                _PENDING_CAPTURES[benchmark_key] = existing
            existing["roles"][role_key] = staged
            existing["bytes"] = int(existing["bytes"]) + staged_bytes
            _PENDING_CAPTURES.move_to_end(benchmark_key)
            if set(existing["roles"]) == {"R", "A", "B"}:
                completed = _PENDING_CAPTURES.pop(benchmark_key)
            else:
                pending = [
                    key
                    for key in ("R", "A", "B")
                    if key not in existing["roles"]
                ]
                total_gib = _pending_bytes_locked() / 1024**3

        for victim in evicted:
            LOG.warning(
                "Spectrum H3 objective sequential capture evicted incomplete "
                "benchmark %s to stay within RAM bounds",
                victim,
            )

        if completed is None:
            summary = (
                f"Spectrum H3 objective capture: stored {role_key} "
                f"benchmark={benchmark_key}; pending={','.join(pending)}; "
                f"source={source_metadata['width']}x{source_metadata['height']}; "
                f"analysis={source_metadata['analysis_width']}x"
                f"{source_metadata['analysis_height']} float16; "
                f"role_bytes={staged_bytes / 1024**3:.2f} GiB; "
                f"pending_cpu_ram={total_gib:.2f} GiB; "
                f"stage={source_metadata['stage_seconds']:.3f}s"
            )
            print(summary)
            return (summary, "", "", "", "")

        roles = completed["roles"]
        retained_gib = int(completed["bytes"]) / 1024**3
        metadata = completed["source_video_metadata"]
        print(
            "Spectrum H3 objective sequential evaluation start: "
            f"benchmark={benchmark_key} source={metadata['width']}x"
            f"{metadata['height']} analysis={metadata['analysis_width']}x"
            f"{metadata['analysis_height']} retained={retained_gib:.2f} GiB "
            f"profile={BOUNDED_PROFILE_NAME}"
        )
        try:
            return _evaluate_and_persist_sequential(
                roles["R"]["video"],
                roles["A"]["video"],
                roles["B"]["video"],
                fps=completed["fps"],
                benchmark_id=benchmark_key,
                seed=completed["seed"],
                provenance=completed["provenance"],
                frame_chunk_size=completed["frame_chunk_size"],
                source_video_metadata=completed["source_video_metadata"],
                reference_audio=roles["R"].get("audio"),
                legacy_audio=roles["A"].get("audio"),
                candidate_audio=roles["B"].get("audio"),
            )
        finally:
            completed.clear()


class SpectrumH3ObjectiveCaptureReset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_id": (
                    "STRING",
                    {
                        "default": "h3-objective-seed-1",
                        "tooltip": (
                            "Benchmark to clear when scope=benchmark."
                        ),
                    },
                ),
                "scope": (
                    ["benchmark", "all"],
                    {"default": "benchmark"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "clear"
    CATEGORY = "sampling/spectrum/research"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Optional helper to release incomplete sequential objective-media "
        "analysis captures from CPU RAM."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def clear(self, benchmark_id, scope):
        released = clear_pending_objective_media(
            None if str(scope) == "all" else str(benchmark_id)
        )
        summary = (
            "Spectrum H3 objective capture reset: released "
            + (", ".join(released) if released else "nothing")
        )
        print(summary)
        return (summary,)


NODE_CLASS_MAPPINGS = {
    "SpectrumH3ObjectiveMediaStage": SpectrumH3ObjectiveMediaStage,
    "SpectrumH3ObjectiveQualityCompare": SpectrumH3ObjectiveQualityCompare,
    "SpectrumH3ObjectiveStagedQualityCompare": (
        SpectrumH3ObjectiveStagedQualityCompare
    ),
    "SpectrumH3ObjectiveSequentialCapture": SpectrumH3ObjectiveSequentialCapture,
    "SpectrumH3ObjectiveCaptureReset": SpectrumH3ObjectiveCaptureReset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumH3ObjectiveMediaStage": (
        "Spectrum H3 Objective Media Stage (One-Shot / Full Media)"
    ),
    "SpectrumH3ObjectiveQualityCompare": (
        "Spectrum H3 Objective Quality Compare (One-Shot / Full Media)"
    ),
    "SpectrumH3ObjectiveStagedQualityCompare": (
        "Spectrum H3 Objective Quality Compare (Staged One-Shot / Full Media)"
    ),
    "SpectrumH3ObjectiveSequentialCapture": (
        "Spectrum H3 Objective Media Capture (Sequential - Bounded)"
    ),
    "SpectrumH3ObjectiveCaptureReset": (
        "Spectrum H3 Objective Media Capture Reset"
    ),
}


__all__ = [
    "DEFAULT_PROVENANCE_JSON",
    "MAX_PENDING_BENCHMARKS",
    "MAX_PENDING_BYTES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SEQUENTIAL_MAX_ANALYSIS_PIXELS",
    "SpectrumH3ObjectiveCaptureReset",
    "SpectrumH3ObjectiveMediaStage",
    "SpectrumH3ObjectiveQualityCompare",
    "SpectrumH3ObjectiveSequentialCapture",
    "SpectrumH3ObjectiveStagedQualityCompare",
    "clear_pending_objective_media",
    "pending_objective_media_state",
]
