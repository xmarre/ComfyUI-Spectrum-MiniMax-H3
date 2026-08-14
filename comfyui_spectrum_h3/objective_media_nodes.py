from __future__ import annotations

import json
import logging
import math
import os
import platform
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
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
SEQUENTIAL_STAGE_WORKSPACE_TARGET_BYTES = 64 * 1024**2
SEQUENTIAL_AUDIO_WORKSPACE_TARGET_BYTES = 8 * 1024**2
SEQUENTIAL_HOST_SAFETY_MARGIN_BYTES = 512 * 1024**2
SEQUENTIAL_CUDA_SAFETY_MARGIN_BYTES = 256 * 1024**2
SEQUENTIAL_UNMEASURED_INCREMENTAL_LIMIT_BYTES = 6 * 1024**3

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


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{int(value) / 1024**3:.3f} GiB"


def _capture_log(
    event: str,
    *,
    started: float,
    detail: bool = False,
    **fields: Any,
) -> None:
    rendered = " ".join(
        f"{name}={value}" for name, value in fields.items() if value is not None
    )
    message = (
        "Spectrum H3 objective capture: "
        f"ts={datetime.now(timezone.utc).isoformat(timespec='milliseconds')} "
        f"elapsed={time.perf_counter() - started:.3f}s event={event}"
    )
    if rendered:
        message += f" {rendered}"
    (LOG.info if detail else LOG.warning)(message)


def _linux_memory_value_bytes(name: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == name:
                    return int(value.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return None


def _process_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            return int(psutil.Process().memory_info().rss)
        except (psutil.Error, OSError, RuntimeError, ValueError):
            pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _available_host_memory_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            return int(psutil.virtual_memory().available)
        except (psutil.Error, OSError, RuntimeError, ValueError):
            pass
    available = _linux_memory_value_bytes("MemAvailable")
    if available is not None:
        return available
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    except (AttributeError, OSError, ValueError):
        return None


def _memory_snapshot(source_device: torch.device) -> dict[str, int | None]:
    snapshot: dict[str, int | None] = {
        "rss_bytes": _process_rss_bytes(),
        "available_host_bytes": _available_host_memory_bytes(),
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
    }
    if source_device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(source_device)
            snapshot.update(
                {
                    "cuda_allocated_bytes": int(
                        torch.cuda.memory_allocated(source_device)
                    ),
                    "cuda_reserved_bytes": int(
                        torch.cuda.memory_reserved(source_device)
                    ),
                    "cuda_free_bytes": int(free_bytes),
                    "cuda_total_bytes": int(total_bytes),
                }
            )
        except (RuntimeError, TypeError, ValueError):
            pass
    return snapshot


def _dtype_bytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _audio_source(audio: Any) -> torch.Tensor | None:
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
    if not waveform.is_floating_point():
        raise ObjectiveMediaError("audio.waveform must be floating point")
    if any(int(size) < 1 for size in waveform.shape):
        raise ObjectiveMediaError("audio.waveform dimensions must be non-empty")
    sample_rate = audio["sample_rate"]
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ObjectiveMediaError("audio.sample_rate must be a positive integer")
    return waveform


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
    if int(video.shape[1]) < 1 or int(video.shape[2]) < 1:
        raise ObjectiveMediaError("video height and width must be non-empty")
    if not video.is_floating_point():
        raise ObjectiveMediaError("video must be floating point")
    audio_topology: tuple[Any, ...] | None = None
    if audio is not None:
        waveform = _audio_source(audio)
        assert waveform is not None
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
        "device": str(video.device),
        "source_tensor_bytes": int(video.numel() * video.element_size()),
    }


def _memory_estimate_for_shapes(
    video_shape: tuple[int, int, int, int],
    video_dtype: torch.dtype,
    source_device_type: str,
    *,
    audio_shape: tuple[int, int, int] | None = None,
    audio_dtype: torch.dtype | None = None,
    audio_device_type: str | None = None,
    existing_pending_bytes: int = 0,
) -> dict[str, int | str]:
    frames, height, width, channels = (int(item) for item in video_shape)
    target_height, target_width = _bounded_analysis_size(height, width)
    source_video_bytes = frames * height * width * channels * _dtype_bytes(
        video_dtype
    )
    source_rgb_bytes_per_frame = height * width * 3 * _dtype_bytes(video_dtype)
    retained_video_bytes = frames * target_height * target_width * 3 * 2
    resize_required = (height, width) != (target_height, target_width)
    conversion_bytes_per_frame = height * width * 3 * _dtype_bytes(
        torch.float32 if resize_required else video_dtype
    )
    interpolation_bytes_per_frame = (
        target_height * target_width * 3 * _dtype_bytes(torch.float32)
        if resize_required
        else 0
    )
    transfer_bytes_per_frame = (
        target_height * target_width * 3 * _dtype_bytes(torch.float16)
        if source_device_type != "cpu"
        else 0
    )
    # Area interpolation has an explicit output plus backend-dependent scratch.
    # Reserve one additional output-sized allowance for that hidden workspace.
    algorithmic_bytes_per_frame = interpolation_bytes_per_frame
    allocated_per_frame = (
        conversion_bytes_per_frame
        + interpolation_bytes_per_frame
        + transfer_bytes_per_frame
        + algorithmic_bytes_per_frame
    )
    chunk_frames = min(
        frames,
        SEQUENTIAL_STAGE_CHUNK_FRAMES,
        max(
            1,
            SEQUENTIAL_STAGE_WORKSPACE_TARGET_BYTES
            // max(1, allocated_per_frame),
        ),
    )
    max_source_chunk_bytes = source_rgb_bytes_per_frame * chunk_frames
    # max_source_chunk_bytes describes the borrowed view into the decoded input.
    # It is logged as live data but is not counted as a Spectrum allocation.
    conversion_workspace_bytes = conversion_bytes_per_frame * chunk_frames
    interpolation_output_bytes = interpolation_bytes_per_frame * chunk_frames
    cpu_transfer_workspace_bytes = transfer_bytes_per_frame * chunk_frames
    algorithmic_workspace_bytes = algorithmic_bytes_per_frame * chunk_frames
    video_workspace_bytes = allocated_per_frame * chunk_frames

    source_audio_bytes = 0
    retained_audio_bytes = 0
    audio_workspace_bytes = 0
    audio_chunk_samples = 0
    audio_source_host_bytes = 0
    audio_source_device_bytes = 0
    if audio_shape is not None:
        assert audio_dtype is not None
        batch, audio_channels, samples = (int(item) for item in audio_shape)
        source_audio_bytes = (
            batch * audio_channels * samples * _dtype_bytes(audio_dtype)
        )
        retained_audio_bytes = batch * audio_channels * samples * _dtype_bytes(
            torch.float32
        )
        bytes_per_output_sample = batch * audio_channels * _dtype_bytes(
            torch.float32
        )
        audio_chunk_samples = min(
            samples,
            max(
                1,
                SEQUENTIAL_AUDIO_WORKSPACE_TARGET_BYTES
                // max(1, bytes_per_output_sample),
            ),
        )
        if audio_device_type != "cpu":
            audio_workspace_bytes = audio_chunk_samples * bytes_per_output_sample
        if audio_device_type == "cpu":
            audio_source_host_bytes = source_audio_bytes
        else:
            audio_source_device_bytes = source_audio_bytes

    retained_analysis_bytes = retained_video_bytes + retained_audio_bytes
    source_video_host_bytes = source_video_bytes if source_device_type == "cpu" else 0
    host_video_workspace_bytes = (
        video_workspace_bytes
        if source_device_type == "cpu"
        else cpu_transfer_workspace_bytes
    )
    device_video_workspace_bytes = (
        0
        if source_device_type == "cpu"
        else video_workspace_bytes - cpu_transfer_workspace_bytes
    )
    staging_workspace_estimate = max(
        host_video_workspace_bytes + device_video_workspace_bytes,
        audio_workspace_bytes,
    )
    host_staging_workspace_bytes = max(
        host_video_workspace_bytes,
        audio_workspace_bytes,
    )
    host_incremental_required_bytes = (
        retained_analysis_bytes
        + host_staging_workspace_bytes
        + SEQUENTIAL_HOST_SAFETY_MARGIN_BYTES
    )
    estimated_host_live_bytes = (
        source_video_host_bytes
        + audio_source_host_bytes
        + int(existing_pending_bytes)
        + retained_analysis_bytes
        + host_staging_workspace_bytes
        + SEQUENTIAL_HOST_SAFETY_MARGIN_BYTES
    )
    estimated_device_live_bytes = (
        (source_video_bytes if source_device_type != "cpu" else 0)
        + audio_source_device_bytes
        + device_video_workspace_bytes
        + (
            SEQUENTIAL_CUDA_SAFETY_MARGIN_BYTES
            if source_device_type == "cuda" or audio_device_type == "cuda"
            else 0
        )
    )
    return {
        "source_device_type": source_device_type,
        "uses_cuda": int(
            source_device_type == "cuda" or audio_device_type == "cuda"
        ),
        "source_tensor_bytes": source_video_bytes,
        "source_audio_bytes": source_audio_bytes,
        "retained_video_bytes": retained_video_bytes,
        "retained_audio_bytes": retained_audio_bytes,
        "retained_analysis_bytes": retained_analysis_bytes,
        "existing_pending_bytes": int(existing_pending_bytes),
        "chunk_frames": chunk_frames,
        "max_source_chunk_bytes": max_source_chunk_bytes,
        "float32_conversion_workspace_bytes": conversion_workspace_bytes,
        "interpolation_output_workspace_bytes": interpolation_output_bytes,
        "cpu_transfer_workspace_bytes": cpu_transfer_workspace_bytes,
        "algorithmic_workspace_bytes": algorithmic_workspace_bytes,
        "staging_workspace_estimate": staging_workspace_estimate,
        "host_staging_workspace_bytes": host_staging_workspace_bytes,
        "device_staging_workspace_bytes": device_video_workspace_bytes,
        "audio_staging_workspace_bytes": audio_workspace_bytes,
        "audio_chunk_samples": audio_chunk_samples,
        "host_safety_margin_bytes": SEQUENTIAL_HOST_SAFETY_MARGIN_BYTES,
        "cuda_safety_margin_bytes": (
            SEQUENTIAL_CUDA_SAFETY_MARGIN_BYTES
            if source_device_type == "cuda" or audio_device_type == "cuda"
            else 0
        ),
        "host_incremental_required_bytes": host_incremental_required_bytes,
        "estimated_host_live_bytes": estimated_host_live_bytes,
        "estimated_device_live_bytes": estimated_device_live_bytes,
        "target_height": target_height,
        "target_width": target_width,
    }


def _estimate_sequential_memory(
    video: torch.Tensor,
    audio: Any,
    *,
    existing_pending_bytes: int,
) -> dict[str, int | str]:
    waveform = _audio_source(audio)
    return _memory_estimate_for_shapes(
        tuple(int(item) for item in video.shape),
        video.dtype,
        video.device.type,
        audio_shape=(
            None
            if waveform is None
            else tuple(int(item) for item in waveform.shape)
        ),
        audio_dtype=None if waveform is None else waveform.dtype,
        audio_device_type=None if waveform is None else waveform.device.type,
        existing_pending_bytes=existing_pending_bytes,
    )


def _preflight_error(
    reason: str,
    estimate: dict[str, int | str],
    snapshot: dict[str, int | None],
) -> ObjectiveMediaError:
    return ObjectiveMediaError(
        "Objective sequential capture aborted before staging:\n"
        f"reason: {reason}\n"
        f"source decoded video (already owned): "
        f"{_format_bytes(int(estimate['source_tensor_bytes']))}\n"
        f"source audio (already owned): "
        f"{_format_bytes(int(estimate['source_audio_bytes']))}\n"
        f"retained analysis destination: "
        f"{_format_bytes(int(estimate['retained_analysis_bytes']))}\n"
        f"estimated staging workspace: "
        f"{_format_bytes(int(estimate['staging_workspace_estimate']))}\n"
        f"existing pending captures: "
        f"{_format_bytes(int(estimate['existing_pending_bytes']))}\n"
        f"estimated host live requirement: "
        f"{_format_bytes(int(estimate['estimated_host_live_bytes']))}\n"
        f"host incremental requirement including safety margin: "
        f"{_format_bytes(int(estimate['host_incremental_required_bytes']))}\n"
        f"process RSS: {_format_bytes(snapshot['rss_bytes'])}\n"
        f"available host RAM: {_format_bytes(snapshot['available_host_bytes'])}\n"
        f"estimated device live requirement: "
        f"{_format_bytes(int(estimate['estimated_device_live_bytes']))}\n"
        f"CUDA free: {_format_bytes(snapshot['cuda_free_bytes'])}"
    )


def _validate_memory_preflight(
    estimate: dict[str, int | str],
    snapshot: dict[str, int | None],
    *,
    current_benchmark_bytes: int,
) -> None:
    retained_bytes = int(estimate["retained_analysis_bytes"])
    if retained_bytes > MAX_PENDING_BYTES:
        raise _preflight_error(
            "one retained analysis capture exceeds the sequential RAM limit "
            "(MAX_PENDING_BYTES)",
            estimate,
            snapshot,
        )
    if current_benchmark_bytes + retained_bytes > MAX_PENDING_BYTES:
        raise _preflight_error(
            "the current benchmark cannot retain this role within MAX_PENDING_BYTES",
            estimate,
            snapshot,
        )
    host_incremental = int(estimate["host_incremental_required_bytes"])
    available_host = snapshot["available_host_bytes"]
    if available_host is not None and host_incremental > available_host:
        raise _preflight_error(
            "estimated new host allocation plus safety margin exceeds available RAM",
            estimate,
            snapshot,
        )
    if (
        available_host is None
        and host_incremental > SEQUENTIAL_UNMEASURED_INCREMENTAL_LIMIT_BYTES
    ):
        raise _preflight_error(
            "host RAM telemetry is unavailable and the conservative absolute "
            "incremental limit would be exceeded",
            estimate,
            snapshot,
        )
    if bool(estimate["uses_cuda"]):
        cuda_free = snapshot["cuda_free_bytes"]
        cuda_reserved = snapshot["cuda_reserved_bytes"]
        cuda_allocated = snapshot["cuda_allocated_bytes"]
        reusable_reserved = (
            max(0, cuda_reserved - cuda_allocated)
            if cuda_reserved is not None and cuda_allocated is not None
            else 0
        )
        cuda_required = (
            int(estimate["device_staging_workspace_bytes"])
            + SEQUENTIAL_CUDA_SAFETY_MARGIN_BYTES
        )
        if cuda_free is not None and cuda_required > cuda_free + reusable_reserved:
            raise _preflight_error(
                "estimated CUDA staging workspace plus safety margin exceeds "
                "CUDA allocator headroom",
                estimate,
                snapshot,
            )


def _validated_min_max(
    chunk: torch.Tensor,
    *,
    name: str,
) -> tuple[float, float]:
    minimum, maximum = torch.aminmax(chunk)
    bounds = torch.stack((minimum, maximum)).detach().to(device="cpu")
    if not bool(torch.isfinite(bounds).all().item()):
        raise ObjectiveMediaError(f"{name} contains NaN or infinite values")
    return float(bounds[0].item()), float(bounds[1].item())


def _stage_audio_sequential(
    audio: Any,
    *,
    estimate: dict[str, int | str],
    benchmark_id: str,
    role: str,
    started: float,
) -> dict[str, Any] | None:
    waveform = _audio_source(audio)
    if waveform is None:
        return None
    destination_bytes = int(estimate["retained_audio_bytes"])
    _capture_log(
        "audio_destination_allocation_start",
        started=started,
        benchmark=benchmark_id,
        role=role,
        bytes=_format_bytes(destination_bytes),
    )
    staged_waveform = torch.empty(
        tuple(int(item) for item in waveform.shape),
        dtype=torch.float32,
        device="cpu",
    )
    _capture_log(
        "audio_destination_allocation_end",
        started=started,
        benchmark=benchmark_id,
        role=role,
        bytes=_format_bytes(destination_bytes),
    )
    chunk_samples = int(estimate["audio_chunk_samples"])
    samples = int(waveform.shape[-1])
    for chunk_index, start_sample in enumerate(
        range(0, samples, chunk_samples),
        start=1,
    ):
        end_sample = min(samples, start_sample + chunk_samples)
        source_chunk = waveform[..., start_sample:end_sample].detach()
        operation_started = time.perf_counter()
        _capture_log(
            "audio_validation_start",
            started=started,
            detail=True,
            benchmark=benchmark_id,
            role=role,
            chunk=chunk_index,
            samples=f"{start_sample}:{end_sample}",
        )
        _validated_min_max(source_chunk, name="audio.waveform")
        _capture_log(
            "audio_validation_end",
            started=started,
            detail=True,
            benchmark=benchmark_id,
            role=role,
            chunk=chunk_index,
            operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
        )
        destination = staged_waveform[..., start_sample:end_sample]
        operation_started = time.perf_counter()
        if waveform.device.type == "cpu":
            _capture_log(
                "audio_cpu_transfer_skipped",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                source_device=waveform.device,
            )
            _capture_log(
                "audio_destination_copy_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
            )
            destination.copy_(source_chunk)
            _capture_log(
                "audio_destination_copy_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )
        else:
            _capture_log(
                "audio_cpu_transfer_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                source_device=waveform.device,
            )
            transferred = torch.empty(
                tuple(int(item) for item in source_chunk.shape),
                dtype=torch.float32,
                device="cpu",
            )
            transferred.copy_(source_chunk)
            _capture_log(
                "audio_cpu_transfer_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )
            operation_started = time.perf_counter()
            _capture_log(
                "audio_destination_copy_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
            )
            destination.copy_(transferred)
            del transferred
            _capture_log(
                "audio_destination_copy_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )
        del destination, source_chunk
    return {
        "waveform": staged_waveform,
        "sample_rate": int(audio["sample_rate"]),
    }


def _stage_media_sequential(
    video: Any,
    audio: Any = None,
    *,
    benchmark_id: str = "direct",
    role: str = "unknown",
    estimate: dict[str, int | str] | None = None,
    capture_started: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = _source_topology(video, audio)
    source_metadata = _source_video_metadata(video)
    frames = source_metadata["frame_count"]
    if estimate is None:
        estimate = _estimate_sequential_memory(
            video,
            audio,
            existing_pending_bytes=0,
        )
    target_height = int(estimate["target_height"])
    target_width = int(estimate["target_width"])
    chunk_frames = int(estimate["chunk_frames"])
    started = time.perf_counter() if capture_started is None else capture_started
    stage_started = time.perf_counter()
    _capture_log(
        "video_destination_allocation_start",
        started=started,
        benchmark=benchmark_id,
        role=role,
        shape=f"{frames}x{target_height}x{target_width}x3",
        dtype=torch.float16,
        bytes=_format_bytes(int(estimate["retained_video_bytes"])),
    )
    staged_video = torch.empty(
        (frames, target_height, target_width, 3),
        dtype=torch.float16,
        device="cpu",
    )
    _capture_log(
        "video_destination_allocation_end",
        started=started,
        benchmark=benchmark_id,
        role=role,
        rss=_format_bytes(_process_rss_bytes()),
    )
    try:
        for chunk_index, start in enumerate(
            range(0, frames, chunk_frames),
            start=1,
        ):
            end = min(frames, start + chunk_frames)
            source_chunk = video[start:end, ..., :3].detach()
            operation_started = time.perf_counter()
            _capture_log(
                "chunk_validation_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                frames=f"{start}:{end}",
            )
            minimum, maximum = _validated_min_max(source_chunk, name="video")
            if minimum < -1.0e-4 or maximum > 1.0001:
                raise ObjectiveMediaError(
                    "IMAGE values must be in the decoded ComfyUI [0, 1] range"
                )
            _capture_log(
                "chunk_validation_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                minimum=f"{minimum:.6g}",
                maximum=f"{maximum:.6g}",
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )

            operation_started = time.perf_counter()
            _capture_log(
                "chunk_resize_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                source=f"{source_metadata['width']}x{source_metadata['height']}",
                target=f"{target_width}x{target_height}",
            )
            work_dtype = (
                torch.float32
                if (source_metadata["height"], source_metadata["width"])
                != (target_height, target_width)
                else video.dtype
            )
            work = torch.empty(
                (end - start, 3, source_metadata["height"], source_metadata["width"]),
                dtype=work_dtype,
                device=video.device,
            )
            work.copy_(source_chunk.movedim(-1, 1))
            work.clamp_(0.0, 1.0)
            if (
                source_metadata["height"] != target_height
                or source_metadata["width"] != target_width
            ):
                resized = F.interpolate(
                    work,
                    size=(target_height, target_width),
                    mode="area",
                )
                del work
            else:
                resized = work
                del work
            del source_chunk
            _capture_log(
                "chunk_resize_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )

            destination = staged_video[start:end]
            if video.device.type == "cpu":
                _capture_log(
                    "chunk_cpu_transfer_skipped",
                    started=started,
                    detail=True,
                    benchmark=benchmark_id,
                    role=role,
                    chunk=chunk_index,
                    source_device=video.device,
                )
                transfer_output = resized.movedim(1, -1)
            else:
                operation_started = time.perf_counter()
                _capture_log(
                    "chunk_cpu_transfer_start",
                    started=started,
                    detail=True,
                    benchmark=benchmark_id,
                    role=role,
                    chunk=chunk_index,
                    source_device=video.device,
                )
                transfer_output = torch.empty(
                    (end - start, target_height, target_width, 3),
                    dtype=torch.float16,
                    device="cpu",
                )
                transfer_output.copy_(resized.movedim(1, -1))
                _capture_log(
                    "chunk_cpu_transfer_end",
                    started=started,
                    detail=True,
                    benchmark=benchmark_id,
                    role=role,
                    chunk=chunk_index,
                    operation_seconds=(
                        f"{time.perf_counter() - operation_started:.3f}"
                    ),
                )
            del resized

            operation_started = time.perf_counter()
            _capture_log(
                "chunk_destination_copy_start",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
            )
            destination.copy_(transfer_output)
            del transfer_output, destination
            _capture_log(
                "chunk_destination_copy_end",
                started=started,
                detail=True,
                benchmark=benchmark_id,
                role=role,
                chunk=chunk_index,
                operation_seconds=f"{time.perf_counter() - operation_started:.3f}",
            )

        staged_audio = _stage_audio_sequential(
            audio,
            estimate=estimate,
            benchmark_id=benchmark_id,
            role=role,
            started=started,
        )
    except Exception:
        _capture_log(
            "staging_failed",
            started=started,
            benchmark=benchmark_id,
            role=role,
            rss=_format_bytes(_process_rss_bytes()),
        )
        del staged_video
        raise

    staged = {"video": staged_video, "audio": staged_audio}
    analysis_bytes = _media_nbytes(staged)
    source_metadata.update(
        {
            "analysis_height": target_height,
            "analysis_width": target_width,
            "analysis_dtype": str(staged_video.dtype),
            "analysis_bytes": analysis_bytes,
            "analysis_profile": BOUNDED_PROFILE_NAME,
            "stage_seconds": time.perf_counter() - stage_started,
            "source_topology": topology,
            "stage_chunk_frames": chunk_frames,
            "staging_workspace_estimate": int(
                estimate["staging_workspace_estimate"]
            ),
        }
    )
    _capture_log(
        "staging_complete",
        started=started,
        benchmark=benchmark_id,
        role=role,
        staging_seconds=f"{time.perf_counter() - stage_started:.3f}",
        retained_analysis_bytes=_format_bytes(analysis_bytes),
        rss=_format_bytes(_process_rss_bytes()),
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


def _validate_existing_capture(
    existing: dict[str, Any] | None,
    *,
    benchmark_key: str,
    role_key: str,
    fps_value: float,
    seed_value: int,
    canonical_provenance: str,
    chunk_value: int,
    source_topology: tuple[Any, ...],
) -> None:
    if existing is None:
        return
    if role_key in existing["roles"]:
        raise ObjectiveMediaError(
            f"benchmark {benchmark_key!r} already contains role "
            f"{role_key}; set reset_before_capture=true to restart the triad"
        )
    if existing["fps"] != fps_value:
        raise ObjectiveMediaError(
            "R/A/B captures for one benchmark must use identical fps"
        )
    if existing["seed"] != seed_value:
        raise ObjectiveMediaError(
            "R/A/B captures for one benchmark must use identical generation_seed"
        )
    if existing["canonical_provenance"] != canonical_provenance:
        raise ObjectiveMediaError(
            "R/A/B captures for one benchmark must use identical steps and "
            "compatibility_tag"
        )
    if existing["frame_chunk_size"] != chunk_value:
        raise ObjectiveMediaError(
            "R/A/B captures for one benchmark must use identical frame_chunk_size"
        )
    if existing["source_topology"] != source_topology:
        raise ObjectiveMediaError(
            "R/A/B captures for one benchmark must have matching decoded "
            "source video/audio topology"
        )


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
    evaluator_started = time.perf_counter()
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
    LOG.warning(
        "Spectrum H3 objective sequential report persistence start: "
        "benchmark=%s evaluator_elapsed=%.3fs rss=%s",
        benchmark_id,
        time.perf_counter() - evaluator_started,
        _format_bytes(_process_rss_bytes()),
    )
    persistence_started = time.perf_counter()
    try:
        result = _persist_report(report)
    except Exception:
        LOG.exception(
            "Spectrum H3 objective sequential report persistence failed: "
            "benchmark=%s elapsed=%.3fs",
            benchmark_id,
            time.perf_counter() - persistence_started,
        )
        raise
    LOG.warning(
        "Spectrum H3 objective sequential report persistence done: "
        "benchmark=%s elapsed=%.3fs rss=%s",
        benchmark_id,
        time.perf_counter() - persistence_started,
        _format_bytes(_process_rss_bytes()),
    )
    return result


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
        "writes reports. Recoverable capture failures are returned as status "
        "text so they do not abort unrelated output nodes."
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
        capture_started = time.perf_counter()
        try:
            return self._capture_or_raise(
                video,
                role,
                fps,
                benchmark_id,
                generation_seed,
                steps,
                compatibility_tag,
                frame_chunk_size,
                reset_before_capture=reset_before_capture,
                audio=audio,
                capture_started=capture_started,
            )
        except Exception as exc:
            if not isinstance(exc, ObjectiveMediaError) and (
                isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))
                or "out of memory" in str(exc).lower()
            ):
                raise
            benchmark_key = str(benchmark_id).strip() or "<empty>"
            role_key = _ROLE_KEYS.get(str(role), str(role))
            error_text = str(exc).replace("\n", " | ")
            _capture_log(
                "capture_skipped_nonfatal",
                started=capture_started,
                benchmark=benchmark_key,
                role=role_key,
                error_type=type(exc).__name__,
                error=error_text,
            )
            if not isinstance(exc, ObjectiveMediaError):
                LOG.exception(
                    "Unexpected Spectrum H3 objective capture failure was "
                    "contained so other output nodes can continue"
                )
            summary = (
                "Spectrum H3 objective capture skipped without aborting the "
                f"workflow: benchmark={benchmark_key} role={role_key}; "
                f"{type(exc).__name__}: {error_text}"
            )
            print(summary)
            return (summary, "", "", "", "")

    def _capture_or_raise(
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
        *,
        capture_started: float | None = None,
    ):
        if capture_started is None:
            capture_started = time.perf_counter()
        _capture_log(
            "input_received",
            started=capture_started,
            benchmark=str(benchmark_id).strip() or "<empty>",
            requested_role=role,
            source_type=type(video).__name__,
        )
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
        fps_value = float(fps)
        chunk_value = int(frame_chunk_size)
        source_metadata = _source_video_metadata(video)
        target_height, target_width = _bounded_analysis_size(
            source_metadata["height"],
            source_metadata["width"],
        )
        waveform = _audio_source(audio)
        _capture_log(
            "capture_start",
            started=capture_started,
            benchmark=benchmark_key,
            role=role_key,
            source_shape=tuple(int(item) for item in video.shape),
            source_dtype=video.dtype,
            source_device=video.device,
            source_tensor_bytes=_format_bytes(
                int(video.numel() * video.element_size())
            ),
            source_resolution=f"{source_metadata['width']}x{source_metadata['height']}",
            frames=source_metadata["frame_count"],
            target_analysis_shape=(
                source_metadata["frame_count"],
                target_height,
                target_width,
                3,
            ),
            target_analysis_dtype=torch.float16,
            audio_shape=(
                None
                if waveform is None
                else tuple(int(item) for item in waveform.shape)
            ),
            audio_dtype=None if waveform is None else waveform.dtype,
            audio_device=None if waveform is None else waveform.device,
        )

        completed: dict[str, Any] | None = None
        evicted: list[str] = []
        staged: dict[str, Any] | None = None
        pending: list[str] = []
        total_gib = 0.0
        # Serialize preflight, staging, and insertion so concurrent capture nodes
        # cannot jointly exceed a preflight computed from stale pending state.
        with _PENDING_LOCK:
            if bool(reset_before_capture):
                released = _PENDING_CAPTURES.pop(benchmark_key, None)
                _capture_log(
                    "reset_before_capture",
                    started=capture_started,
                    benchmark=benchmark_key,
                    role=role_key,
                    released_bytes=(
                        _format_bytes(int(released["bytes"]))
                        if released is not None
                        else _format_bytes(0)
                    ),
                )
            existing = _PENDING_CAPTURES.get(benchmark_key)
            _validate_existing_capture(
                existing,
                benchmark_key=benchmark_key,
                role_key=role_key,
                fps_value=fps_value,
                seed_value=seed_value,
                canonical_provenance=canonical_provenance,
                chunk_value=chunk_value,
                source_topology=source_topology,
            )
            existing_pending_bytes = _pending_bytes_locked()
            estimate = _estimate_sequential_memory(
                video,
                audio,
                existing_pending_bytes=existing_pending_bytes,
            )
            telemetry_device = (
                waveform.device
                if video.device.type != "cuda"
                and waveform is not None
                and waveform.device.type == "cuda"
                else video.device
            )
            snapshot = _memory_snapshot(telemetry_device)
            projected_rss = (
                None
                if snapshot["rss_bytes"] is None
                else snapshot["rss_bytes"]
                + int(estimate["host_incremental_required_bytes"])
            )
            _capture_log(
                "preflight",
                started=capture_started,
                benchmark=benchmark_key,
                role=role_key,
                retained_analysis_bytes=_format_bytes(
                    int(estimate["retained_analysis_bytes"])
                ),
                destination_video_bytes=_format_bytes(
                    int(estimate["retained_video_bytes"])
                ),
                destination_audio_bytes=_format_bytes(
                    int(estimate["retained_audio_bytes"])
                ),
                max_source_chunk_bytes=_format_bytes(
                    int(estimate["max_source_chunk_bytes"])
                ),
                float32_workspace=_format_bytes(
                    int(estimate["float32_conversion_workspace_bytes"])
                ),
                interpolation_workspace=_format_bytes(
                    int(estimate["interpolation_output_workspace_bytes"])
                ),
                cpu_transfer_workspace=_format_bytes(
                    int(estimate["cpu_transfer_workspace_bytes"])
                ),
                algorithmic_workspace=_format_bytes(
                    int(estimate["algorithmic_workspace_bytes"])
                ),
                staging_workspace_estimate=_format_bytes(
                    int(estimate["staging_workspace_estimate"])
                ),
                existing_pending_bytes=_format_bytes(existing_pending_bytes),
                estimated_host_live_bytes=_format_bytes(
                    int(estimate["estimated_host_live_bytes"])
                ),
                host_incremental_required=_format_bytes(
                    int(estimate["host_incremental_required_bytes"])
                ),
                rss=_format_bytes(snapshot["rss_bytes"]),
                projected_rss=_format_bytes(projected_rss),
                available_host_ram=_format_bytes(
                    snapshot["available_host_bytes"]
                ),
                cuda_allocated=_format_bytes(snapshot["cuda_allocated_bytes"]),
                cuda_reserved=_format_bytes(snapshot["cuda_reserved_bytes"]),
                cuda_free=_format_bytes(snapshot["cuda_free_bytes"]),
                chunk_frames=estimate["chunk_frames"],
            )
            try:
                _validate_memory_preflight(
                    estimate,
                    snapshot,
                    current_benchmark_bytes=(
                        0 if existing is None else int(existing["bytes"])
                    ),
                )
            except Exception:
                _capture_log(
                    "preflight_rejected",
                    started=capture_started,
                    benchmark=benchmark_key,
                    role=role_key,
                )
                raise
            _capture_log(
                "preflight_passed",
                started=capture_started,
                benchmark=benchmark_key,
                role=role_key,
            )
            staged, source_metadata = _stage_media_sequential(
                video,
                audio,
                benchmark_id=benchmark_key,
                role=role_key,
                estimate=estimate,
                capture_started=capture_started,
            )
            staged_bytes = _media_nbytes(staged)
            if staged_bytes != int(estimate["retained_analysis_bytes"]):
                raise ObjectiveMediaError(
                    "internal sequential retained-byte estimate mismatch"
                )
            creating = existing is None
            evicted = _evict_for_capture_locked(
                benchmark_key,
                staged_bytes,
                creating=creating,
            )
            _capture_log(
                "pending_store_start",
                started=capture_started,
                benchmark=benchmark_key,
                role=role_key,
                staged_bytes=_format_bytes(staged_bytes),
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
            _capture_log(
                "pending_store_end",
                started=capture_started,
                benchmark=benchmark_key,
                role=role_key,
                triad_complete=completed is not None,
                pending_total=_format_bytes(_pending_bytes_locked()),
                rss=_format_bytes(_process_rss_bytes()),
            )

        for victim in evicted:
            LOG.warning(
                "Spectrum H3 objective sequential capture evicted incomplete "
                "benchmark %s to stay within RAM bounds",
                victim,
            )
        del staged

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
        _capture_log(
            "evaluation_start",
            started=capture_started,
            benchmark=benchmark_key,
            retained=_format_bytes(int(completed["bytes"])),
            rss=_format_bytes(_process_rss_bytes()),
        )
        print(
            "Spectrum H3 objective sequential evaluation start: "
            f"benchmark={benchmark_key} source={metadata['width']}x"
            f"{metadata['height']} analysis={metadata['analysis_width']}x"
            f"{metadata['analysis_height']} retained={retained_gib:.2f} GiB "
            f"profile={BOUNDED_PROFILE_NAME}"
        )
        try:
            result = _evaluate_and_persist_sequential(
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
            _capture_log(
                "evaluation_end",
                started=capture_started,
                benchmark=benchmark_key,
                rss=_format_bytes(_process_rss_bytes()),
            )
            return result
        except Exception:
            _capture_log(
                "evaluation_failed",
                started=capture_started,
                benchmark=benchmark_key,
                rss=_format_bytes(_process_rss_bytes()),
            )
            raise
        finally:
            roles.clear()
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
