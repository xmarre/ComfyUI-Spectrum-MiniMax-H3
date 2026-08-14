from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any

import torch

from .objective_media import (
    ObjectiveMediaError,
    evaluate_objective_media,
    persist_objective_report,
)

LOG = logging.getLogger(__name__)

OBJECTIVE_MEDIA_TYPE = "SPECTRUM_H3_OBJECTIVE_MEDIA"
MAX_PENDING_BENCHMARKS = 3
MAX_PENDING_BYTES = 12 * 1024**3

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
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_generation_seed(value: Any) -> int:
    text = str(value).strip()
    try:
        seed = int(text, 10)
    except ValueError as exc:
        raise ObjectiveMediaError("generation_seed must be a decimal integer") from exc
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ObjectiveMediaError("generation_seed must be in the unsigned 64-bit range")
    return seed


def _sequential_provenance(compatibility_tag: str, steps: int) -> dict[str, Any]:
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
            },
        },
        **_ROLE_PROVENANCE,
    }


def _stage_media(video: Any, audio: Any = None) -> dict[str, Any]:
    if not isinstance(video, torch.Tensor):
        raise ObjectiveMediaError("video must be a torch IMAGE tensor")
    staged_video = video.detach().to(device="cpu").contiguous()
    staged_audio = None
    if audio is not None:
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ObjectiveMediaError("audio must be a ComfyUI AUDIO object with waveform and sample_rate")
        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise ObjectiveMediaError("audio.waveform must be a torch tensor")
        staged_audio = {
            "waveform": waveform.detach().to(device="cpu").contiguous(),
            "sample_rate": int(audio["sample_rate"]),
        }
    return {"video": staged_video, "audio": staged_audio}


def _media_nbytes(media: dict[str, Any]) -> int:
    video = media["video"]
    total = int(video.numel() * video.element_size())
    audio = media.get("audio")
    if audio is not None:
        waveform = audio["waveform"]
        total += int(waveform.numel() * waveform.element_size())
    return total


def _media_topology(media: dict[str, Any]) -> tuple[Any, ...]:
    video = media["video"]
    audio = media.get("audio")
    audio_topology: tuple[Any, ...] | None = None
    if audio is not None:
        waveform = audio["waveform"]
        audio_topology = (
            int(waveform.ndim),
            tuple(int(item) for item in waveform.shape[:-1]),
        )
    return (
        tuple(int(item) for item in video.shape),
        str(video.dtype),
        audio is not None,
        audio_topology,
    )


def _pending_bytes_locked() -> int:
    return sum(int(entry["bytes"]) for entry in _PENDING_CAPTURES.values())


def clear_pending_objective_media(benchmark_id: str | None = None) -> tuple[str, ...]:
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
                }
                for benchmark_id, entry in _PENDING_CAPTURES.items()
            },
        }


def _evict_for_capture_locked(benchmark_id: str, extra_bytes: int, *, creating: bool) -> list[str]:
    if extra_bytes > MAX_PENDING_BYTES:
        raise ObjectiveMediaError(
            "one objective-media capture exceeds the sequential capture RAM limit "
            f"({extra_bytes / 1024**3:.2f} GiB > {MAX_PENDING_BYTES / 1024**3:.2f} GiB)"
        )
    evicted: list[str] = []
    while True:
        benchmark_overflow = len(_PENDING_CAPTURES) + int(creating) > MAX_PENDING_BENCHMARKS
        byte_overflow = _pending_bytes_locked() + extra_bytes > MAX_PENDING_BYTES
        if not benchmark_overflow and not byte_overflow:
            break
        victim = next((key for key in _PENDING_CAPTURES if key != benchmark_id), None)
        if victim is None:
            raise ObjectiveMediaError(
                "sequential objective-media capture would exceed its RAM bound; "
                "clear/restart the pending benchmark or reduce decoded media size"
            )
        del _PENDING_CAPTURES[victim]
        evicted.append(victim)
    return evicted


def _summary_from_report(report: dict[str, Any], persisted) -> str:
    rows = {row["metric"]: row for row in report["comparisons"]}
    summary_lines = [
        f"Spectrum H3 objective media: {report['verdict']['value']}",
        f"benchmark={report['benchmark_id']} seed={report['seed']} group={persisted.group_id}",
        f"independent compatible triads={persisted.run_count}",
        (
            "VIDEO: MS-SSIM advantage="
            f"{rows['video_ms_ssim']['candidate_relative_advantage']:+.3%}; "
            "temporal advantage="
            f"{rows['video_temporal_derivative_error']['candidate_relative_advantage']:+.3%}; "
            "motion-detail advantage="
            f"{rows['video_motion_weighted_detail_error']['candidate_relative_advantage']:+.3%}"
        ),
    ]
    if "audio_mrstft_log_magnitude_error" in rows:
        summary_lines.append(
            "AUDIO: MR-STFT advantage="
            f"{rows['audio_mrstft_log_magnitude_error']['candidate_relative_advantage']:+.3%}; "
            "correlation advantage="
            f"{rows['audio_normalized_correlation']['candidate_relative_advantage']:+.3%}"
        )
    summary_lines.append(f"report={persisted.markdown_path}")
    return "\n".join(summary_lines)


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
        "For ordinary R/A/B runs performed sequentially, use Objective Media Capture (Sequential)."
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
                    {"tooltip": "R: decoded native H3 IMAGE batch with Spectrum bypassed."},
                ),
                "legacy_video": (
                    "IMAGE",
                    {"tooltip": "A: decoded accelerated legacy Spectrum IMAGE batch."},
                ),
                "candidate_video": (
                    "IMAGE",
                    {"tooltip": "B: decoded accelerated correction-candidate IMAGE batch."},
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
                "benchmark_id": (
                    "STRING",
                    {
                        "default": "h3-objective-seed-1",
                        "tooltip": "Unique ID for one same-input R/A/B triad.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "provenance_json": (
                    "STRING",
                    {
                        "default": DEFAULT_PROVENANCE_JSON,
                        "multiline": True,
                        "tooltip": (
                            "R/A/B generation provenance plus compatibility metadata. The default is valid but marked "
                            "user-unverified; fill exact values before using cross-seed aggregate evidence."
                        ),
                    },
                ),
                "frame_chunk_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
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
        "One-shot full-reference decoded-media comparison requiring native R, legacy A, and candidate B in the same "
        "execution. For normal sequential testing, use Objective Media Capture (Sequential)."
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
        "One-shot CPU-staged comparison. Requires three separate Media Stage nodes in one execution. "
        "For ordinary sequential testing, use Objective Media Capture (Sequential)."
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
                            "Run the same workflow three times with one benchmark_id: R native reference, "
                            "A legacy Spectrum, then B candidate. Order does not matter."
                        ),
                    },
                ),
                "fps": direct["fps"],
                "benchmark_id": direct["benchmark_id"],
                "generation_seed": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": (
                            "The actual generation seed, copied once and kept identical for R/A/B. This is a STRING "
                            "deliberately so ComfyUI does not attach seed 'control after generate' randomization."
                        ),
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "step": 1}),
                "compatibility_tag": (
                    "STRING",
                    {
                        "default": "minimax-h3-er-sde-current-workflow",
                        "tooltip": (
                            "Short user assertion identifying the unchanged model/precision/scheduler/conditioning/"
                            "decoder setup. Keep it identical across R/A/B and compatible seeds; change it when those "
                            "generation settings change."
                        ),
                    },
                ),
                "frame_chunk_size": direct["frame_chunk_size"],
                "reset_before_capture": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear any incomplete capture for this benchmark_id before storing this role.",
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
        "Recommended objective benchmark workflow. Branch the same decoded IMAGE/AUDIO that also feeds Video Combine "
        "into this node. Run R, A, and B as three ordinary queue executions with the same benchmark_id. Captures live "
        "only in bounded CPU RAM; the third role automatically evaluates, writes reports, and releases all raw media."
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
        provenance = _sequential_provenance(compatibility_tag, int(steps))
        canonical_provenance = _canonical_provenance(provenance)
        staged = _stage_media(video, audio)
        staged_bytes = _media_nbytes(staged)
        topology = _media_topology(staged)
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
                        f"benchmark {benchmark_key!r} already contains role {role_key}; "
                        "set reset_before_capture=true to restart the triad"
                    )
                if existing["fps"] != fps_value:
                    raise ObjectiveMediaError("R/A/B captures for one benchmark must use identical fps")
                if existing["seed"] != seed_value:
                    raise ObjectiveMediaError("R/A/B captures for one benchmark must use identical generation_seed")
                if existing["canonical_provenance"] != canonical_provenance:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical steps and compatibility_tag"
                    )
                if existing["frame_chunk_size"] != chunk_value:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must use identical frame_chunk_size"
                    )
                if existing["topology"] != topology:
                    raise ObjectiveMediaError(
                        "R/A/B captures for one benchmark must have matching decoded video/audio topology"
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
                    "topology": topology,
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
                pending = [key for key in ("R", "A", "B") if key not in existing["roles"]]
                total_gib = _pending_bytes_locked() / 1024**3

        for victim in evicted:
            LOG.warning(
                "Spectrum H3 objective sequential capture evicted incomplete benchmark %s to stay within RAM bounds",
                victim,
            )

        if completed is None:
            summary = (
                f"Spectrum H3 objective capture: stored {role_key} for benchmark={benchmark_key}; "
                f"pending={','.join(pending)}; pending_cpu_ram={total_gib:.2f} GiB"
            )
            print(summary)
            return (summary, "", "", "", "")

        roles = completed["roles"]
        try:
            return _evaluate_and_persist(
                roles["R"]["video"],
                roles["A"]["video"],
                roles["B"]["video"],
                fps=completed["fps"],
                benchmark_id=benchmark_key,
                seed=completed["seed"],
                provenance=completed["provenance"],
                frame_chunk_size=completed["frame_chunk_size"],
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
                        "tooltip": "Benchmark to clear when scope=benchmark.",
                    },
                ),
                "scope": (["benchmark", "all"], {"default": "benchmark"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "clear"
    CATEGORY = "sampling/spectrum/research"
    OUTPUT_NODE = True
    DESCRIPTION = "Optional helper to release incomplete sequential objective-media captures from CPU RAM."

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
    "SpectrumH3ObjectiveStagedQualityCompare": SpectrumH3ObjectiveStagedQualityCompare,
    "SpectrumH3ObjectiveSequentialCapture": SpectrumH3ObjectiveSequentialCapture,
    "SpectrumH3ObjectiveCaptureReset": SpectrumH3ObjectiveCaptureReset,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumH3ObjectiveMediaStage": "Spectrum H3 Objective Media Stage (One-Shot)",
    "SpectrumH3ObjectiveQualityCompare": "Spectrum H3 Objective Quality Compare (One-Shot)",
    "SpectrumH3ObjectiveStagedQualityCompare": "Spectrum H3 Objective Quality Compare (Staged One-Shot)",
    "SpectrumH3ObjectiveSequentialCapture": "Spectrum H3 Objective Media Capture (Sequential - Recommended)",
    "SpectrumH3ObjectiveCaptureReset": "Spectrum H3 Objective Media Capture Reset",
}


__all__ = [
    "DEFAULT_PROVENANCE_JSON",
    "MAX_PENDING_BENCHMARKS",
    "MAX_PENDING_BYTES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SpectrumH3ObjectiveCaptureReset",
    "SpectrumH3ObjectiveMediaStage",
    "SpectrumH3ObjectiveQualityCompare",
    "SpectrumH3ObjectiveSequentialCapture",
    "SpectrumH3ObjectiveStagedQualityCompare",
    "clear_pending_objective_media",
    "pending_objective_media_state",
]
