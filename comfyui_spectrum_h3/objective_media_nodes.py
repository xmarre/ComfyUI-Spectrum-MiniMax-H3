from __future__ import annotations

import json

from .objective_media import (
    ObjectiveMediaError,
    evaluate_objective_media,
    persist_objective_report,
)

OBJECTIVE_MEDIA_TYPE = "SPECTRUM_H3_OBJECTIVE_MEDIA"


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
        "Moves one decoded result to CPU immediately. Use one stage per R/A/B branch to keep decoded triad "
        "media out of VRAM before the objective comparison; nothing is written to disk."
    )

    def stage(self, video, audio=None):
        staged_audio = None
        if audio is not None:
            staged_audio = {
                "waveform": audio["waveform"].detach().to(device="cpu"),
                "sample_rate": int(audio["sample_rate"]),
            }
        return ({"video": video.detach().to(device="cpu"), "audio": staged_audio},)


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
                        "default": (
                            '{"compatibility":{"model":"","model_weights":"","precision":"",'
                            '"sampler":"","scheduler":"","steps":20,"conditioning":"",'
                            '"video_vae":"","audio_decoder":"","generation_settings":{}},'
                            '"R":{"spectrum":"bypassed"},"A":{"generic_correction_mode":"legacy"},'
                            '"B":{"generic_correction_mode":"coordinate_rls"}}'
                        ),
                        "multiline": True,
                        "tooltip": (
                            "R/A/B generation provenance plus a compatibility object containing model, weights, precision, "
                            "sampler, scheduler, steps, conditioning, decoders, and remaining generation settings."
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
        "Full-reference decoded-media comparison of native H3 (R), accelerated legacy (A), and an accelerated "
        "candidate (B). It persists bounded JSON/Markdown reports only; raw media is never stored."
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
        try:
            provenance = json.loads(provenance_json)
        except json.JSONDecodeError as exc:
            raise ObjectiveMediaError(f"provenance_json is invalid: {exc}") from exc
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
        summary = "\n".join(summary_lines)
        print(summary)
        return (
            summary,
            str(persisted.json_path),
            str(persisted.markdown_path),
            str(persisted.aggregate_json_path),
            str(persisted.aggregate_markdown_path),
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
        "CPU-staged form of Spectrum H3 Objective Quality Compare. It accepts three outputs from the media-stage node."
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


NODE_CLASS_MAPPINGS = {
    "SpectrumH3ObjectiveMediaStage": SpectrumH3ObjectiveMediaStage,
    "SpectrumH3ObjectiveQualityCompare": SpectrumH3ObjectiveQualityCompare,
    "SpectrumH3ObjectiveStagedQualityCompare": SpectrumH3ObjectiveStagedQualityCompare,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumH3ObjectiveMediaStage": "Spectrum H3 Objective Media Stage",
    "SpectrumH3ObjectiveQualityCompare": "Spectrum H3 Objective Quality Compare",
    "SpectrumH3ObjectiveStagedQualityCompare": "Spectrum H3 Objective Quality Compare (Staged)",
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SpectrumH3ObjectiveMediaStage",
    "SpectrumH3ObjectiveQualityCompare",
    "SpectrumH3ObjectiveStagedQualityCompare",
]
