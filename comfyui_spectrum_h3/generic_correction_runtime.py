from __future__ import annotations

import math
import time
from typing import Any

import torch

from .generic_correction_calibration import (
    GenericCalibrationState,
    exact_segment_moments,
)
from .generic_correction_calibration import (
    record_row as record_calibration_row,
)
from .generic_correction_controller import GenericCorrectionController
from .generic_correction_core import (
    GainApplication,
    ScalarMoments,
    candidate_gain_family,
    combine_moments,
    coordinate_transport_scale,
)
from .generic_correction_topology import TemporalRegion, temporal_video_regions
from .model_aware import ModelAwareForecastDecision
from .runtime import SpectrumH3Runtime


def _weights_with_latest_delta(
    weights: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    resolved = float(coefficient)
    if not math.isfinite(resolved):
        raise ValueError("generic correction coefficient is nonfinite")
    if abs(resolved) <= 1e-15:
        return weights
    if weights.ndim != 1 or weights.numel() < 2:
        raise RuntimeError("generic correction requires two retained actual anchors")
    corrected = weights.clone()
    corrected[-2] -= resolved
    corrected[-1] += resolved
    return corrected


def _advanced_weight_segments(
    runtime: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
    controller: GenericCorrectionController,
) -> tuple[tuple[int, int, torch.Tensor], ...]:
    """Apply a bounded gain to the coordinate-transported latest delta.

    ``generic_correction_limit`` bounds the dimensionless gain multiplying the
    transported direction. The equivalent coefficient on the original anchor
    delta may therefore exceed the limit when the coordinate-spacing ratio has
    magnitude greater than one; bounding that coefficient again would change
    the declared geometry and disagree with exact quadratic calibration.
    """
    started = time.perf_counter()
    history = runtime.forecaster._history
    if len(history) < 2:
        raise RuntimeError("advanced generic correction requires two actual anchors")
    transport_scale, transport_active = coordinate_transport_scale(
        history[-2].coordinate,
        history[-1].coordinate,
        coordinate,
    )
    if transport_active:
        controller.coordinate_active_count += 1
    else:
        controller.coordinate_fallback_count += 1

    ranges = runtime._stream_ranges(call)
    weighted: list[tuple[int, int, torch.Tensor]] = []

    def base_weights(blend: float) -> torch.Tensor:
        return runtime.forecaster.model_aware_weights(
            coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=0.0,
            correction_coefficients=(),
            correction_anchor_ids=(),
        )

    if len(ranges) == 2:
        audio = next(item for item in ranges if item[0] == "audio")
        video = next(item for item in ranges if item[0] == "video")
        audio_application = controller.application(
            "audio",
            general_confidence=decision.confidence,
            record=True,
        )
        audio_weights = _weights_with_latest_delta(
            base_weights(decision.audio_blend_weight),
            audio_application.bounded_gain * transport_scale,
        )
        weighted.append((audio[1], audio[2], audio_weights))

        video_base = base_weights(decision.video_blend_weight)
        regions = temporal_video_regions(
            call.topology,
            video_start_row=video[1],
            video_end_row=video[2],
        )
        regional = bool(
            controller.mode == "regional"
            and regions is not None
            and controller.configure_regions(
                tuple(region.region_id for region in regions)
            )
        )
        if regional:
            region_applications = controller.region_applications(
                general_confidence=decision.confidence,
                record=True,
            )
            if len(region_applications) != len(regions):
                raise RuntimeError("regional correction state is inconsistent")
            for region, application in zip(
                regions,
                region_applications,
                strict=True,
            ):
                weighted.append(
                    (
                        region.start_row,
                        region.end_row,
                        _weights_with_latest_delta(
                            video_base,
                            application.bounded_gain * transport_scale,
                        ),
                    )
                )
            controller.regional_active_count += 1
        else:
            video_application = controller.application(
                "video",
                general_confidence=decision.confidence,
                record=True,
            )
            weighted.append(
                (
                    video[1],
                    video[2],
                    _weights_with_latest_delta(
                        video_base,
                        video_application.bounded_gain * transport_scale,
                    ),
                )
            )
            if controller.mode == "regional":
                controller.regional_fallback_count += 1
    else:
        if not math.isclose(
            decision.audio_blend_weight,
            decision.video_blend_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "packed H3 topology does not expose audio/video correction boundary"
            )
        audio_application = controller.application(
            "audio",
            general_confidence=decision.confidence,
        )
        video_application = controller.application(
            "video",
            general_confidence=decision.confidence,
        )
        coefficient = 0.5 * (
            audio_application.bounded_gain + video_application.bounded_gain
        )
        _name, start, end = ranges[0]
        weighted.append(
            (
                start,
                end,
                _weights_with_latest_delta(
                    base_weights(decision.video_blend_weight),
                    coefficient * transport_scale,
                ),
            )
        )
        if controller.mode == "regional":
            controller.regional_fallback_count += 1

    elapsed = time.perf_counter() - started
    fit_elapsed = max(
        0.0,
        runtime.forecaster.model_aware_fit_seconds
        - getattr(runtime, "_generic_fit_marker", 0.0),
    )
    runtime._generic_fit_marker = runtime.forecaster.model_aware_fit_seconds
    correction_elapsed = max(0.0, elapsed - fit_elapsed)
    runtime.stats.model_aware_fit_seconds += fit_elapsed
    runtime.stats.model_aware_causal_correction_seconds += correction_elapsed
    runtime.stats.model_aware_correction_seconds = (
        runtime.stats.model_aware_causal_correction_seconds
        + runtime.stats.model_aware_offline_correction_seconds
    )
    runtime.stats.model_aware_overhead_seconds += fit_elapsed + correction_elapsed
    controller.overhead_seconds += correction_elapsed
    return tuple(weighted)


def _calibration_row(
    runtime: SpectrumH3Runtime,
    step: Any,
    *,
    stream: str,
    region: TemporalRegion | None,
    legacy_moments: ScalarMoments,
    coordinate_moments: ScalarMoments,
    coordinate_scale: float,
    coordinate_active: bool,
    legacy_decision: ModelAwareForecastDecision,
    live_decision: ModelAwareForecastDecision,
    advanced_application: GainApplication,
    raw_candidate_gain: float,
) -> dict[str, Any]:
    if stream == "audio":
        legacy_telemetry = legacy_decision.audio_correction_telemetry
        projection_ewma = runtime.model_aware.audio_projection_ewma
    else:
        legacy_telemetry = legacy_decision.video_correction_telemetry
        projection_ewma = runtime.model_aware.video_projection_ewma
    legacy_gain = float(legacy_telemetry.generic_gain)
    uncorrected_ratio = legacy_moments.ratio(0.0)
    legacy_ratio = legacy_moments.ratio(legacy_gain)
    candidate_gains = candidate_gain_family(
        raw_candidate_gain,
        general_confidence=live_decision.confidence,
        correction_reliability=advanced_application.correction_reliability,
    )
    history = runtime.forecaster._history
    previous = history[-2]
    latest = history[-1]
    return {
        "schema_version": 1,
        "run_id": int(runtime.stats.run_id),
        "target_step_id": int(step.step_id),
        "target_coordinate": float(step.coordinate),
        "previous_exact_anchor_id": previous.anchor_id,
        "previous_exact_coordinate": float(previous.coordinate),
        "latest_exact_anchor_id": latest.anchor_id,
        "latest_exact_coordinate": float(latest.coordinate),
        "target_exact_anchor_id": int(step.step_id),
        "target_exact_coordinate": float(step.coordinate),
        "forecast_horizon_steps": (
            int(step.step_id) - int(latest.anchor_id)
            if latest.anchor_id is not None
            else None
        ),
        "sampler": runtime.stats.sampler_name,
        "schedule_steps": int(runtime.stats.total_steps),
        "stream": stream,
        "region_id": None if region is None else region.region_id,
        "region_start_row": None if region is None else region.start_row,
        "region_end_row": None if region is None else region.end_row,
        "region_start_temporal_token": (
            None if region is None else region.start_temporal_token
        ),
        "region_end_temporal_token": (
            None if region is None else region.end_temporal_token
        ),
        "sample_count": int(coordinate_moments.sample_count),
        "A": float(coordinate_moments.residual_sq_mean),
        "B": float(coordinate_moments.residual_dot_direction_mean),
        "C": float(coordinate_moments.direction_sq_mean),
        "legacy_A": float(legacy_moments.residual_sq_mean),
        "legacy_B": float(legacy_moments.residual_dot_direction_mean),
        "legacy_C": float(legacy_moments.direction_sq_mean),
        "hold_error_sq_mean": float(coordinate_moments.hold_error_sq_mean),
        "actual_sq_mean": float(coordinate_moments.actual_sq_mean),
        "ratio_epsilon": float(coordinate_moments.ratio_epsilon),
        "ratio_denominator_rms": float(coordinate_moments.ratio_denominator_rms),
        "coordinate_transport_scale": float(coordinate_scale),
        "coordinate_transport_active": bool(coordinate_active),
        "oracle_gain": float(coordinate_moments.oracle_gain),
        "legacy_oracle_gain": float(legacy_moments.oracle_gain),
        "uncorrected_ratio": float(uncorrected_ratio),
        "legacy_corrected_ratio": float(legacy_ratio),
        "target_legacy_projection": float(legacy_moments.oracle_gain),
        "projection_ewma_pre_target": float(projection_ewma),
        "raw_legacy_gain": float(legacy_telemetry.raw_generic_gain),
        "bounded_legacy_gain": legacy_gain,
        "legacy_bound_active": bool(legacy_telemetry.generic_bound_active),
        "general_forecast_confidence": float(live_decision.confidence),
        "horizon_scale": float(min(1.5, max(0.5, live_decision.forecast_horizon))),
        "direction_rms": math.sqrt(max(0.0, coordinate_moments.direction_sq_mean)),
        "residual_rms": math.sqrt(max(0.0, coordinate_moments.residual_sq_mean)),
        "directional_cosine": float(coordinate_moments.directional_cosine),
        "legacy_improved": bool(legacy_ratio < uncorrected_ratio),
        "rls_raw_gain_pre_target": float(raw_candidate_gain),
        "rls_support_pre_target": float(advanced_application.rls_support),
        "rls_energy_pre_target": float(advanced_application.rls_energy),
        "rls_effective_age_pre_target": float(advanced_application.rls_age),
        "correction_reliability_pre_target": float(
            advanced_application.correction_reliability
        ),
        "live_advanced_gain": float(advanced_application.bounded_gain),
        "candidate_gains": candidate_gains,
    }


def _exact_anchor_analysis(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    live_decision: ModelAwareForecastDecision,
    legacy_decision: ModelAwareForecastDecision,
    base_weights: dict[str, torch.Tensor],
) -> None:
    controller = getattr(runtime, "_generic_correction_controller", None)
    if not isinstance(controller, GenericCorrectionController):
        return
    ranges = runtime._stream_ranges(step.calls[0])
    if len(ranges) != 2 or {item[0] for item in ranges} != {"audio", "video"}:
        controller.regional_fallback_count += int(controller.mode == "regional")
        return
    audio = next(item for item in ranges if item[0] == "audio")
    video = next(item for item in ranges if item[0] == "video")
    if "audio" not in base_weights or "video" not in base_weights:
        raise RuntimeError("exact generic correction analysis lacks stream weights")

    regions = temporal_video_regions(
        step.calls[0].topology,
        video_start_row=video[1],
        video_end_row=video[2],
    )
    weighted_segments: list[tuple[str, int, int, torch.Tensor]] = [
        ("audio", audio[1], audio[2], base_weights["audio"]),
    ]
    if regions is None:
        weighted_segments.append(("video", video[1], video[2], base_weights["video"]))
    else:
        weighted_segments.extend(
            (
                f"video:{region.region_id}",
                region.start_row,
                region.end_row,
                base_weights["video"],
            )
            for region in regions
        )
    exact = exact_segment_moments(
        runtime.forecaster,
        combined,
        weighted_segments,
    )
    audio_legacy = exact["audio"]
    if regions is None:
        video_region_legacy: list[ScalarMoments] = []
        video_legacy = exact["video"]
    else:
        video_region_legacy = [exact[f"video:{region.region_id}"] for region in regions]
        video_legacy = combine_moments(video_region_legacy)

    history = runtime.forecaster._history
    transport_scale, transport_active = coordinate_transport_scale(
        history[-2].coordinate,
        history[-1].coordinate,
        step.coordinate,
    )
    audio_coordinate = audio_legacy.transported(transport_scale)
    video_coordinate = video_legacy.transported(transport_scale)
    video_region_coordinate = [
        item.transported(transport_scale) for item in video_region_legacy
    ]

    audio_application = controller.application(
        "audio",
        general_confidence=live_decision.confidence,
    )
    video_application = controller.application(
        "video",
        general_confidence=live_decision.confidence,
    )
    region_applications: list[GainApplication] = []
    if regions is not None and controller.configure_regions(
        tuple(region.region_id for region in regions)
    ):
        region_applications = controller.region_applications(
            general_confidence=live_decision.confidence,
        )

    state = getattr(runtime, "_generic_correction_calibration", None)
    calibration_state = state if isinstance(state, GenericCalibrationState) else None
    for stream, legacy, transported, application, raw_gain in (
        (
            "audio",
            audio_legacy,
            audio_coordinate,
            audio_application,
            controller.audio.raw_gain(),
        ),
        (
            "video",
            video_legacy,
            video_coordinate,
            video_application,
            controller.video.raw_gain(),
        ),
    ):
        row = _calibration_row(
            runtime,
            step,
            stream=stream,
            region=None,
            legacy_moments=legacy,
            coordinate_moments=transported,
            coordinate_scale=transport_scale,
            coordinate_active=transport_active,
            legacy_decision=legacy_decision,
            live_decision=live_decision,
            advanced_application=application,
            raw_candidate_gain=raw_gain,
        )
        record_calibration_row(
            calibration_state,
            row,
            topology=step.calls[0].topology,
        )

    if regions is not None and len(region_applications) == len(regions):
        for index, (region, legacy, transported, application) in enumerate(
            zip(
                regions,
                video_region_legacy,
                video_region_coordinate,
                region_applications,
                strict=True,
            )
        ):
            row = _calibration_row(
                runtime,
                step,
                stream="video",
                region=region,
                legacy_moments=legacy,
                coordinate_moments=transported,
                coordinate_scale=transport_scale,
                coordinate_active=transport_active,
                legacy_decision=legacy_decision,
                live_decision=live_decision,
                advanced_application=application,
                raw_candidate_gain=controller.regions[index].raw_gain(),
            )
            record_calibration_row(
                calibration_state,
                row,
                topology=step.calls[0].topology,
            )

    controller.observe_stream(
        "audio",
        audio_coordinate,
        audio_application.bounded_gain,
    )
    controller.observe_stream(
        "video",
        video_coordinate,
        video_application.bounded_gain,
    )
    if video_region_coordinate and len(region_applications) == len(
        video_region_coordinate
    ):
        controller.observe_regions(
            video_region_coordinate,
            [application.bounded_gain for application in region_applications],
        )


__all__ = ["_advanced_weight_segments", "_exact_anchor_analysis"]
