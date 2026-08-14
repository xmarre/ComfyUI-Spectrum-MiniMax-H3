from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .generic_correction_core import (
    GENERIC_CORRECTION_LIMITERS,
    GENERIC_CORRECTION_MODES,
    GainApplication,
    GainState,
    ScalarMoments,
    apply_gain,
    regularize_region_raw_gains,
)


@dataclass(slots=True)
class ApplicationAggregate:
    count: int = 0
    raw_sum: float = 0.0
    applied_sum: float = 0.0
    raw_abs_max: float = 0.0
    applied_abs_max: float = 0.0
    reliability_sum: float = 0.0
    reliability_min: float = 1.0
    reliability_max: float = 0.0

    def observe(self, application: GainApplication) -> None:
        self.count += 1
        self.raw_sum += application.raw_gain
        self.applied_sum += application.bounded_gain
        self.raw_abs_max = max(self.raw_abs_max, abs(application.raw_gain))
        self.applied_abs_max = max(
            self.applied_abs_max,
            abs(application.bounded_gain),
        )
        reliability = application.correction_reliability
        self.reliability_sum += reliability
        self.reliability_min = min(self.reliability_min, reliability)
        self.reliability_max = max(self.reliability_max, reliability)

    @property
    def raw_mean(self) -> float:
        return self.raw_sum / self.count if self.count else 0.0

    @property
    def applied_mean(self) -> float:
        return self.applied_sum / self.count if self.count else 0.0

    @property
    def reliability_mean(self) -> float:
        return self.reliability_sum / self.count if self.count else 0.0

    @property
    def resolved_reliability_min(self) -> float:
        return self.reliability_min if self.count else 0.0


@dataclass(slots=True)
class GenericCorrectionController:
    mode: str
    limiter: str
    limit: float
    audio: GainState = field(default_factory=GainState)
    video: GainState = field(default_factory=GainState)
    region_ids: tuple[str, ...] = ()
    regions: list[GainState] = field(default_factory=list)
    coordinate_active_count: int = 0
    coordinate_fallback_count: int = 0
    regional_active_count: int = 0
    regional_fallback_count: int = 0
    audio_aggregate: ApplicationAggregate = field(default_factory=ApplicationAggregate)
    video_aggregate: ApplicationAggregate = field(default_factory=ApplicationAggregate)
    overhead_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in GENERIC_CORRECTION_MODES:
            raise ValueError("invalid generic correction mode")
        if self.limiter not in GENERIC_CORRECTION_LIMITERS:
            raise ValueError("invalid generic correction limiter")
        if not math.isfinite(self.limit) or self.limit <= 0.0:
            raise ValueError("generic correction limit must be positive and finite")

    def state_for_stream(self, stream: str) -> GainState:
        if stream == "audio":
            return self.audio
        if stream == "video":
            return self.video
        raise ValueError("generic correction stream must be audio or video")

    def configure_regions(self, region_ids: tuple[str, ...]) -> bool:
        resolved = tuple(str(value) for value in region_ids)
        if not resolved:
            return False
        if not self.region_ids:
            self.region_ids = resolved
            self.regions = [GainState() for _ in resolved]
            return True
        return self.region_ids == resolved and len(self.regions) == len(resolved)

    def application(
        self,
        stream: str,
        *,
        general_confidence: float,
        raw_override: float | None = None,
        record: bool = False,
    ) -> GainApplication:
        state = self.state_for_stream(stream)
        use_reliability = self.mode in {
            "coordinate_rls_reliability",
            "regional",
        }
        application = apply_gain(
            state,
            general_confidence=general_confidence,
            use_general_confidence=True,
            use_reliability=use_reliability,
            limiter=self.limiter,
            limit=self.limit,
            raw_override=raw_override,
        )
        if record:
            aggregate = (
                self.audio_aggregate if stream == "audio" else self.video_aggregate
            )
            aggregate.observe(application)
        return application

    def region_applications(
        self,
        *,
        general_confidence: float,
        record: bool = False,
    ) -> list[GainApplication]:
        if not self.regions:
            return []
        raw_values = regularize_region_raw_gains(self.regions, self.video)
        applications = [
            apply_gain(
                state,
                general_confidence=general_confidence,
                use_general_confidence=True,
                use_reliability=True,
                limiter=self.limiter,
                limit=self.limit,
                raw_override=raw,
            )
            for state, raw in zip(self.regions, raw_values, strict=True)
        ]
        if record:
            for application in applications:
                self.video_aggregate.observe(application)
        return applications

    @staticmethod
    def observe_state(
        state: GainState,
        moments: ScalarMoments,
        predicted_gain: float,
    ) -> None:
        state.reliability.update(moments, predicted_gain)
        state.rls.update(
            moments.residual_dot_direction_mean,
            moments.direction_sq_mean,
        )

    def observe_stream(
        self,
        stream: str,
        moments: ScalarMoments,
        predicted_gain: float,
    ) -> None:
        self.observe_state(self.state_for_stream(stream), moments, predicted_gain)

    def observe_regions(
        self,
        moments: list[ScalarMoments],
        predicted_gains: list[float],
    ) -> None:
        if not (len(moments) == len(predicted_gains) == len(self.regions)):
            raise ValueError("regional correction observation shape changed")
        for state, item, gain in zip(
            self.regions,
            moments,
            predicted_gains,
            strict=True,
        ):
            self.observe_state(state, item, gain)

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "limiter": self.limiter,
            "limit": self.limit,
            "audio": self.audio.snapshot(),
            "video": self.video.snapshot(),
            "region_ids": self.region_ids,
            "regions": [state.snapshot() for state in self.regions],
            "coordinate_active_count": self.coordinate_active_count,
            "coordinate_fallback_count": self.coordinate_fallback_count,
            "regional_active_count": self.regional_active_count,
            "regional_fallback_count": self.regional_fallback_count,
            "audio_aggregate": asdict(self.audio_aggregate),
            "video_aggregate": asdict(self.video_aggregate),
            "overhead_seconds": self.overhead_seconds,
        }

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> GenericCorrectionController:
        controller = cls(
            mode=str(value["mode"]),
            limiter=str(value["limiter"]),
            limit=float(value["limit"]),
            audio=GainState.from_snapshot(dict(value["audio"])),
            video=GainState.from_snapshot(dict(value["video"])),
            region_ids=tuple(str(item) for item in value.get("region_ids", ())),
            regions=[
                GainState.from_snapshot(dict(item)) for item in value.get("regions", ())
            ],
            coordinate_active_count=int(value.get("coordinate_active_count", 0)),
            coordinate_fallback_count=int(value.get("coordinate_fallback_count", 0)),
            regional_active_count=int(value.get("regional_active_count", 0)),
            regional_fallback_count=int(value.get("regional_fallback_count", 0)),
            audio_aggregate=ApplicationAggregate(
                **dict(value.get("audio_aggregate", {}))
            ),
            video_aggregate=ApplicationAggregate(
                **dict(value.get("video_aggregate", {}))
            ),
            overhead_seconds=float(value.get("overhead_seconds", 0.0)),
        )
        if controller.region_ids and len(controller.region_ids) != len(
            controller.regions
        ):
            raise ValueError("generic correction regional snapshot is inconsistent")
        return controller


__all__ = ["ApplicationAggregate", "GenericCorrectionController"]
