from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

GENERIC_CORRECTION_MODES = (
    "legacy",
    "coordinate_rls",
    "coordinate_rls_reliability",
    "regional",
)
GENERIC_CORRECTION_LIMITERS = ("rational", "hard_clip", "tanh")
GENERIC_CORRECTION_ATTENUATIONS = (
    "mode_default",
    "no_attenuation",
    "general_confidence",
    "correction_reliability",
    "combined_conservative",
)
DEFAULT_RLS_FORGETTING = 0.90
RESEARCH_RLS_FORGETTING = (0.75, 0.90, 0.97, 1.0)
RESEARCH_LIMITS = (0.15, 0.25, 0.40)
EPSILON = 1.0e-12


def resolve_attenuation_policy(mode: str, attenuation: str) -> str:
    """Resolve the attenuation policy actually applied by a live mode."""
    if attenuation not in GENERIC_CORRECTION_ATTENUATIONS:
        raise ValueError(f"unknown generic correction attenuation {attenuation!r}")
    if mode not in GENERIC_CORRECTION_MODES:
        raise ValueError(f"unknown generic correction mode {mode!r}")
    if mode == "legacy":
        return "legacy_internal"
    if attenuation != "mode_default":
        return attenuation
    if mode == "coordinate_rls":
        return "general_confidence"
    return "combined_conservative"


def attenuation_flags(policy: str) -> tuple[bool, bool]:
    if policy == "no_attenuation":
        return False, False
    if policy == "general_confidence":
        return True, False
    if policy == "correction_reliability":
        return False, True
    if policy == "combined_conservative":
        return True, True
    raise ValueError(f"unknown resolved attenuation policy {policy!r}")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        return low
    return max(float(low), min(float(high), resolved))


def coordinate_transport_scale(
    previous_coordinate: float,
    latest_coordinate: float,
    target_coordinate: float,
    *,
    epsilon: float = 1.0e-9,
) -> tuple[float, bool]:
    """Return the signed target/latest spacing ratio, or legacy scale one."""
    values = (
        float(previous_coordinate),
        float(latest_coordinate),
        float(target_coordinate),
    )
    if not all(math.isfinite(value) for value in values):
        return 1.0, False
    spacing = values[1] - values[0]
    if abs(spacing) <= float(epsilon):
        return 1.0, False
    scale = (values[2] - values[1]) / spacing
    if not math.isfinite(scale):
        return 1.0, False
    return float(scale), True


def limit_gain(value: float, limiter: str, limit: float) -> float:
    resolved = float(value)
    bound = float(limit)
    if not math.isfinite(resolved) or not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(
            "gain limiter requires a finite gain and positive finite limit"
        )
    if limiter == "rational":
        return resolved / (1.0 + abs(resolved) / bound)
    if limiter == "hard_clip":
        return max(-bound, min(bound, resolved))
    if limiter == "tanh":
        return bound * math.tanh(resolved / bound)
    raise ValueError(f"unknown generic correction limiter {limiter!r}")


@dataclass(frozen=True, slots=True)
class ScalarMoments:
    sample_count: int
    residual_sq_mean: float
    residual_dot_direction_mean: float
    direction_sq_mean: float
    hold_error_sq_mean: float
    actual_sq_mean: float
    ratio_epsilon: float
    ratio_denominator_rms: float

    def validate(self) -> ScalarMoments:
        if self.sample_count <= 0:
            raise ValueError("quadratic moments require at least one sample")
        values = (
            self.residual_sq_mean,
            self.residual_dot_direction_mean,
            self.direction_sq_mean,
            self.hold_error_sq_mean,
            self.actual_sq_mean,
            self.ratio_epsilon,
            self.ratio_denominator_rms,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("quadratic moments contain a nonfinite value")
        if (
            min(
                self.residual_sq_mean,
                self.direction_sq_mean,
                self.hold_error_sq_mean,
                self.actual_sq_mean,
                self.ratio_epsilon,
                self.ratio_denominator_rms,
            )
            < 0.0
        ):
            raise ValueError("quadratic moments contain an invalid negative magnitude")
        if self.ratio_denominator_rms <= 0.0:
            raise ValueError("quadratic ratio denominator must be positive")
        return self

    def transported(self, scale: float) -> ScalarMoments:
        resolved = float(scale)
        if not math.isfinite(resolved):
            raise ValueError("coordinate transport scale must be finite")
        return ScalarMoments(
            sample_count=self.sample_count,
            residual_sq_mean=self.residual_sq_mean,
            residual_dot_direction_mean=(self.residual_dot_direction_mean * resolved),
            direction_sq_mean=self.direction_sq_mean * resolved * resolved,
            hold_error_sq_mean=self.hold_error_sq_mean,
            actual_sq_mean=self.actual_sq_mean,
            ratio_epsilon=self.ratio_epsilon,
            ratio_denominator_rms=self.ratio_denominator_rms,
        ).validate()

    def mse(self, gain: float) -> float:
        resolved = float(gain)
        if not math.isfinite(resolved):
            raise ValueError("quadratic gain must be finite")
        value = (
            self.residual_sq_mean
            - 2.0 * resolved * self.residual_dot_direction_mean
            + resolved * resolved * self.direction_sq_mean
        )
        tolerance = 1.0e-10 * max(
            1.0,
            abs(self.residual_sq_mean),
            abs(self.residual_dot_direction_mean),
            abs(self.direction_sq_mean),
        )
        if value < 0.0 and abs(value) <= tolerance:
            value = 0.0
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("quadratic moments produced an invalid MSE")
        return float(value)

    def ratio(self, gain: float) -> float:
        return math.sqrt(self.mse(gain)) / self.ratio_denominator_rms

    @property
    def oracle_gain(self) -> float:
        if self.direction_sq_mean <= max(EPSILON, self.ratio_epsilon**2):
            return 0.0
        value = self.residual_dot_direction_mean / self.direction_sq_mean
        return float(value) if math.isfinite(value) else 0.0

    @property
    def directional_cosine(self) -> float:
        denominator = math.sqrt(
            max(0.0, self.residual_sq_mean) * max(0.0, self.direction_sq_mean)
        )
        if denominator <= max(EPSILON, self.ratio_epsilon**2):
            return 0.0
        return clamp(
            self.residual_dot_direction_mean / denominator,
            -1.0,
            1.0,
        )


def combine_moments(parts: list[ScalarMoments]) -> ScalarMoments:
    if not parts:
        raise ValueError("cannot combine an empty moment set")
    total = sum(item.sample_count for item in parts)
    if total <= 0:
        raise ValueError("combined moment set has no samples")

    def weighted(name: str) -> float:
        return (
            sum(item.sample_count * float(getattr(item, name)) for item in parts)
            / total
        )

    actual_sq = weighted("actual_sq_mean")
    hold_sq = weighted("hold_error_sq_mean")
    epsilon = max(math.sqrt(max(0.0, actual_sq)) * 1.0e-6, 1.1920928955078125e-7)
    denominator = max(math.sqrt(max(0.0, hold_sq)), epsilon)
    return ScalarMoments(
        sample_count=total,
        residual_sq_mean=weighted("residual_sq_mean"),
        residual_dot_direction_mean=weighted("residual_dot_direction_mean"),
        direction_sq_mean=weighted("direction_sq_mean"),
        hold_error_sq_mean=hold_sq,
        actual_sq_mean=actual_sq,
        ratio_epsilon=epsilon,
        ratio_denominator_rms=denominator,
    ).validate()


@dataclass(slots=True)
class RecursiveLeastSquares:
    forgetting: float = DEFAULT_RLS_FORGETTING
    b_acc: float = 0.0
    c_acc: float = 0.0
    observations: int = 0
    effective_age: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.forgetting) or not 0.0 < self.forgetting <= 1.0:
            raise ValueError("RLS forgetting factor must be finite and in (0, 1]")

    @property
    def gain(self) -> float:
        if self.c_acc <= EPSILON:
            return 0.0
        value = self.b_acc / self.c_acc
        return float(value) if math.isfinite(value) else 0.0

    @property
    def support(self) -> float:
        return clamp(self.effective_age / 3.0)

    def update(self, b_value: float, c_value: float) -> None:
        b = float(b_value)
        c = float(c_value)
        if not math.isfinite(b) or not math.isfinite(c) or c < 0.0:
            raise ValueError("RLS observation must contain finite B and nonnegative C")
        self.b_acc = self.forgetting * self.b_acc + b
        self.c_acc = self.forgetting * self.c_acc + c
        self.effective_age = self.forgetting * self.effective_age + 1.0
        self.observations += 1
        if not all(
            math.isfinite(value)
            for value in (self.b_acc, self.c_acc, self.effective_age)
        ):
            raise ValueError("RLS state became nonfinite")


@dataclass(slots=True)
class CorrectionReliability:
    observations: int = 0
    alignment_ewma: float = 0.0
    sign_stability_ewma: float = 0.0
    advantage_ewma: float = 0.0
    previous_oracle_gain: float | None = None
    nondegenerate_observations: int = 0

    @property
    def support(self) -> float:
        return clamp(self.nondegenerate_observations / 3.0)

    @property
    def value(self) -> float:
        if self.observations == 0:
            return 0.0
        success = clamp(0.5 + 2.0 * self.advantage_ewma)
        return clamp(
            0.30 * self.alignment_ewma
            + 0.25 * self.sign_stability_ewma
            + 0.25 * success
            + 0.20 * self.support
        )

    def update(self, moments: ScalarMoments, predicted_gain: float) -> None:
        moments.validate()
        gain = float(predicted_gain)
        if not math.isfinite(gain):
            raise ValueError("reliability update gain must be finite")
        alpha = 0.5 if self.observations < 2 else 0.3
        alignment = abs(moments.directional_cosine)
        oracle = moments.oracle_gain
        direction_valid = moments.direction_sq_mean > max(
            EPSILON,
            moments.ratio_epsilon**2,
        )
        if direction_valid:
            self.nondegenerate_observations += 1
        if (
            self.previous_oracle_gain is None
            or abs(oracle) <= 1.0e-9
            or abs(self.previous_oracle_gain) <= 1.0e-9
        ):
            sign_stability = 0.5
        else:
            sign_stability = 1.0 if oracle * self.previous_oracle_gain > 0.0 else 0.0
        advantage = (moments.mse(0.0) - moments.mse(gain)) / max(
            moments.mse(0.0), moments.ratio_epsilon**2
        )
        self.alignment_ewma = (1.0 - alpha) * self.alignment_ewma + alpha * alignment
        self.sign_stability_ewma = (
            1.0 - alpha
        ) * self.sign_stability_ewma + alpha * sign_stability
        self.advantage_ewma = (1.0 - alpha) * self.advantage_ewma + alpha * clamp(
            advantage, -1.0, 1.0
        )
        self.previous_oracle_gain = oracle
        self.observations += 1


@dataclass(slots=True)
class GainState:
    rls: RecursiveLeastSquares = field(default_factory=RecursiveLeastSquares)
    reliability: CorrectionReliability = field(default_factory=CorrectionReliability)

    def raw_gain(self) -> float:
        return self.rls.gain

    def snapshot(self) -> dict[str, Any]:
        return {
            "rls": asdict(self.rls),
            "reliability": asdict(self.reliability),
        }

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> GainState:
        return cls(
            rls=RecursiveLeastSquares(**dict(value["rls"])),
            reliability=CorrectionReliability(**dict(value["reliability"])),
        )


@dataclass(frozen=True, slots=True)
class GainApplication:
    raw_gain: float
    scaled_gain: float
    bounded_gain: float
    attenuation: float
    bound_active: bool
    general_confidence: float
    correction_reliability: float
    rls_support: float
    rls_energy: float
    rls_age: float


def apply_gain(
    state: GainState,
    *,
    general_confidence: float,
    use_general_confidence: bool,
    use_reliability: bool,
    limiter: str,
    limit: float,
    raw_override: float | None = None,
) -> GainApplication:
    raw = state.raw_gain() if raw_override is None else float(raw_override)
    if not math.isfinite(raw):
        raw = 0.0
    confidence = clamp(general_confidence)
    reliability = state.reliability.value
    scale = 1.0
    if use_general_confidence:
        scale *= confidence
    if use_reliability:
        scale *= reliability
    scaled = raw * scale
    bounded = limit_gain(scaled, limiter, limit)
    return GainApplication(
        raw_gain=raw,
        scaled_gain=scaled,
        bounded_gain=bounded,
        attenuation=scaled - bounded,
        bound_active=abs(scaled - bounded) > 1.0e-12,
        general_confidence=confidence,
        correction_reliability=reliability,
        rls_support=state.rls.support,
        rls_energy=state.rls.c_acc,
        rls_age=state.rls.effective_age,
    )


def regularize_region_raw_gains(
    regional: list[GainState],
    global_state: GainState,
) -> list[float]:
    if not regional:
        return []
    global_gain = global_state.raw_gain()
    shrunk = [
        global_gain + state.rls.support * (state.raw_gain() - global_gain)
        for state in regional
    ]
    if len(shrunk) == 1:
        return shrunk
    smoothed: list[float] = []
    for index, value in enumerate(shrunk):
        neighbors = [value]
        if index > 0:
            neighbors.append(shrunk[index - 1])
        if index + 1 < len(shrunk):
            neighbors.append(shrunk[index + 1])
        neighbor_mean = sum(neighbors) / len(neighbors)
        smoothed.append(0.75 * value + 0.25 * neighbor_mean)
    return smoothed


def candidate_gain_family(
    raw_gain: float,
    *,
    general_confidence: float,
    correction_reliability: float,
) -> dict[str, float]:
    scalings = {
        "no_attenuation": 1.0,
        "general_confidence": clamp(general_confidence),
        "correction_reliability": clamp(correction_reliability),
        "combined_conservative": (
            clamp(general_confidence) * clamp(correction_reliability)
        ),
    }
    family: dict[str, float] = {}
    for scaling_name, scale in scalings.items():
        scaled = float(raw_gain) * scale
        for limiter in GENERIC_CORRECTION_LIMITERS:
            for limit in RESEARCH_LIMITS:
                key = f"{scaling_name}__{limiter}__L{limit:.2f}"
                family[key] = limit_gain(scaled, limiter, limit)
    return family


__all__ = [
    "DEFAULT_RLS_FORGETTING",
    "GENERIC_CORRECTION_ATTENUATIONS",
    "GENERIC_CORRECTION_LIMITERS",
    "GENERIC_CORRECTION_MODES",
    "RESEARCH_LIMITS",
    "RESEARCH_RLS_FORGETTING",
    "GainApplication",
    "GainState",
    "RecursiveLeastSquares",
    "ScalarMoments",
    "apply_gain",
    "attenuation_flags",
    "candidate_gain_family",
    "clamp",
    "combine_moments",
    "coordinate_transport_scale",
    "limit_gain",
    "regularize_region_raw_gains",
    "resolve_attenuation_policy",
]
