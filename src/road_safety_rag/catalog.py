from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    description: str
    aliases: tuple[str, ...]
    preferred_standards: tuple[str, ...]
    required_context: tuple[str, ...]
    plausible_range_m: tuple[float, float]
    default_comparator: str

    def search_phrases(self) -> tuple[str, ...]:
        return (self.name, self.description, *self.aliases)


METRICS: dict[str, MetricSpec] = {
    "min_lane_width": MetricSpec(
        key="min_lane_width",
        name="minimum traffic lane width",
        description="standard width of one traffic lane or carriageway lane",
        aliases=("lane width", "width of traffic lane", "carriageway width per lane"),
        preferred_standards=("IRC:73", "IRC:SP:73", "IRC:86", "IRC:SP:84", "IRC:SP:87", "IRC:SP:99"),
        required_context=("road_class", "setting", "lanes_total"),
        plausible_range_m=(2.0, 5.5),
        default_comparator=">=",
    ),
    "min_sign_height": MetricSpec(
        key="min_sign_height",
        name="minimum traffic-sign mounting height",
        description="vertical clearance from road or ground to the lower edge of a traffic sign",
        aliases=("sign mounting height", "bottom edge height", "vertical clearance of sign"),
        preferred_standards=("IRC:67",),
        required_context=("setting", "sign_mounting"),
        plausible_range_m=(0.3, 8.0),
        default_comparator=">=",
    ),
    "traffic_sign_width": MetricSpec(
        key="traffic_sign_width",
        name="traffic-sign face width",
        description="specified width or diameter of the applicable traffic sign",
        aliases=("sign size", "sign diameter", "sign plate width"),
        preferred_standards=("IRC:67",),
        required_context=("sign_class", "design_speed_kmph"),
        plausible_range_m=(0.2, 4.0),
        default_comparator="=",
    ),
    "traffic_sign_height": MetricSpec(
        key="traffic_sign_height",
        name="traffic-sign face height",
        description="specified height or diameter of the applicable traffic sign face",
        aliases=("sign size", "sign diameter", "sign plate height"),
        preferred_standards=("IRC:67",),
        required_context=("sign_class", "design_speed_kmph"),
        plausible_range_m=(0.2, 4.0),
        default_comparator="=",
    ),
    "min_kerb_height": MetricSpec(
        key="min_kerb_height",
        name="kerb height",
        description="specified height of the applicable raised, median, or barrier kerb",
        aliases=("curb height", "raised kerb", "median kerb height"),
        preferred_standards=("IRC:86", "IRC:SP:84", "IRC:SP:99"),
        required_context=("kerb_type",),
        plausible_range_m=(0.05, 0.6),
        default_comparator="=",
    ),
    "min_w_beam_barrier_height": MetricSpec(
        key="min_w_beam_barrier_height",
        name="W-beam barrier mounting height",
        description="height of the W-beam rail or top of barrier above the adjacent road level",
        aliases=("W beam rail height", "metal beam crash barrier height", "semi-rigid barrier height"),
        preferred_standards=("IRC:119", "IRC:SP:84", "IRC:SP:99"),
        required_context=(),
        plausible_range_m=(0.3, 1.5),
        default_comparator="=",
    ),
    "min_concrete_barrier_height": MetricSpec(
        key="min_concrete_barrier_height",
        name="concrete safety-barrier height",
        description="height of a rigid concrete or New Jersey barrier above road level",
        aliases=("rigid barrier height", "New Jersey barrier height", "concrete crash barrier height"),
        preferred_standards=("IRC:119", "IRC:SP:84", "IRC:SP:99"),
        required_context=(),
        plausible_range_m=(0.3, 2.0),
        default_comparator=">=",
    ),
    "min_radius_curvature": MetricSpec(
        key="min_radius_curvature",
        name="minimum horizontal-curve radius",
        description=(
            "minimum radius of a horizontal circular curve from an applicable direct IRC table "
            "or the best available radius-evaluation speed and terrain"
        ),
        aliases=("minimum horizontal radius", "ruling minimum radius", "curve radius"),
        preferred_standards=("IRC:73", "IRC:SP:73", "IRC:86", "IRC:SP:84", "IRC:SP:87", "IRC:SP:99"),
        required_context=("radius_speed_kmph", "terrain"),
        plausible_range_m=(10.0, 10000.0),
        default_comparator=">=",
    ),
}


def get_metric(key: str) -> MetricSpec:
    try:
        return METRICS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(METRICS))
        raise KeyError(f"Unknown metric '{key}'. Valid metrics: {valid}") from exc
