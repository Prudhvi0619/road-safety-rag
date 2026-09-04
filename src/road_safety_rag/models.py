from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RuleStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NEEDS_CONTEXT = "needs_context"
    NOT_APPLICABLE = "not_applicable"
    INVALID_EVIDENCE = "invalid_evidence"


class RoadContext(StrictModel):
    """Facts that determine whether a standard is applicable.

    Unknown facts stay null. They are never silently replaced by a National
    Highway/default-road assumption.
    """

    jurisdiction: str = "India"
    road_class: str | None = None
    road_class_source: str | None = None
    road_class_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    highway_ref: str | None = None
    setting: Literal["urban", "rural", "unknown"] = "unknown"
    terrain: Literal["plain", "rolling", "mountainous", "steep", "unknown"] = "unknown"
    carriageway: Literal["divided", "undivided", "one_way", "unknown"] = "unknown"
    # OSM's lanes=* describes the individual OSM way. On a mapped dual
    # carriageway that is normally one direction only, so it must not be
    # treated as the total number of lanes across both carriageways.
    osm_way_lanes: int | None = Field(default=None, ge=1, le=30)
    lanes_per_carriageway: int | None = Field(default=None, ge=1, le=30)
    opposite_carriageway_lanes: int | None = Field(default=None, ge=1, le=30)
    osm_total_lanes: int | None = Field(default=None, ge=1, le=30)
    carriageway_count: int | None = Field(default=None, ge=1, le=4)
    # Base through-lane configuration used for standards applicability. This
    # can differ from raw OSM totals when a short auxiliary/turn lane exists.
    total_road_lanes: int | None = Field(default=None, ge=1, le=30)
    lane_count_source: str | None = None
    # True when the lane configuration is inferred from mapping/CV evidence
    # rather than confirmed by a reviewed drawing or explicit user input.
    lane_configuration_provisional: bool | None = None
    # Backward-compatible alias used by the original CLI and report code. Its
    # meaning is now strictly the verified total across the complete road.
    lanes_total: int | None = Field(default=None, ge=1, le=30)
    # Design speed and posted speed are different engineering facts. OSM's
    # maxspeed tag (and a detected speed-limit sign) is retained only as a
    # posted-speed proxy; it must never be silently promoted to verified
    # design speed.
    design_speed_kmph: float | None = Field(default=None, gt=0, le=250)
    design_speed_source: str | None = None
    posted_speed_kmph: float | None = Field(default=None, gt=0, le=250)
    posted_speed_source: str | None = None
    osm_maxspeed_values_kmph: list[float] = Field(default_factory=list)
    sign_class: str | None = None
    sign_shape: str | None = None
    sign_mounting: Literal["shoulder", "overhead", "median", "unknown"] = "unknown"
    barrier_type: Literal["w_beam", "thrie_beam", "concrete", "wire_rope", "unknown"] = "unknown"
    kerb_type: str | None = None
    osm_highway: str | None = None
    osm_tags: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_total_lane_alias_consistent(self) -> "RoadContext":
        if (
            self.lanes_total is not None
            and self.total_road_lanes is not None
            and self.lanes_total != self.total_road_lanes
        ):
            raise ValueError("lanes_total and total_road_lanes must describe the same verified total")
        if self.total_road_lanes is None and self.lanes_total is not None:
            self.total_road_lanes = self.lanes_total
        elif self.lanes_total is None and self.total_road_lanes is not None:
            self.lanes_total = self.total_road_lanes
        return self

    @property
    def radius_speed_kmph(self) -> float | None:
        """Best available speed for a speed-indexed radius rule.

        Verified design speed always wins. Posted speed is a deliberately
        labelled fallback and therefore produces a provisional assessment.
        """

        return self.design_speed_kmph or self.posted_speed_kmph

    @property
    def radius_speed_basis(self) -> str:
        if self.design_speed_kmph is not None:
            return "verified_design_speed"
        if self.posted_speed_kmph is not None:
            source = (self.posted_speed_source or "").casefold()
            if "openstreetmap" in source or "osm" in source:
                return "osm_posted_speed_proxy"
            return "detected_posted_speed_proxy"
        return "unknown"

    def compact_description(self) -> str:
        fields: list[str] = []
        for label, value in (
            ("jurisdiction", self.jurisdiction),
            ("road class", self.road_class),
            ("highway reference", self.highway_ref),
            ("setting", self.setting),
            ("terrain", self.terrain),
            ("carriageway", self.carriageway),
            ("lanes on OSM way", self.osm_way_lanes),
            ("lanes per carriageway", self.lanes_per_carriageway),
            ("opposite carriageway lanes", self.opposite_carriageway_lanes),
            ("raw OSM total lanes", self.osm_total_lanes),
            ("carriageway count", self.carriageway_count),
            ("base through-lane total", self.total_road_lanes),
            (
                "lane configuration status",
                "provisional" if self.lane_configuration_provisional else None,
            ),
            (
                "verified design speed",
                f"{self.design_speed_kmph:g} km/h ({self.design_speed_source or 'verified input'})"
                if self.design_speed_kmph
                else None,
            ),
            (
                "posted speed",
                f"{self.posted_speed_kmph:g} km/h ({self.posted_speed_source or 'unverified source'})"
                if self.posted_speed_kmph
                else None,
            ),
            (
                "radius speed basis",
                self.radius_speed_basis if self.radius_speed_basis != "unknown" else None,
            ),
            ("sign class", self.sign_class),
            ("sign mounting", self.sign_mounting),
            ("barrier type", self.barrier_type),
        ):
            if value not in (None, "", "unknown"):
                fields.append(f"{label}: {value}")
        return "; ".join(fields) or "jurisdiction: India; road context otherwise unknown"


class Citation(StrictModel):
    evidence_id: str
    standard_id: str | None = None
    edition_year: int | None = None
    source: str
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    quote: str = Field(min_length=1, max_length=700)
    retrieval_score: float | None = None
    content_hash: str | None = None

    @property
    def label(self) -> str:
        standard = self.standard_id or self.source
        page = f", p. {self.page}" if self.page else ""
        section = f", {self.section}" if self.section else ""
        return f"{standard}{page}{section}"


class RetrievalHit(StrictModel):
    evidence_id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    standard_id: str | None = None
    edition_year: int | None = None
    chunk_index: int | None = None
    content_hash: str | None = None
    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRuleExtraction(StrictModel):
    """Schema sent to Ollama. Source metadata is attached after validation."""

    status: Literal["found", "not_found", "ambiguous", "not_applicable"]
    evidence_id: str | None = None
    verbatim_quote: str | None = Field(default=None, max_length=700)
    raw_value: float | None = None
    raw_unit: Literal["mm", "cm", "m", "km", "km/h", "none"] | None = None
    comparator: Literal[">=", ">", "<=", "<", "=", "range"] | None = None
    second_raw_value: float | None = None
    applies_to: str | None = None
    conditions: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=900)


class ThresholdResult(StrictModel):
    metric_key: str
    metric_name: str
    status: RuleStatus
    value_m: float | None = None
    second_value_m: float | None = None
    comparator: Literal[">=", ">", "<=", "<", "=", "range"] | None = None
    citation: Citation | None = None
    conditions: list[str] = Field(default_factory=list)
    applicability_basis: str | None = None
    provisional: bool = False
    missing_context: list[str] = Field(default_factory=list)
    # This is a deterministic, descriptive score for evidence completeness and
    # retrieval quality. It is deliberately not presented as a probability.
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # These remain empty until a version-matched calibration artifact has been
    # fitted and validated against independently reviewed gold cases.
    calibrated_correctness_probability: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    calibration_id: str | None = None
    reason: str
    alternatives: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def found_requires_verified_value_and_citation(self) -> "ThresholdResult":
        if self.status == RuleStatus.FOUND:
            if self.value_m is None or self.citation is None or self.comparator is None:
                raise ValueError("A found rule requires value_m, comparator, and citation")
        return self

    @property
    def audit_ready(self) -> bool:
        # FOUND is assigned only after deterministic evidence, applicability,
        # source-policy, and edition checks pass. A heuristic score is not an
        # acceptance gate and is never represented as a correctness probability.
        return self.status == RuleStatus.FOUND


class ThresholdSet(StrictModel):
    road_context: RoadContext
    results: dict[str, ThresholdResult]
    run_id: str
    collection_name: str
    embedding_model: str
    llm_model: str
    warnings: list[str] = Field(default_factory=list)

    def legacy_values(self) -> dict[str, Any]:
        """Compatibility adapter for the original audit script.

        Only audit-ready values are exposed. Ambiguous/unsupported rules become
        None so the compliance layer cannot accidentally treat a guess as law.
        """

        values: dict[str, Any] = {
            key: result.value_m if result.audit_ready else None
            for key, result in self.results.items()
        }
        values["source_citations"] = sorted(
            {
                result.citation.label
                for result in self.results.values()
                if result.audit_ready and result.citation
            }
        )
        values["rule_details"] = {
            key: result.model_dump(mode="json") for key, result in self.results.items()
        }
        return values
