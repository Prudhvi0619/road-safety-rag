from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path

from .catalog import METRICS, MetricSpec, get_metric
from .config import Settings
from .models import (
    Citation,
    LLMRuleExtraction,
    RetrievalHit,
    RoadContext,
    RuleStatus,
    ThresholdResult,
    ThresholdSet,
)
from .ollama_client import OllamaClient, OllamaUnavailable
from .registry import StandardsRegistry
from .retrieval import (
    HybridRetriever,
    configuration_missing_context,
    format_evidence,
    standard_configuration_status,
    tokenize,
)
from .structured_evidence import StructuredEvidenceRegistry

NUMBER_WITH_UNIT_RE = re.compile(
    r"(?<![\w.])(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>mm|cm|km/h|km|m|metres?|meters?)\b",
    re.IGNORECASE,
)
TOLERANCE_WITH_UNIT_RE = re.compile(
    r"(?<![\w.])(?P<center>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?:±|\+\s*/\s*-|\+\s*-|plus\s+or\s+minus)\s*"
    r"(?P<tolerance>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|km|m|metres?|meters?)\b",
    re.IGNORECASE,
)

COMPARATOR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (">=", (r"\bminimum\b", r"\bat\s+least\b", r"\bnot\s+less\s+than\b")),
    ("<=", (r"\bmaximum\b", r"\bat\s+most\b", r"\bnot\s+more\s+than\b", r"\bshall\s+not\s+exceed\b")),
    (">", (r"\bgreater\s+than\b", r"\bmore\s+than\b")),
    ("<", (r"\bless\s+than\b", r"\bbelow\b")),
    (
        "=",
        (
            r"\bshall\s+be\b",
            r"\bequal\s+to\b",
            r"\bstandard\b.{0,50}\b(?:is|shall\s+be)\b",
        ),
    ),
)


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def convert_to_metres(value: float, unit: str) -> float:
    normalized = unit.casefold()
    if normalized == "mm":
        return value / 1000.0
    if normalized == "cm":
        return value / 100.0
    if normalized in {"m", "metre", "metres", "meter", "meters"}:
        return value
    if normalized == "km":
        return value * 1000.0
    raise ValueError(f"Unit '{unit}' is not a length unit")


def quote_contains_measurement(quote: str, raw_value: float, raw_unit: str) -> bool:
    for match in NUMBER_WITH_UNIT_RE.finditer(quote):
        value = float(match.group("value").replace(",", ""))
        unit = match.group("unit").casefold()
        if math.isclose(value, raw_value, rel_tol=1e-6, abs_tol=1e-6):
            canonical = "m" if unit in {"metre", "metres", "meter", "meters"} else unit
            requested = "m" if raw_unit in {"metre", "metres", "meter", "meters"} else raw_unit.casefold()
            if canonical == requested:
                return True
    return False


def quote_supports_tolerance_range(
    quote: str, first_value: float, second_value: float, raw_unit: str
) -> bool:
    """Verify endpoints that are derived from an explicit centre +/- tolerance."""

    requested = raw_unit.casefold()
    requested = "m" if requested in {"metre", "metres", "meter", "meters"} else requested
    expected = sorted((first_value, second_value))
    for match in TOLERANCE_WITH_UNIT_RE.finditer(quote):
        unit = match.group("unit").casefold()
        unit = "m" if unit in {"metre", "metres", "meter", "meters"} else unit
        if unit != requested:
            continue
        center = float(match.group("center").replace(",", ""))
        tolerance = float(match.group("tolerance"))
        derived = sorted((center - tolerance, center + tolerance))
        if all(
            math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-6)
            for actual, wanted in zip(derived, expected)
        ):
            return True
    return False


def comparator_from_quote(quote: str) -> str | None:
    """Return only a relationship explicitly supported by normative wording."""

    normalized = normalize_for_match(quote)
    for comparator, patterns in COMPARATOR_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return comparator
    return None


def quote_supports_comparator(
    quote: str, comparator: str, metric_key: str | None = None
) -> bool:
    if comparator == "range":
        return True
    if (
        metric_key == "min_lane_width"
        and comparator == ">="
        and re.search(
            r"(?i)\b(?:standard\s+)?lane\s+width\b.{0,100}\bshall\s+be\b",
            quote,
        )
    ):
        # This project performs a lower-bound screening check. The cited design
        # dimension is not presented as a construction-tolerance decision.
        return True
    return comparator_from_quote(quote) == comparator


class StandardsRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        llm: OllamaClient | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.retriever = retriever or HybridRetriever(self.settings)
        self.llm = llm or OllamaClient(self.settings)
        registry_path = self.settings.standards_registry or (
            self.settings.project_dir / "config" / "standards_registry.json"
        )
        self.registry = StandardsRegistry.load(registry_path)
        self.structured_evidence_path = (
            self.settings.project_dir / "config" / "structured_evidence.json"
        )
        self.structured_evidence = StructuredEvidenceRegistry.load(
            self.structured_evidence_path
        )

    def extract_metric(
        self,
        metric_key: str,
        context: RoadContext,
        retrieved_hits: list[RetrievalHit] | None = None,
    ) -> ThresholdResult:
        metric = get_metric(metric_key)
        missing = self._missing_context(metric, context)
        hits = (
            list(retrieved_hits)
            if retrieved_hits is not None
            else self.retriever.retrieve(metric, context)
        )
        structured_hits = self.structured_evidence.hits(
            metric.key, self.settings.persist_directory / "index_manifest.json"
        )
        if structured_hits:
            structured_ids = {hit.evidence_id for hit in structured_hits}
            hits = structured_hits + [
                hit for hit in hits if hit.evidence_id not in structured_ids
            ]
        if not hits:
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=RuleStatus.NOT_FOUND,
                reason="No relevant evidence was retrieved from the indexed standards corpus.",
            )
        if missing:
            # Some project-highway manuals specify curve radius directly by
            # terrain using desirable/absolute columns.  If that exact table is
            # present, terrain is sufficient for the desirable-minimum value.
            if metric.key == "min_radius_curvature":
                deterministic = self._explicit_clause_extraction(metric, context, hits)
                if deterministic is not None and self._is_direct_radius_table_extraction(
                    deterministic
                ):
                    recovered = self._validate(metric, context, deterministic, hits)
                    if recovered.status in {
                        RuleStatus.FOUND,
                        RuleStatus.AMBIGUOUS,
                        RuleStatus.NEEDS_CONTEXT,
                    }:
                        return self._with_radius_basis(
                            metric, context, recovered, direct_table=True
                        )
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=RuleStatus.NEEDS_CONTEXT,
                missing_context=missing,
                reason=(
                    "A defensible single value cannot be selected until these applicability facts are supplied: "
                    + ", ".join(missing)
                ),
                alternatives=self._alternative_citations(hits),
            )

        # Prefer a narrow deterministic parser whenever the evidence contains
        # one explicit clause/table row. This is faster and more reproducible
        # than asking a generative model to reinterpret straightforward text or
        # drawing geometry. Complex/conditional evidence still falls through.
        deterministic = self._explicit_clause_extraction(metric, context, hits)
        if deterministic is not None:
            recovered = self._validate(metric, context, deterministic, hits)
            if recovered.status in {RuleStatus.FOUND, RuleStatus.AMBIGUOUS}:
                return self._with_radius_basis(
                    metric,
                    context,
                    recovered,
                    direct_table=self._is_direct_radius_table_extraction(deterministic),
                )

        evidence_limit = 45000 if self.settings.exhaustive_retrieval else 18000
        try:
            extraction = self.llm.extract_rule(
                metric, context, format_evidence(hits, max_chars=evidence_limit)
            )
        except (OllamaUnavailable, RuntimeError) as exc:
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=RuleStatus.INVALID_EVIDENCE,
                reason=(
                    "Evidence was retrieved, but Ollama could not complete the remaining "
                    f"complex extraction: {exc}"
                ),
                alternatives=self._alternative_citations(hits),
            )
        result = self._validate(metric, context, extraction, hits)
        # Small local models sometimes copy the start of the right chunk instead
        # of the exact clause. Recover only a simple, explicit single-value
        # clause; tables, ranges, and conditional cases still require the LLM or
        # manual review. This path never invents text or values.
        if result.status == RuleStatus.INVALID_EVIDENCE and isinstance(self.llm, OllamaClient):
            deterministic = self._explicit_clause_extraction(metric, context, hits)
            if deterministic is not None:
                recovered = self._validate(metric, context, deterministic, hits)
                if recovered.status in {RuleStatus.FOUND, RuleStatus.AMBIGUOUS}:
                    return self._with_radius_basis(
                        metric,
                        context,
                        recovered,
                        direct_table=self._is_direct_radius_table_extraction(deterministic),
                    )
        return self._with_radius_basis(metric, context, result, direct_table=False)

    @staticmethod
    def _is_direct_radius_table_extraction(extraction: LLMRuleExtraction) -> bool:
        return "applicable terrain row" in extraction.rationale.casefold()

    def _with_radius_basis(
        self,
        metric: MetricSpec,
        context: RoadContext,
        result: ThresholdResult,
        *,
        direct_table: bool,
    ) -> ThresholdResult:
        """Expose whether radius selection used authoritative or proxy context."""

        if metric.key != "min_radius_curvature" or result.value_m is None:
            return result
        conditions = list(result.conditions)
        reason = result.reason
        prior_basis = result.applicability_basis
        if direct_table:
            basis = "Applicable IRC terrain/carriageway table (no speed proxy used)"
            note = (
                "The threshold was selected directly from the applicable IRC "
                "terrain/carriageway table; OSM posted speed was not used."
            )
            provisional = result.provisional
            evidence_quality_score = result.evidence_quality_score
        elif context.radius_speed_basis == "verified_design_speed":
            basis = (
                f"Verified design speed: {context.design_speed_kmph:g} km/h "
                f"({context.design_speed_source or 'verified input'})"
            )
            note = "The threshold was selected using verified highway design speed."
            provisional = result.provisional
            evidence_quality_score = result.evidence_quality_score
        elif context.radius_speed_basis == "osm_posted_speed_proxy":
            basis = (
                f"OSM posted-speed proxy: {context.posted_speed_kmph:g} km/h "
                f"({context.posted_speed_source or 'OpenStreetMap maxspeed'})"
            )
            note = (
                "PROVISIONAL: OSM maxspeed is a posted operating limit, not verified "
                "highway design speed; confirm against design records before statutory use."
            )
            provisional = True
            evidence_quality_score = min(result.evidence_quality_score, 0.78)
        elif context.radius_speed_basis == "detected_posted_speed_proxy":
            basis = (
                f"Detected posted-speed proxy: {context.posted_speed_kmph:g} km/h "
                f"({context.posted_speed_source or 'input observation'})"
            )
            note = (
                "PROVISIONAL: a detected posted-speed sign is not verified highway design "
                "speed; confirm against design records before statutory use."
            )
            provisional = True
            evidence_quality_score = min(result.evidence_quality_score, 0.74)
        else:
            return result
        if note not in conditions:
            conditions.append(note)
        if note not in reason:
            reason = f"{reason} {note}"
        if prior_basis and prior_basis not in basis:
            basis = f"{basis}; {prior_basis}"
        return result.model_copy(
            update={
                "conditions": conditions,
                "applicability_basis": basis,
                "provisional": provisional,
                "evidence_quality_score": evidence_quality_score,
                "reason": reason,
            }
        )

    @staticmethod
    def _explicit_clause_extraction(
        metric: MetricSpec, context: RoadContext, hits: list[RetrievalHit]
    ) -> LLMRuleExtraction | None:
        # Horizontal-radius requirements are commonly printed as compact table
        # rows.  They intentionally bypass the generic single-number clause
        # recovery below, but only when the table identifies both the terrain
        # and the desirable/absolute columns.  In the absence of an explicit
        # exception context, the ordinary (desirable minimum) value is used;
        # the lower absolute minimum is never silently substituted.
        if metric.key == "min_radius_curvature":
            terrain_pattern = {
                "plain": r"plain\s+and\s+rolling",
                "rolling": r"plain\s+and\s+rolling",
                "mountainous": r"mountainous\s+and\s+steep",
                "steep": r"mountainous\s+and\s+steep",
            }.get(context.terrain)
            if terrain_pattern:
                for hit in hits:
                    source_text = re.sub(r"\s+", " ", hit.text).strip()
                    table = re.search(
                        r"(?i)(?:radii\s+of\s+horizontal\s+curves|minimum\s+radii\s+of\s+horizontal\s+curves)"
                        r".{0,650}?desirable\s+minimum\s+radius.{0,180}?absolute\s+minimum\s+radius"
                        r".{0,220}?(?P<row>" + terrain_pattern
                        + r"\s+(?P<desirable>\d+(?:\.\d+)?)\s*m(?:etres?)?\s+"
                        r"(?P<absolute>\d+(?:\.\d+)?)\s*m(?:etres?)?)",
                        source_text,
                    )
                    if table:
                        value = float(table.group("desirable"))
                        return LLMRuleExtraction(
                            status="found",
                            evidence_id=hit.evidence_id,
                            verbatim_quote=table.group(0),
                            raw_value=value,
                            raw_unit="m",
                            comparator=metric.default_comparator,
                            applies_to=f"{context.terrain} terrain; desirable minimum radius",
                            conditions=[
                                "The absolute-minimum value requires an explicitly documented exception."
                            ],
                            rationale=(
                                "Recovered deterministically from the applicable terrain row and the "
                                "desirable-minimum column."
                            ),
                        )

        if metric.key == "min_w_beam_barrier_height":
            for hit in hits:
                source_text = re.sub(r"\s+", " ", hit.text).strip()
                tolerance = TOLERANCE_WITH_UNIT_RE.search(source_text)
                if not tolerance or not re.search(r"(?i)w[\s-]*beam", source_text):
                    continue
                center = float(tolerance.group("center").replace(",", ""))
                spread = float(tolerance.group("tolerance"))
                unit = tolerance.group("unit").casefold()
                canonical_unit = (
                    "m" if unit in {"metre", "metres", "meter", "meters"} else unit
                )
                return LLMRuleExtraction(
                    status="found",
                    evidence_id=hit.evidence_id,
                    verbatim_quote=source_text,
                    raw_value=center - spread,
                    second_raw_value=center + spread,
                    raw_unit=canonical_unit,
                    comparator="range",
                    applies_to="W-beam rail top height above the adjacent ground line",
                    conditions=[
                        "Evaluate as the full drawing tolerance, not as a one-sided minimum.",
                        "The input measurement must represent the W-beam rail top above the adjacent ground/road reference level.",
                    ],
                    rationale=(
                        "Recovered deterministically from a source-hash-bound transcription "
                        "of the dimension in IRC:119 Fig. 11."
                    ),
                )

        phrases = {
            " ".join(tokenize(phrase))
            for phrase in metric.search_phrases()
            if len(tokenize(phrase)) >= 2
        }
        low, high = metric.plausible_range_m
        for hit in hits:
            source_text = re.sub(r"\s+", " ", hit.text).strip()
            for clause in re.split(r"(?<=[.!?])\s+", source_text):
                normalized_clause = " ".join(tokenize(clause))
                semantic_match = StandardsRAG._clause_matches_metric(
                    metric.key, normalized_clause
                )
                if not any(phrase in normalized_clause for phrase in phrases) and not semantic_match:
                    continue
                measurements = [
                    match
                    for match in NUMBER_WITH_UNIT_RE.finditer(clause)
                    if match.group("unit").casefold() != "km/h"
                ]
                plausible: list[tuple[float, str]] = []
                for match in measurements:
                    raw_value = float(match.group("value").replace(",", ""))
                    raw_unit = match.group("unit")
                    try:
                        value_m = convert_to_metres(raw_value, raw_unit)
                    except ValueError:
                        continue
                    if low <= value_m <= high:
                        plausible.append((raw_value, raw_unit))
                if len(plausible) != 1:
                    continue
                raw_value, raw_unit = plausible[0]
                comparator = comparator_from_quote(clause)
                if comparator is None:
                    continue
                conditions: list[str] = []
                if metric.key == "min_lane_width" and comparator == "=":
                    comparator = metric.default_comparator
                    conditions.append(
                        "Screening interpretation: the cited standard lane dimension is "
                        "used as a lower-bound check; construction tolerances and approved "
                        "departures require separate engineering review."
                    )
                return LLMRuleExtraction(
                    status="found",
                    evidence_id=hit.evidence_id,
                    verbatim_quote=clause.strip(),
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    comparator=comparator,
                    applies_to=clause.strip(),
                    conditions=conditions,
                    rationale=(
                        "Recovered deterministically from one explicit evidence clause."
                    ),
                )
        return None

    @staticmethod
    def _clause_matches_metric(metric_key: str, clause: str) -> bool:
        concepts = {
            "min_lane_width": (("lane", "carriageway"), ("width",)),
            "min_sign_height": (("sign",), ("height", "lower edge", "lowest edge", "above ground")),
            "min_kerb_height": (("kerb", "curb"), ("height", "raised")),
            "min_w_beam_barrier_height": (("w beam",), ("height", "above ground")),
            "min_concrete_barrier_height": (("concrete barrier", "rigid barrier"), ("height",)),
            "min_radius_curvature": (("radius",), ("curve", "horizontal")),
        }
        groups = concepts.get(metric_key)
        return bool(groups) and all(any(term in clause for term in group) for group in groups)

    def extract_all(
        self,
        context: RoadContext,
        metric_keys: list[str] | None = None,
        use_cache: bool = True,
    ) -> ThresholdSet:
        keys = list(METRICS) if metric_keys is None else metric_keys
        cache_key = self._cache_key(context, keys)
        cache_path = self._cache_dir / f"{cache_key}.json"
        if use_cache and cache_path.exists():
            return ThresholdSet.model_validate_json(cache_path.read_text(encoding="utf-8"))

        results: dict[str, ThresholdResult] = {}
        for key in keys:
            metric_context = context
            # The DL/CV workbook has distinct W-beam and concrete-barrier
            # measurement columns.  Bind the applicable barrier profile to
            # each standards query instead of using one ambiguous global type.
            if key == "min_w_beam_barrier_height":
                metric_context = context.model_copy(update={"barrier_type": "w_beam"})
            elif key == "min_concrete_barrier_height":
                metric_context = context.model_copy(update={"barrier_type": "concrete"})
            results[key] = self.extract_metric(key, metric_context)
        warnings: list[str] = []
        if not self.settings.require_verified_standards:
            warnings.append(
                "PROVISIONAL SOURCE MODE: screening decisions may use the latest indexed "
                "edition even when its authenticity, amendments, and licence record have "
                "not been reviewer-verified."
            )
        if context.lane_configuration_provisional:
            warnings.append(
                "PROVISIONAL LANE CONFIGURATION: base through-lanes came from project/OSM/CV "
                "context rather than a reviewed design record."
            )
        if context.road_class_confidence is not None and context.road_class_confidence < 0.65:
            warnings.append(
                "Road classification confidence is below 0.65; road-class-dependent thresholds require review."
            )
        threshold_set = ThresholdSet(
            road_context=context,
            results=results,
            run_id=str(uuid.uuid4()),
            collection_name=self.settings.collection_name,
            embedding_model=self.settings.embedding_model,
            llm_model=self.settings.ollama_model,
            warnings=warnings,
        )
        if use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(threshold_set.model_dump_json(indent=2), encoding="utf-8")
        return threshold_set

    def _validate(
        self,
        metric: MetricSpec,
        context: RoadContext,
        extraction: LLMRuleExtraction,
        hits: list[RetrievalHit],
    ) -> ThresholdResult:
        if extraction.status != "found":
            status = {
                "not_found": RuleStatus.NOT_FOUND,
                "ambiguous": RuleStatus.AMBIGUOUS,
                "not_applicable": RuleStatus.NOT_APPLICABLE,
            }[extraction.status]
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=status,
                conditions=extraction.conditions,
                reason=extraction.rationale or f"The extractor returned {extraction.status}.",
                alternatives=self._alternative_citations(hits),
            )

        hit_by_id = {hit.evidence_id: hit for hit in hits}
        hit = hit_by_id.get(extraction.evidence_id or "")
        if hit is not None:
            configuration_status = standard_configuration_status(
                metric, context, hit.standard_id or ""
            )
            if configuration_status == "inapplicable":
                return ThresholdResult(
                    metric_key=metric.key,
                    metric_name=metric.name,
                    status=RuleStatus.INVALID_EVIDENCE,
                    reason=(
                        f"{hit.standard_id or hit.source} is not applicable to the verified "
                        f"{context.carriageway} / "
                        f"{context.total_road_lanes or 'unknown-total'}-lane configuration."
                    ),
                    alternatives=self._alternative_citations(hits, exclude=hit.evidence_id),
                )
            if configuration_status == "unknown":
                return ThresholdResult(
                    metric_key=metric.key,
                    metric_name=metric.name,
                    status=RuleStatus.NEEDS_CONTEXT,
                    missing_context=configuration_missing_context(context),
                    reason=(
                        f"{hit.standard_id or hit.source} is configuration-specific, but the "
                        "complete carriageway arrangement and total road lanes were not verified."
                    ),
                    alternatives=self._alternative_citations(hits, exclude=hit.evidence_id),
                )
        invalid_reason = self._evidence_error(metric, context, extraction, hit)
        if invalid_reason:
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=RuleStatus.INVALID_EVIDENCE,
                conditions=extraction.conditions,
                reason=invalid_reason,
                alternatives=self._alternative_citations(hits),
            )
        assert hit is not None
        assert extraction.verbatim_quote is not None
        assert extraction.raw_value is not None
        assert extraction.raw_unit is not None
        assert extraction.comparator is not None

        value_m = convert_to_metres(extraction.raw_value, extraction.raw_unit)
        second_value_m = None
        if extraction.comparator == "range":
            if extraction.second_raw_value is None:
                return ThresholdResult(
                    metric_key=metric.key,
                    metric_name=metric.name,
                    status=RuleStatus.INVALID_EVIDENCE,
                    reason="A range rule was returned without its second endpoint.",
                )
            second_value_m = convert_to_metres(extraction.second_raw_value, extraction.raw_unit)

        low, high = metric.plausible_range_m
        if not low <= value_m <= high or (
            second_value_m is not None and not low <= second_value_m <= high
        ):
            return ThresholdResult(
                metric_key=metric.key,
                metric_name=metric.name,
                status=RuleStatus.INVALID_EVIDENCE,
                reason=(
                    f"Converted value {value_m:g} m is outside the configured engineering sanity range "
                    f"({low:g}–{high:g} m). The source or unit requires manual review."
                ),
            )

        citation = Citation(
            evidence_id=hit.evidence_id,
            standard_id=hit.standard_id,
            edition_year=hit.edition_year,
            source=hit.source,
            page=hit.page,
            section=hit.section,
            quote=extraction.verbatim_quote,
            retrieval_score=round(hit.score, 6),
            content_hash=hit.content_hash,
        )
        evidence_quality_score, quality_notes, blockers = self._evidence_quality(
            metric, context, hit, hits
        )
        status = RuleStatus.AMBIGUOUS if blockers else RuleStatus.FOUND
        reason = extraction.rationale or "The value is directly supported by the cited evidence."
        review_items = [*blockers, *quality_notes]
        if review_items:
            reason = f"{reason} Review note: {'; '.join(review_items)}."
        conditions = list(extraction.conditions)
        provisional_reasons: list[str] = []
        policy = self.registry.get(hit.standard_id)
        if (
            status == RuleStatus.FOUND
            and not self.settings.require_verified_standards
            and not (policy and policy.verified)
        ):
            provisional_reasons.append(
                "source/edition is indexed but not reviewer-verified"
            )
            conditions.append(
                "PROVISIONAL SCREENING: confirm the standard edition, source authenticity, "
                "and amendments before statutory or contractual use."
            )
        if (
            status == RuleStatus.FOUND
            and context.lane_configuration_provisional
            and metric.key in {"min_lane_width", "min_radius_curvature"}
        ):
            provisional_reasons.append(
                "base through-lane configuration is inferred from workbook/OSM evidence"
            )
            evidence_quality_score = min(evidence_quality_score, 0.76)
            conditions.append(
                "PROVISIONAL APPLICABILITY: confirm the base through-lane configuration "
                "against a reviewed route record or design drawing."
            )
        provisional = bool(provisional_reasons)
        applicability_basis = (
            "Provisional screening: " + "; ".join(provisional_reasons)
            if provisional_reasons
            else None
        )
        return ThresholdResult(
            metric_key=metric.key,
            metric_name=metric.name,
            status=status,
            # Keep a fully evidence-validated candidate visible even when an
            # deterministic source/edition blockers prevent compliance use.
            # audit_ready remains false for AMBIGUOUS.
            value_m=value_m,
            second_value_m=second_value_m,
            comparator=extraction.comparator,
            citation=citation,
            conditions=list(dict.fromkeys(conditions)),
            applicability_basis=applicability_basis,
            provisional=provisional,
            evidence_quality_score=evidence_quality_score,
            reason=reason,
            alternatives=self._alternative_citations(hits, exclude=hit.evidence_id),
        )

    @staticmethod
    def _evidence_error(
        metric: MetricSpec,
        context: RoadContext,
        extraction: LLMRuleExtraction,
        hit: RetrievalHit | None,
    ) -> str | None:
        if hit is None:
            return "The model cited an evidence ID that was not supplied by retrieval."
        if not extraction.verbatim_quote:
            return "The model did not provide a verbatim evidence quote."
        if normalize_for_match(extraction.verbatim_quote) not in normalize_for_match(hit.text):
            return "The claimed quote is not present in the cited retrieved chunk."
        if extraction.raw_value is None or not extraction.raw_unit:
            return "The model did not provide a numeric value and explicit unit."
        if extraction.raw_unit in {"none", "km/h"}:
            return f"The extracted unit '{extraction.raw_unit}' is not a length unit for {metric.name}."
        if not extraction.comparator:
            return "The source relationship (minimum, maximum, exact, or range) was not identified."
        if extraction.comparator == "range":
            if extraction.second_raw_value is None:
                return "A range rule was returned without its second endpoint."
            endpoints_are_explicit = quote_contains_measurement(
                extraction.verbatim_quote, extraction.raw_value, extraction.raw_unit
            ) and quote_contains_measurement(
                extraction.verbatim_quote,
                extraction.second_raw_value,
                extraction.raw_unit,
            )
            tolerance_is_explicit = quote_supports_tolerance_range(
                extraction.verbatim_quote,
                extraction.raw_value,
                extraction.second_raw_value,
                extraction.raw_unit,
            )
            if not endpoints_are_explicit and not tolerance_is_explicit:
                return (
                    "The range endpoints are neither written explicitly nor derivable from an "
                    "explicit centre +/- tolerance in the evidence quote."
                )
        elif not quote_contains_measurement(
            extraction.verbatim_quote, extraction.raw_value, extraction.raw_unit
        ):
            return "The numeric value and unit do not occur together in the verbatim quote."

        claimed_scope = " ".join(
            [
                extraction.verbatim_quote,
                extraction.applies_to or "",
                *extraction.conditions,
            ]
        ).casefold()
        required_terms = {
            "min_lane_width": ("lane", "carriageway"),
            "min_sign_height": ("sign",),
            "traffic_sign_width": ("sign",),
            "traffic_sign_height": ("sign",),
            "min_kerb_height": ("kerb", "curb"),
            "min_w_beam_barrier_height": ("w-beam", "w beam", "metal beam"),
            "min_concrete_barrier_height": ("concrete", "new jersey", "rigid barrier"),
            "min_radius_curvature": ("radius", "curve"),
        }
        if not any(term in claimed_scope for term in required_terms.get(metric.key, ())):
            return "The quoted evidence does not identify the requested engineering feature."

        # A common false-positive in the legacy system was selecting a special
        # facility lane merely because it shared the same plausible dimension.
        if metric.key == "min_lane_width":
            special_facilities = (
                "toll lane",
                "etc lane",
                "shoulder width",
                "service road",
                "access road",
                "parking lane",
                "acceleration lane",
                "deceleration lane",
                "climbing lane",
            )
            supplied_context = context.compact_description().casefold()
            mismatches = [
                facility
                for facility in special_facilities
                if facility in claimed_scope and facility not in supplied_context
            ]
            if mismatches:
                return (
                    "The evidence applies to a different facility "
                    f"({', '.join(mismatches)}) than the supplied road context."
                )
        # Do not let a nearby comparison to W-beam make a Thrie-beam drawing
        # look applicable.  Barrier profiles have different installation
        # geometry, so cross-profile numeric reuse is unsafe.
        if metric.key == "min_w_beam_barrier_height" and any(
            term in claimed_scope for term in ("thrie-beam", "thrie beam")
        ):
            return "The evidence dimension applies to a Thrie-beam barrier, not a W-beam barrier."
        if extraction.comparator != "range" and not quote_supports_comparator(
            extraction.verbatim_quote, extraction.comparator, metric.key
        ):
            return (
                f"The comparator '{extraction.comparator}' is not supported by explicit "
                "normative wording in the evidence quote."
            )
        return None

    @staticmethod
    def _missing_context(metric: MetricSpec, context: RoadContext) -> list[str]:
        missing: list[str] = []
        for field in metric.required_context:
            value = getattr(context, field)
            if value in (None, "", "unknown"):
                missing.append(field)
        if (
            "road_class" in metric.required_context
            and context.road_class
            and (
                context.road_class_confidence is None
                or context.road_class_confidence < 0.65
            )
        ):
            missing.append("verified_road_class")
        return missing

    @staticmethod
    def _alternative_citations(
        hits: list[RetrievalHit], exclude: str | None = None, limit: int = 3
    ) -> list[Citation]:
        alternatives: list[Citation] = []
        for hit in hits:
            if hit.evidence_id == exclude or not hit.text.strip():
                continue
            quote = re.sub(r"\s+", " ", hit.text.strip())[:320]
            alternatives.append(
                Citation(
                    evidence_id=hit.evidence_id,
                    standard_id=hit.standard_id,
                    edition_year=hit.edition_year,
                    source=hit.source,
                    page=hit.page,
                    section=hit.section,
                    quote=quote,
                    retrieval_score=round(hit.score, 6),
                    content_hash=hit.content_hash,
                )
            )
            if len(alternatives) >= limit:
                break
        return alternatives

    def _evidence_quality(
        self,
        metric: MetricSpec,
        context: RoadContext,
        selected: RetrievalHit,
        hits: list[RetrievalHit],
    ) -> tuple[float, list[str], list[str]]:
        """Return a descriptive evidence-quality score and deterministic blockers.

        The score is intentionally heuristic and is not used as a probability or
        an acceptance threshold. Audit readiness is controlled only by explicit,
        explainable blockers returned from this method plus earlier validation.
        """

        score = 0.62
        score_cap = 1.0
        notes: list[str] = []
        blockers: list[str] = []
        if selected.page is not None:
            score += 0.08
        else:
            blockers.append("citation has no page number")
            score_cap = min(score_cap, 0.60)
        if selected.standard_id and any(
            selected.standard_id.upper().startswith(item.upper())
            for item in metric.preferred_standards
        ):
            score += 0.08
        else:
            blockers.append("source is outside the metric's preferred standard list")
        if not selected.standard_id:
            blockers.append("citation has no normalized standard identifier")
        if selected.edition_year is None:
            blockers.append("citation has no edition year")
        if selected in hits[:3]:
            score += 0.06

        editions = set(
            getattr(self.retriever, "standard_editions", {}).get(
                selected.standard_id, set()
            )
        ) or {
            hit.edition_year
            for hit in hits
            if hit.standard_id == selected.standard_id and hit.edition_year is not None
        }
        policy = self.registry.get(selected.standard_id)
        if policy and policy.verified:
            if selected.edition_year != policy.active_edition_year:
                score = 0.0
                blockers.append(
                    f"registry approves edition {policy.active_edition_year}, not {selected.edition_year}"
                )
            elif not self.registry.source_is_approved(
                policy,
                selected.source,
                str(
                    selected.metadata.get("document_sha256")
                    or selected.metadata.get("source_sha256")
                    or ""
                )
                or None,
            ):
                score = 0.0
                blockers.append(
                    "selected filename or document SHA-256 does not match the reviewer-approved registry entry"
                )
            else:
                score += 0.05
        elif self.settings.require_verified_standards:
            score_cap = min(score_cap, 0.55)
            blockers.append(
                "the standard edition and source are not reviewer-verified in the standards registry"
            )
        elif len(known_editions := {edition for edition in editions if edition is not None}) > 1:
            newest = max(known_editions)
            if selected.edition_year == newest:
                score_cap = min(score_cap, 0.72)
                notes.append(
                    "latest indexed edition selected automatically; source/edition is not reviewer-verified"
                )
            else:
                score = 0.0
                blockers.append(
                    f"a newer {selected.standard_id} edition ({newest}) is present in the corpus"
                )
        elif not (policy and policy.verified):
            score_cap = min(score_cap, 0.78)
            notes.append("source/edition is not reviewer-verified")

        context_terms = set(tokenize(context.compact_description()))
        evidence_terms = set(tokenize(selected.text))
        meaningful_context = {
            token
            for token in context_terms
            if token
            not in {
                "india",
                "jurisdiction",
                "road",
                "class",
                "setting",
                "total",
                "lanes",
            }
        }
        if meaningful_context and meaningful_context.intersection(evidence_terms):
            score += 0.05
        else:
            score -= 0.04
            notes.append("applicability wording has weak lexical overlap with the road context")
        if selected.page is None:
            score_cap = min(score_cap, 0.60)
        score = min(score, score_cap)
        return round(max(0.0, min(score, 1.0)), 3), notes, blockers

    @property
    def _cache_dir(self) -> Path:
        return self.settings.project_dir / ".rag_cache"

    def _cache_key(self, context: RoadContext, metric_keys: list[str]) -> str:
        manifest = self.settings.persist_directory / "index_manifest.json"
        manifest_stamp = manifest.stat().st_mtime_ns if manifest.exists() else 0
        registry_path = self.settings.standards_registry or (
            self.settings.project_dir / "config" / "standards_registry.json"
        )
        registry_stamp = registry_path.stat().st_mtime_ns if registry_path.exists() else 0
        structured_stamp = (
            self.structured_evidence_path.stat().st_mtime_ns
            if self.structured_evidence_path.exists()
            else 0
        )
        payload = {
            "context": context.model_dump(mode="json"),
            "metrics": metric_keys,
            "collection": self.settings.collection_name,
            "embedding": self.settings.embedding_model,
            "llm": self.settings.ollama_model,
            "manifest_stamp": manifest_stamp,
            "registry_stamp": registry_stamp,
            "structured_evidence_stamp": structured_stamp,
            "schema": 14,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
