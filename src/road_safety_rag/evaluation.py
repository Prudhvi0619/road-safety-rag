from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .audit import evaluate_measurements
from .catalog import get_metric
from .config import Settings
from .models import Citation, RetrievalHit, RoadContext, RuleStatus, StrictModel
from .retrieval import HybridRetriever
from .service import StandardsRAG

ABSTENTION_STATUSES = {
    RuleStatus.NOT_FOUND.value,
    RuleStatus.AMBIGUOUS.value,
    RuleStatus.NEEDS_CONTEXT.value,
    RuleStatus.INVALID_EVIDENCE.value,
}


class GoldCase(StrictModel):
    """One independently reviewed retrieval, extraction, and audit decision."""

    case_id: str
    metric_key: str
    road_context: RoadContext
    expected_standard_id: str | None = None
    expected_edition_year: int | None = None
    expected_page: int | None = Field(default=None, ge=1)
    expected_evidence_id: str | None = None
    expected_content_hash: str | None = None
    expected_quote: str | None = None
    expected_value_m: float | None = Field(default=None, gt=0)
    expected_second_value_m: float | None = Field(default=None, gt=0)
    expected_comparator: str | None = None
    expected_status: str = RuleStatus.FOUND.value
    expected_applicable: bool | None = None
    observed_values_m: list[float] = Field(default_factory=list)
    expected_audit_decision: Literal[
        "PASS", "FAIL", "REVIEW_REQUIRED", "NOT_OBSERVED"
    ] | None = None
    reviewer_id: str | None = None
    reviewed_on: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def require_passage_locator_for_found_case(self) -> "GoldCase":
        if self.expected_status == RuleStatus.FOUND.value and not any(
            (self.expected_evidence_id, self.expected_content_hash, self.expected_quote)
        ):
            raise ValueError(
                "A found gold case requires expected_evidence_id, "
                "expected_content_hash, or expected_quote"
            )
        return self


def load_gold(path: Path) -> list[GoldCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Gold file must contain a non-empty JSON array")
    return [GoldCase.model_validate(item) for item in payload]


def evaluate_gold(
    settings: Settings,
    gold_path: Path,
    with_llm: bool = False,
) -> dict[str, object]:
    cases = load_gold(gold_path)
    retriever = HybridRetriever(settings)
    rag = StandardsRAG(settings, retriever=retriever) if with_llm else None
    case_results: list[dict[str, object]] = []

    for case in cases:
        total_started = time.perf_counter()
        metric = get_metric(case.metric_key)
        retrieval_started = time.perf_counter()
        hits = retriever.retrieve(metric, case.road_context)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        matching_ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if _retrieval_matches(hit, case)
        ]
        first_rank = min(matching_ranks) if matching_ranks else None
        record: dict[str, object] = {
            "case_id": case.case_id,
            "metric_key": case.metric_key,
            "first_matching_rank": first_rank,
            "retrieval_hit_at_5": bool(first_rank and first_rank <= 5),
            "retrieval_hit_at_10": bool(first_rank and first_rank <= 10),
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "retrieved": [
                {
                    "rank": rank,
                    "evidence_id": hit.evidence_id,
                    "standard_id": hit.standard_id,
                    "edition_year": hit.edition_year,
                    "source": hit.source,
                    "page": hit.page,
                    "content_hash": hit.content_hash,
                    "score": hit.score,
                }
                for rank, hit in enumerate(hits, start=1)
            ],
        }
        if with_llm and rag:
            extraction_started = time.perf_counter()
            result = rag.extract_metric(
                case.metric_key,
                case.road_context,
                retrieved_hits=hits,
            )
            extraction_latency_ms = (time.perf_counter() - extraction_started) * 1000
            standard_match = (
                bool(
                    result.citation
                    and result.citation.standard_id
                    and result.citation.standard_id.casefold()
                    == case.expected_standard_id.casefold()
                )
                if case.expected_standard_id
                else None
            )
            edition_match = (
                bool(
                    result.citation
                    and result.citation.edition_year == case.expected_edition_year
                )
                if case.expected_edition_year is not None
                else None
            )
            citation_match = (
                bool(
                    result.citation and _citation_matches(result.citation, case)
                )
                if case.expected_standard_id
                else None
            )
            value_match = (
                bool(
                    result.value_m is not None
                    and math.isclose(
                        result.value_m,
                        case.expected_value_m,
                        rel_tol=1e-4,
                        abs_tol=1e-4,
                    )
                    and (
                        case.expected_second_value_m is None
                        or (
                            result.second_value_m is not None
                            and math.isclose(
                                result.second_value_m,
                                case.expected_second_value_m,
                                rel_tol=1e-4,
                                abs_tol=1e-4,
                            )
                        )
                    )
                )
                if case.expected_value_m is not None
                else None
            )
            comparator_match = (
                result.comparator == case.expected_comparator
                if case.expected_comparator is not None
                else None
            )
            applicability_match = (
                (result.status != RuleStatus.NOT_APPLICABLE)
                == case.expected_applicable
                if case.expected_applicable is not None
                else None
            )
            expected_abstention = case.expected_status in ABSTENTION_STATUSES
            predicted_abstention = result.status.value in ABSTENTION_STATUSES
            status_match = result.status.value == case.expected_status
            audit_decision_match: bool | None = None
            predicted_audit_decision: str | None = None
            if case.expected_audit_decision is not None:
                check = evaluate_measurements(case.observed_values_m, result)
                predicted_audit_decision = check.status
                audit_decision_match = check.status == case.expected_audit_decision

            scored_checks = [
                status_match,
                standard_match,
                edition_match,
                citation_match,
                value_match,
                comparator_match,
                applicability_match,
                audit_decision_match,
            ]
            overall_correct = all(
                check for check in scored_checks if check is not None
            )
            record.update(
                {
                    "extraction": result.model_dump(mode="json"),
                    "status_match": status_match,
                    "standard_match": standard_match,
                    "edition_match": edition_match,
                    "citation_match": citation_match,
                    "value_match": value_match,
                    "comparator_match": comparator_match,
                    "applicability_match": applicability_match,
                    "expected_abstention": expected_abstention,
                    "predicted_abstention": predicted_abstention,
                    "predicted_audit_decision": predicted_audit_decision,
                    "audit_decision_match": audit_decision_match,
                    "overall_correct": overall_correct,
                    "extraction_latency_ms": round(extraction_latency_ms, 3),
                    "evidence_quality_score": result.evidence_quality_score,
                    "calibrated_correctness_probability": (
                        result.calibrated_correctness_probability
                    ),
                }
            )
        record["total_latency_ms"] = round(
            (time.perf_counter() - total_started) * 1000, 3
        )
        case_results.append(record)

    return {
        "protocol_version": "2.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gold_file": gold_path.name,
        "summary": summarize_case_results(case_results, with_llm=with_llm),
        "results": case_results,
    }


def summarize_case_results(
    records: list[dict[str, object]], with_llm: bool
) -> dict[str, object]:
    """Aggregate reviewed cases without treating missing labels as failures."""

    if not records:
        raise ValueError("At least one evaluation record is required")

    def mean_bool(key: str) -> float | None:
        values = [bool(row[key]) for row in records if row.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary: dict[str, object] = {
        "cases": len(records),
        "recall_at_5": mean_bool("retrieval_hit_at_5"),
        "recall_at_10": mean_bool("retrieval_hit_at_10"),
        "mean_reciprocal_rank": round(
            statistics.fmean(
                float(row.get("reciprocal_rank", 0.0)) for row in records
            ),
            4,
        ),
        "with_llm": with_llm,
        "latency_ms_by_metric": _latency_by_metric(records),
    }
    if not with_llm:
        return summary

    for output_key, record_key in (
        ("citation_accuracy", "citation_match"),
        ("standard_accuracy", "standard_match"),
        ("edition_accuracy", "edition_match"),
        ("value_accuracy", "value_match"),
        ("comparator_accuracy", "comparator_match"),
        ("applicability_accuracy", "applicability_match"),
        ("end_to_end_audit_decision_accuracy", "audit_decision_match"),
        ("overall_accuracy", "overall_correct"),
    ):
        summary[output_key] = mean_bool(record_key)

    expected = [bool(row.get("expected_abstention")) for row in records]
    predicted = [bool(row.get("predicted_abstention")) for row in records]
    true_positive = sum(e and p for e, p in zip(expected, predicted))
    predicted_positive = sum(predicted)
    actual_positive = sum(expected)
    summary["abstention_precision"] = (
        round(true_positive / predicted_positive, 4) if predicted_positive else None
    )
    summary["abstention_recall"] = (
        round(true_positive / actual_positive, 4) if actual_positive else None
    )
    return summary


def _latency_by_metric(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row["metric_key"])].append(float(row["total_latency_ms"]))
    output: dict[str, object] = {}
    for metric, values in sorted(grouped.items()):
        ordered = sorted(values)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        output[metric] = {
            "cases": len(ordered),
            "mean": round(statistics.fmean(ordered), 3),
            "median": round(statistics.median(ordered), 3),
            "p95": round(ordered[p95_index], 3),
        }
    return output


def _metadata_matches(
    standard_id: str | None, edition_year: int | None, page: int | None, case: GoldCase
) -> bool:
    if case.expected_standard_id and not standard_id:
        return False
    if (
        case.expected_standard_id
        and standard_id
        and standard_id.casefold() != case.expected_standard_id.casefold()
    ):
        return False
    if case.expected_edition_year is not None and edition_year != case.expected_edition_year:
        return False
    if case.expected_page is not None and page != case.expected_page:
        return False
    return True


def _retrieval_matches(hit: RetrievalHit, case: GoldCase) -> bool:
    if not _metadata_matches(hit.standard_id, hit.edition_year, hit.page, case):
        return False
    checks: list[bool] = []
    if case.expected_evidence_id:
        checks.append(hit.evidence_id == case.expected_evidence_id)
    if case.expected_content_hash:
        checks.append(hit.content_hash == case.expected_content_hash)
    if case.expected_quote:
        checks.append(_normalize(case.expected_quote) in _normalize(hit.text))
    return bool(checks) and all(checks)


def _citation_matches(citation: Citation, case: GoldCase) -> bool:
    if not _metadata_matches(
        citation.standard_id, citation.edition_year, citation.page, case
    ):
        return False
    checks: list[bool] = []
    if case.expected_evidence_id:
        checks.append(citation.evidence_id == case.expected_evidence_id)
    if case.expected_content_hash:
        checks.append(citation.content_hash == case.expected_content_hash)
    if case.expected_quote:
        checks.append(_normalize(case.expected_quote) in _normalize(citation.quote))
    return bool(checks) and all(checks)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
