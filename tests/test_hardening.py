from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from road_safety_rag.audit import RoadSafetyAuditPipeline
from road_safety_rag.config import Settings
from road_safety_rag.evaluation import GoldCase, _retrieval_matches
from road_safety_rag.ingestion import IndexBuilder
from road_safety_rag.models import (
    LLMRuleExtraction,
    RetrievalHit,
    RoadContext,
    RuleStatus,
    ThresholdResult,
    ThresholdSet,
)
from road_safety_rag.road_context import RoadTypeResolver
from road_safety_rag.service import StandardsRAG, comparator_from_quote


class StaticRetriever:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits
        self.standard_editions = {"IRC:73": {2022}}
        self.calls = 0

    def retrieve(self, metric, context):
        self.calls += 1
        return self.hits


class StaticLLM:
    def __init__(self, extraction: LLMRuleExtraction):
        self.extraction = extraction

    def extract_rule(self, metric, road_context, evidence):
        return self.extraction


def _settings(root: Path, *, strict: bool = True) -> Settings:
    return Settings(
        project_dir=root,
        corpus_dirs=(root / "corpus",),
        persist_directory=root / "db",
        require_verified_standards=strict,
    )


def _lane_hit(text: str = "The minimum traffic lane width shall be 3.5 m.") -> RetrievalHit:
    return RetrievalHit(
        evidence_id="E-lane",
        text=text,
        source="IRC 73.pdf",
        page=24,
        section="Lane width",
        standard_id="IRC:73",
        edition_year=2022,
        content_hash="hash-lane",
        score=0.9,
    )


def _lane_context() -> RoadContext:
    return RoadContext(
        road_class="National Highway",
        road_class_confidence=0.95,
        setting="rural",
        carriageway="undivided",
        lanes_total=2,
    )


def test_strict_source_policy_keeps_unreviewed_rule_out_of_compliance(tmp_path: Path):
    hit = _lane_hit()
    extraction = LLMRuleExtraction(
        status="found",
        evidence_id=hit.evidence_id,
        verbatim_quote=hit.text,
        raw_value=3.5,
        raw_unit="m",
        comparator=">=",
        rationale="Explicit minimum.",
    )
    rag = StandardsRAG(
        _settings(tmp_path, strict=True),
        StaticRetriever([hit]),
        StaticLLM(extraction),
    )

    result = rag.extract_metric("min_lane_width", _lane_context())

    assert result.status == RuleStatus.AMBIGUOUS
    assert not result.audit_ready
    assert result.value_m == 3.5
    assert "not reviewer-verified" in result.reason


def test_strict_source_policy_accepts_exact_reviewer_approved_file_and_hash(
    tmp_path: Path,
):
    document_hash = "a" * 64
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "standards_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "standards": {
                    "IRC:73": {
                        "active_edition_year": 2022,
                        "approved_sources": ["IRC 73.pdf"],
                        "approved_sha256": [document_hash],
                        "official_source_url": "https://example.invalid/authoritative-record",
                        "licence_basis": "Institutional licensed copy",
                        "amendments": [],
                        "supersedes": None,
                        "reviewed_by": "domain-reviewer",
                        "reviewed_on": "2026-09-01",
                        "notes": "Test fixture",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    hit = _lane_hit().model_copy(update={"metadata": {"document_sha256": document_hash}})
    extraction = LLMRuleExtraction(
        status="found",
        evidence_id=hit.evidence_id,
        verbatim_quote=hit.text,
        raw_value=3.5,
        raw_unit="m",
        comparator=">=",
        rationale="Explicit minimum.",
    )
    rag = StandardsRAG(
        _settings(tmp_path, strict=True),
        StaticRetriever([hit]),
        StaticLLM(extraction),
    )

    result = rag.extract_metric("min_lane_width", _lane_context())

    assert result.status == RuleStatus.FOUND
    assert result.audit_ready


def test_comparator_must_come_from_normative_wording():
    assert comparator_from_quote("The minimum width shall be 3.5 m") == ">="
    assert comparator_from_quote("The width shall be 3.5 m") == "="
    assert comparator_from_quote("The width is commonly 3.5 m") is None


def test_exact_dimension_does_not_use_an_invented_one_percent_tolerance():
    from road_safety_rag.audit import evaluate_measurements

    rule = ThresholdResult(
        metric_key="min_kerb_height",
        metric_name="kerb height",
        status=RuleStatus.FOUND,
        value_m=0.15,
        comparator="=",
        citation={
            "evidence_id": "E-kerb",
            "standard_id": "IRC:86",
            "source": "IRC 86.pdf",
            "page": 10,
            "quote": "The kerb height shall be 0.15 m.",
        },
        reason="Exact specified dimension.",
    )

    assert evaluate_measurements([0.15], rule).status == "PASS"
    assert evaluate_measurements([0.151], rule).status == "REVIEW_REQUIRED"


def test_gold_retrieval_requires_the_expected_passage_not_only_metadata():
    case = GoldCase(
        case_id="lane-1",
        metric_key="min_lane_width",
        road_context=_lane_context(),
        expected_standard_id="IRC:73",
        expected_edition_year=2022,
        expected_page=24,
        expected_quote="minimum traffic lane width shall be 3.5 m",
    )
    wrong = _lane_hit("The minimum shoulder width shall be 2.5 m.")
    right = _lane_hit()

    assert not _retrieval_matches(wrong, case)
    assert _retrieval_matches(right, case)


def test_found_gold_case_without_passage_locator_is_rejected():
    with pytest.raises(ValidationError):
        GoldCase(
            case_id="invalid",
            metric_key="min_lane_width",
            road_context=_lane_context(),
            expected_standard_id="IRC:73",
        )


def test_precomputed_hits_prevent_duplicate_retrieval(tmp_path: Path):
    hit = _lane_hit()
    retriever = StaticRetriever([hit])
    extraction = LLMRuleExtraction(
        status="found",
        evidence_id=hit.evidence_id,
        verbatim_quote=hit.text,
        raw_value=3.5,
        raw_unit="m",
        comparator=">=",
        rationale="Explicit minimum.",
    )
    rag = StandardsRAG(_settings(tmp_path, strict=False), retriever, StaticLLM(extraction))

    result = rag.extract_metric("min_lane_width", _lane_context(), retrieved_hits=[hit])

    assert result.status == RuleStatus.FOUND
    assert retriever.calls == 0


def test_research_mode_uses_latest_indexed_edition_as_provisional(tmp_path: Path):
    hit = _lane_hit().model_copy(update={"edition_year": 2022})
    retriever = StaticRetriever([hit])
    retriever.standard_editions = {"IRC:73": {2001, 2022}}
    extraction = LLMRuleExtraction(
        status="found",
        evidence_id=hit.evidence_id,
        verbatim_quote=hit.text,
        raw_value=3.5,
        raw_unit="m",
        comparator=">=",
        rationale="Explicit minimum.",
    )
    result = StandardsRAG(
        _settings(tmp_path, strict=False), retriever, StaticLLM(extraction)
    ).extract_metric("min_lane_width", _lane_context())

    assert result.status == RuleStatus.FOUND
    assert result.provisional
    assert "latest indexed edition selected automatically" in result.reason


def test_prune_removes_missing_sources_only_inside_configured_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    database = tmp_path / "db"
    database.mkdir()
    stale = corpus / "old.pdf"
    outside = tmp_path / "other" / "keep.pdf"
    manifest = {
        "schema_version": 3,
        "chunking_version": "table-aware-v1",
        "documents": {
            str(stale.resolve()): {"sha256": "old", "chunks": 2},
            str(outside.resolve()): {"sha256": "outside", "chunks": 1},
        },
    }
    (database / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    deleted: list[str] = []

    class Collection:
        def delete(self, *, where):
            deleted.append(where["source_path"])

    class Builder(IndexBuilder):
        def _vector_store(self):
            return SimpleNamespace(_collection=Collection())

    summary = Builder(_settings(tmp_path)).build(prune=True)
    updated = json.loads((database / "index_manifest.json").read_text(encoding="utf-8"))

    assert summary["pruned_documents"] == 1
    assert deleted == [str(stale.resolve())]
    assert str(stale.resolve()) not in updated["documents"]
    assert str(outside.resolve()) in updated["documents"]


def test_excel_to_html_report_smoke_path(tmp_path: Path):
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "audit.xlsx"
    html_path = tmp_path / "audit.html"
    pd.DataFrame(
        {
            "Timestamp (sec)": [1.0],
            "Raw OCR Text": ["N: 17.6610 E: 78.1068"],
            "Expected Total Lanes": [2],
            "Centre Lanes Detected": [1],
            "Shoulder Lanes Detected": [2],
            "Tracked Lane Widths (m)": ["3.6, 3.7"],
        }
    ).to_excel(input_path, index=False)
    lane_rule = ThresholdResult(
        metric_key="min_lane_width",
        metric_name="minimum traffic lane width",
        status=RuleStatus.FOUND,
        value_m=3.5,
        comparator=">=",
        citation={
            "evidence_id": "E-lane",
            "standard_id": "IRC:73",
            "edition_year": 2022,
            "source": "IRC 73.pdf",
            "page": 24,
            "quote": "The minimum traffic lane width shall be 3.5 m.",
        },
        reason="Verified test fixture.",
    )

    class FakeRAG:
        settings = SimpleNamespace(project_dir=tmp_path)

        def extract_all(self, context, metric_keys):
            return ThresholdSet(
                road_context=context,
                results={"min_lane_width": lane_rule},
                run_id="test-run",
                collection_name="test",
                embedding_model="test",
                llm_model="test",
            )

    context = _lane_context()
    RoadSafetyAuditPipeline(input_path, FakeRAG(), context).run(output_path, html_output=html_path)

    assert output_path.exists()
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Evidence-bound Road Safety Audit" in html
    assert "Network notice" in html
    assert "Lane 1: 3.6m (PASS)" in html


def test_osm_route_matching_prefers_trajectory_coverage_over_nearby_way_count():
    class Resolver(RoadTypeResolver):
        def _features_near(self, points):
            route = {
                "highway": "trunk",
                "ref": "NH 65",
                "lanes": "2",
                "oneway": "no",
                "_osm_id": "route",
                "_geometry": [
                    {"lat": 17.66, "lon": 78.10},
                    {"lat": 17.66, "lon": 78.11},
                ],
            }
            crossings = [
                {
                    "highway": "primary",
                    "ref": f"SH {index}",
                    "lanes": "2",
                    "oneway": "no",
                    "_osm_id": f"crossing-{index}",
                    "_geometry": [
                        {"lat": 17.655, "lon": 78.105 + index * 0.0001},
                        {"lat": 17.665, "lon": 78.105 + index * 0.0001},
                    ],
                }
                for index in range(3)
            ]
            return [route, *crossings]

        def _elevations(self, points):
            return [100.0 for _point in points]

    trajectory = [(17.66, 78.10 + index * 0.001) for index in range(11)]
    context = Resolver().resolve(trajectory)

    assert context.road_class == "National Highway"
    assert "map-match quality" in " ".join(context.notes)
