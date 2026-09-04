from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from road_safety_rag.audit import RoadSafetyAuditPipeline, evaluate_measurements, extract_numbers
from road_safety_rag.catalog import get_metric
from road_safety_rag.cli import (
    build_parser,
    context_from_args,
    corpus_folder_settings,
    deep_retrieval_settings,
)
from road_safety_rag.config import Settings
from road_safety_rag.ingestion import PageChunker, PageExtractor, PageText, identify_standard
from road_safety_rag.models import (
    LLMRuleExtraction,
    RetrievalHit,
    RoadContext,
    RuleStatus,
    ThresholdResult,
)
from road_safety_rag.ollama_client import OllamaUnavailable
from road_safety_rag.retrieval import (
    BM25Index,
    HybridRetriever,
    standard_configuration_status,
    tokenize,
)
from road_safety_rag.road_context import RoadTypeResolver, parse_coordinate_text, sample_trajectory
from road_safety_rag.service import (
    StandardsRAG,
    convert_to_metres,
    quote_contains_measurement,
    quote_supports_tolerance_range,
)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.contexts = []

    def retrieve(self, metric, context):
        self.contexts.append((metric.key, context))
        return self.hits


class FakeLLM:
    def __init__(self, extraction):
        self.extraction = extraction
        self.calls = 0

    def extract_rule(self, metric, road_context, evidence):
        self.calls += 1
        return self.extraction


class TimeoutLLM:
    def extract_rule(self, metric, road_context, evidence):
        raise OllamaUnavailable("test timeout")


def settings(temp_dir: str) -> Settings:
    base = Path(temp_dir)
    return Settings(
        project_dir=base,
        corpus_dirs=(),
        persist_directory=base / "db",
        require_verified_standards=False,
    )


def lane_context() -> RoadContext:
    return RoadContext(
        road_class="National Highway",
        road_class_source="test",
        road_class_confidence=0.95,
        setting="rural",
        lanes_total=2,
    )


def lane_hit(year: int = 2022) -> RetrievalHit:
    return RetrievalHit(
        evidence_id=f"E-lane-{year}",
        text=(
            "Table 4 Rural National Highway, two-lane carriageway. "
            "The minimum traffic lane width shall be 3.5 m."
        ),
        source=f"IRC 73 {year}.pdf",
        page=18,
        section="4.2 Carriageway",
        standard_id="IRC:73",
        edition_year=year,
        chunk_index=4,
        content_hash=str(year),
        score=0.9,
    )


class EvidenceValidationTests(unittest.TestCase):
    def test_evidence_quality_is_descriptive_not_a_probability_gate(self):
        result = ThresholdResult(
            metric_key="min_lane_width",
            metric_name="minimum traffic lane width",
            status=RuleStatus.FOUND,
            value_m=3.5,
            comparator=">=",
            citation={
                "evidence_id": "E-reviewed",
                "standard_id": "IRC:73",
                "edition_year": 2022,
                "source": "IRC 73 2022.pdf",
                "page": 18,
                "quote": "The minimum traffic lane width shall be 3.5 m.",
            },
            evidence_quality_score=0.10,
            reason="All deterministic blockers have passed.",
        )
        payload = result.model_dump(mode="json")
        self.assertTrue(result.audit_ready)
        self.assertIn("evidence_quality_score", payload)
        self.assertNotIn("confidence", payload)
        self.assertIsNone(payload["calibrated_correctness_probability"])
        self.assertIsNone(payload["calibration_id"])

    def test_valid_measurement_becomes_found_rule(self):
        hit = lane_hit()
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            applies_to="Rural two-lane National Highway",
            conditions=["two-lane carriageway"],
            rationale="The selected row matches the supplied context.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), lane_context(), extraction, [hit])
        self.assertEqual(result.status, RuleStatus.FOUND)
        self.assertEqual(result.value_m, 3.5)
        self.assertEqual(result.citation.page, 18)
        self.assertTrue(result.audit_ready)

    def test_missing_edition_is_an_explicit_audit_blocker(self):
        hit = lane_hit().model_copy(update={"edition_year": None})
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            rationale="Numeric clause is present.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), lane_context(), extraction, [hit])
        self.assertEqual(result.status, RuleStatus.AMBIGUOUS)
        self.assertFalse(result.audit_ready)
        self.assertIn("no edition year", result.reason)

    def test_invented_quote_is_rejected(self):
        hit = lane_hit()
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The lane width shall be 3.75 m.",
            raw_value=3.75,
            raw_unit="m",
            comparator=">=",
            rationale="Claimed value",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), lane_context(), extraction, [hit])
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("quote is not present", result.reason)

    def test_unknown_evidence_id_is_rejected(self):
        hit = lane_hit()
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id="E-invented",
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            rationale="Claimed value",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), lane_context(), extraction, [hit])
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("evidence ID", result.reason)

    def test_ollama_timeout_degrades_one_metric_instead_of_aborting_report(self):
        hit = RetrievalHit(
            evidence_id="E-complex",
            text="Complex sign applicability discussion requiring table interpretation.",
            source="IRC 67.pdf",
            page=10,
            standard_id="IRC:67",
            edition_year=2022,
            score=0.9,
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), TimeoutLLM())
            result = rag.extract_metric(
                "min_sign_height", RoadContext(setting="rural", sign_mounting="shoulder")
            )
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("test timeout", result.reason)

    def test_missing_applicability_context_skips_llm(self):
        llm = FakeLLM(LLMRuleExtraction(status="not_found", rationale="should not be invoked"))
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([lane_hit()]), llm)
            result = rag.extract_metric(
                "min_radius_curvature", RoadContext(road_class="National Highway")
            )
        self.assertEqual(result.status, RuleStatus.NEEDS_CONTEXT)
        self.assertEqual(set(result.missing_context), {"radius_speed_kmph", "terrain"})
        self.assertEqual(llm.calls, 0)

    def test_unverified_road_class_is_not_enough_for_lane_rule(self):
        llm = FakeLLM(LLMRuleExtraction(status="not_found", rationale="not called"))
        context = RoadContext(
            road_class="National Highway",
            road_class_source="OSM functional-tag inference",
            road_class_confidence=0.60,
            setting="rural",
            lanes_total=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([lane_hit()]), llm)
            result = rag.extract_metric("min_lane_width", context)
        self.assertEqual(result.status, RuleStatus.NEEDS_CONTEXT)
        self.assertIn("verified_road_class", result.missing_context)
        self.assertEqual(llm.calls, 0)

    def test_two_laning_manual_is_rejected_for_verified_divided_four_lane_road(self):
        hit = lane_hit().model_copy(update={"standard_id": "IRC:SP:73"})
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            rationale="Incorrect manual for this configuration.",
        )
        context = RoadContext(
            road_class="National Highway",
            road_class_source="test",
            road_class_confidence=0.95,
            setting="rural",
            carriageway="divided",
            lanes_per_carriageway=2,
            carriageway_count=2,
            total_road_lanes=4,
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), context, extraction, [hit])
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("not applicable", result.reason)

    def test_older_selected_edition_is_not_audit_ready(self):
        old = lane_hit(1980)
        new = lane_hit(2022)
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=old.evidence_id,
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            rationale="Old edition selected.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([old, new]), FakeLLM(extraction))
            result = rag.extract_metric("min_lane_width", lane_context())
        self.assertEqual(result.status, RuleStatus.AMBIGUOUS)
        self.assertFalse(result.audit_ready)
        self.assertEqual(result.value_m, 3.5)
        self.assertIsNotNone(result.citation)

    def test_page_less_legacy_evidence_is_not_audit_ready(self):
        hit = lane_hit().model_copy(update={"page": None})
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The minimum traffic lane width shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator=">=",
            rationale="Legacy evidence",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag.extract_metric("min_lane_width", lane_context())
        self.assertEqual(result.status, RuleStatus.AMBIGUOUS)
        self.assertFalse(result.audit_ready)

    def test_special_facility_lane_is_rejected_for_mainline_query(self):
        hit = lane_hit().model_copy(
            update={
                "text": "The width of each ETC toll lane shall be 3.5 m.",
                "source": "Expressway Manual.pdf",
                "standard_id": None,
            }
        )
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote="The width of each ETC toll lane shall be 3.5 m.",
            raw_value=3.5,
            raw_unit="m",
            comparator="=",
            applies_to="ETC toll lane",
            conditions=["electronic toll collection lane"],
            rationale="Same numeric dimension but different facility.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag.extract_metric("min_lane_width", lane_context())
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("different facility", result.reason)

    def test_access_road_lane_exception_is_rejected_for_nh_mainline(self):
        hit = lane_hit().model_copy(
            update={
                "text": "For access roads to residential areas, a lower lane width of 3 m is permissible.",
                "source": "IRC 86.pdf",
                "standard_id": "IRC:86",
            }
        )
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote=hit.text,
            raw_value=3,
            raw_unit="m",
            comparator=">=",
            applies_to="access roads to residential areas",
            rationale="Exception row.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag._validate(get_metric("min_lane_width"), lane_context(), extraction, [hit])
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("not applicable", result.reason)

    def test_irc86_is_filtered_for_rural_lane_retrieval(self):
        status = standard_configuration_status(
            get_metric("min_lane_width"),
            RoadContext(setting="rural", carriageway="divided", total_road_lanes=5),
            "IRC:86",
        )
        self.assertEqual(status, "inapplicable")

    def test_thrie_beam_dimension_is_rejected_for_w_beam_metric(self):
        hit = RetrievalHit(
            evidence_id="E-thrie",
            text=(
                "A Blocked-out Thrie-Beam type steel barrier is shown in Fig. 12. "
                "This is costlier than the W beam type. The post is 850 mm above the ground."
            ),
            source="IRC 119.pdf",
            page=19,
            section="4.5 Barrier types",
            standard_id="IRC:119",
            edition_year=2015,
            chunk_index=1,
            content_hash="thrie",
            score=0.9,
        )
        extraction = LLMRuleExtraction(
            status="found",
            evidence_id=hit.evidence_id,
            verbatim_quote=hit.text,
            raw_value=850,
            raw_unit="mm",
            comparator="=",
            applies_to="Blocked-out Thrie-Beam type steel barrier",
            rationale="Nearby text also mentions W beam.",
        )
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), FakeLLM(extraction))
            result = rag.extract_metric("min_w_beam_barrier_height", RoadContext())
        self.assertEqual(result.status, RuleStatus.INVALID_EVIDENCE)
        self.assertIn("Thrie-beam", result.reason)


class DeterministicUtilityTests(unittest.TestCase):
    def test_heuristic_severity_mapping_is_explicit(self):
        severity = RoadSafetyAuditPipeline._heuristic_severity
        self.assertEqual(severity(0, 0), "SAFE")
        self.assertEqual(severity(0, 2), "REVIEW REQUIRED")
        self.assertEqual(severity(1, 0), "LOW SEVERITY")
        self.assertEqual(severity(2, 1), "MEDIUM SEVERITY")
        self.assertEqual(severity(3, 0), "HIGH SEVERITY (DANGEROUS)")

    def test_gps_resolver_derives_map_speed_setting_and_plain_terrain(self):
        class OfflineResolver(RoadTypeResolver):
            def _features_near(self, points):
                return [
                    {
                        "highway": "trunk",
                        "ref": "NH 65",
                        "lanes": "2",
                        "maxspeed": "80",
                        "oneway": "no",
                    },
                    {"landuse": "residential"},
                ]

            def _elevations(self, points):
                return [100.0 if index % 2 == 0 else 103.0 for index in range(len(points))]

        context = OfflineResolver().resolve([(17.66, 78.10), (17.67, 78.11)])
        self.assertEqual(context.road_class, "National Highway")
        self.assertGreaterEqual(context.road_class_confidence, 0.65)
        self.assertEqual(context.setting, "urban")
        self.assertEqual(context.terrain, "plain")
        self.assertEqual(context.carriageway, "undivided")
        self.assertEqual(context.osm_way_lanes, 2)
        self.assertEqual(context.osm_total_lanes, 2)
        self.assertEqual(context.total_road_lanes, 2)
        self.assertIsNone(context.design_speed_kmph)
        self.assertEqual(context.posted_speed_kmph, 80)
        self.assertEqual(context.radius_speed_kmph, 80)
        self.assertEqual(context.radius_speed_basis, "osm_posted_speed_proxy")
        self.assertEqual(context.posted_speed_source, "OpenStreetMap maxspeed tag")

    def test_verified_design_speed_takes_precedence_over_posted_speed(self):
        context = RoadContext(
            design_speed_kmph=100,
            design_speed_source="approved DPR",
            posted_speed_kmph=80,
            posted_speed_source="OpenStreetMap maxspeed tag",
        )
        self.assertEqual(context.radius_speed_kmph, 100)
        self.assertEqual(context.radius_speed_basis, "verified_design_speed")

    def test_osm_speed_parser_converts_mph(self):
        self.assertEqual(RoadTypeResolver._parse_speed("50 mph"), 80.5)

    def test_unpaired_oneway_lanes_are_not_treated_as_road_total(self):
        class OfflineResolver(RoadTypeResolver):
            def _features_near(self, points):
                return [
                    {
                        "_osm_id": "100",
                        "_geometry": [
                            {"lat": 17.66, "lon": 78.10},
                            {"lat": 17.67, "lon": 78.10},
                        ],
                        "highway": "trunk",
                        "ref": "NH 161",
                        "lanes": "2",
                        "oneway": "yes",
                    }
                ]

            def _elevations(self, points):
                return [100.0 for _ in points]

        context = OfflineResolver().resolve([(17.66, 78.10), (17.67, 78.10)])
        self.assertEqual(context.carriageway, "one_way")
        self.assertEqual(context.osm_way_lanes, 2)
        self.assertEqual(context.lanes_per_carriageway, 2)
        self.assertIsNone(context.osm_total_lanes)
        self.assertIsNone(context.total_road_lanes)
        self.assertIsNone(context.lanes_total)

    def test_opposite_osm_ways_produce_verified_four_lane_divided_context(self):
        class OfflineResolver(RoadTypeResolver):
            def _features_near(self, points):
                return [
                    {
                        "_osm_id": "100",
                        "_geometry": [
                            {"lat": 17.66, "lon": 78.1000},
                            {"lat": 17.67, "lon": 78.1000},
                        ],
                        "highway": "trunk",
                        "ref": "NH 161",
                        "lanes": "2",
                        "oneway": "yes",
                    },
                    {
                        "_osm_id": "200",
                        "_geometry": [
                            {"lat": 17.67, "lon": 78.1002},
                            {"lat": 17.66, "lon": 78.1002},
                        ],
                        "highway": "trunk",
                        "ref": "NH 161",
                        "lanes": "2",
                        "oneway": "yes",
                    },
                ]

            def _elevations(self, points):
                return [100.0 for _ in points]

        context = OfflineResolver().resolve([(17.66, 78.1000), (17.67, 78.1000)])
        self.assertEqual(context.carriageway, "divided")
        self.assertEqual(context.lanes_per_carriageway, 2)
        self.assertEqual(context.opposite_carriageway_lanes, 2)
        self.assertEqual(context.osm_total_lanes, 4)
        self.assertEqual(context.carriageway_count, 2)
        self.assertEqual(context.total_road_lanes, 4)
        self.assertEqual(context.lanes_total, 4)

    def test_asymmetric_paired_carriageways_do_not_claim_uniform_lane_count(self):
        resolver = RoadTypeResolver()
        observations = [
            (
                "National Highway",
                0.92,
                {
                    "_osm_id": "100",
                    "_geometry": [
                        {"lat": 17.66, "lon": 78.1000},
                        {"lat": 17.67, "lon": 78.1000},
                    ],
                    "highway": "trunk",
                    "ref": "NH161",
                    "lanes": "2",
                    "oneway": "yes",
                },
            ),
            (
                "National Highway",
                0.92,
                {
                    "_osm_id": "200",
                    "_geometry": [
                        {"lat": 17.67, "lon": 78.1002},
                        {"lat": 17.66, "lon": 78.1002},
                    ],
                    "highway": "trunk",
                    "ref": "NH161",
                    "lanes": "3",
                    "oneway": "yes",
                },
            ),
        ]
        values = resolver._infer_lane_context(observations, observations[0][2])
        self.assertEqual(values[0], "divided")
        self.assertEqual(values[1], 2)
        self.assertIsNone(values[2])
        self.assertEqual(values[3], 3)
        self.assertEqual(values[4], 5)
        self.assertEqual(values[5], 2)
        self.assertIsNone(values[6])

    def test_verified_base_override_preserves_raw_asymmetric_osm_total(self):
        args = build_parser().parse_args(
            [
                "query",
                "--carriageway",
                "divided",
                "--lanes",
                "4",
                "--lanes-per-carriageway",
                "2",
                "--carriageway-count",
                "2",
            ]
        )
        supplied = context_from_args(args)
        resolved = RoadContext(
            carriageway="divided",
            osm_way_lanes=2,
            opposite_carriageway_lanes=3,
            osm_total_lanes=5,
        )
        merged = resolved.model_copy(
            update={
                key: value
                for key, value in supplied.model_dump().items()
                if value not in (None, "", "unknown", [], {})
            }
        )
        self.assertEqual(merged.osm_total_lanes, 5)
        self.assertEqual(merged.total_road_lanes, 4)
        self.assertEqual(merged.lanes_per_carriageway, 2)
        self.assertEqual(merged.lane_count_source, "user/CLI verified base road configuration")
        self.assertEqual(
            standard_configuration_status(get_metric("min_lane_width"), merged, "IRC:SP:84"),
            "applicable",
        )

    def test_route_scoped_override_applies_only_inside_matching_nh_corridor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config"
            config.mkdir()
            (config / "road_context_overrides.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "overrides": [
                            {
                                "id": "test_nh161",
                                "highway_ref": "NH161",
                                "bounds": {
                                    "min_lat": 17.65,
                                    "max_lat": 17.67,
                                    "min_lon": 78.09,
                                    "max_lon": 78.12,
                                },
                                "carriageway": "divided",
                                "lanes_per_carriageway": 2,
                                "carriageway_count": 2,
                                "total_road_lanes": 4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            raw = RoadContext(
                highway_ref="NH 161",
                carriageway="divided",
                osm_way_lanes=2,
                opposite_carriageway_lanes=3,
                osm_total_lanes=5,
            )
            applied = RoadSafetyAuditPipeline._apply_reviewed_route_override(
                raw, [(17.66, 78.10)], root
            )
            outside = RoadSafetyAuditPipeline._apply_reviewed_route_override(
                raw, [(18.0, 78.10)], root
            )
        self.assertEqual(applied.osm_total_lanes, 5)
        self.assertEqual(applied.total_road_lanes, 4)
        self.assertIn("reviewed route override", applied.lane_count_source)
        self.assertIsNone(outside.total_road_lanes)

    def test_project_manual_applicability_uses_verified_road_total(self):
        metric = get_metric("min_lane_width")
        context = RoadContext(
            road_class="National Highway",
            carriageway="divided",
            lanes_per_carriageway=2,
            carriageway_count=2,
            osm_total_lanes=5,
            total_road_lanes=4,
        )
        self.assertEqual(
            standard_configuration_status(metric, context, "IRC:SP:73"),
            "inapplicable",
        )
        self.assertEqual(
            standard_configuration_status(metric, context, "IRC:SP:84"),
            "applicable",
        )
        self.assertEqual(
            standard_configuration_status(metric, context, "IRC:SP:87"),
            "inapplicable",
        )

    def test_workbook_context_infers_lanes_carriageway_and_roadside_signs(self):
        frame = pd.DataFrame(
            {
                "Expected Total Lanes": [2, 2, 2],
                "Centre Lanes Detected": [1, 1, 1],
                "Shoulder Lanes Detected": [2, 2, 2],
                "Traffic Sign Class": ["Chevron", "Warning Sign", "Mandatory Sign"],
                "Raw OCR Text": ["N: 17.66 E: 78.10"] * 3,
            }
        )
        fake_rag = SimpleNamespace(settings=SimpleNamespace(project_dir=Path.cwd()))
        pipeline = RoadSafetyAuditPipeline("input.xlsx", fake_rag)
        context = pipeline._resolve_context(frame)
        self.assertEqual(context.lanes_total, 2)
        self.assertEqual(context.carriageway, "undivided")
        self.assertEqual(context.sign_mounting, "shoulder")

    def test_cli_without_subcommand_selects_guided_mode(self):
        self.assertIsNone(build_parser().parse_args([]).command)

    def test_deep_retrieval_settings_expand_candidate_pools(self):
        with tempfile.TemporaryDirectory() as temp:
            base = settings(temp)
            deep = deep_retrieval_settings(base, True)
        self.assertTrue(deep.exhaustive_retrieval)
        self.assertTrue(deep.enable_reranker)
        self.assertGreaterEqual(deep.dense_k, 80)
        self.assertGreaterEqual(deep.lexical_k, 100)
        self.assertGreaterEqual(deep.final_k, 12)
        self.assertGreaterEqual(deep.neighbor_window, 2)

    def test_corpus_folder_override_keeps_existing_database(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "irc"
            folder.mkdir()
            base = settings(temp)
            selected = corpus_folder_settings(base, folder)
        self.assertEqual(selected.corpus_dirs, (folder.resolve(),))
        self.assertEqual(selected.persist_directory, base.persist_directory)

    def test_exhaustive_scan_finds_explicit_w_beam_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.settings = deep_retrieval_settings(settings(temp), True)
            retriever.ids = ["general", "wbeam"]
            retriever.documents = [
                "General road safety barrier discussion without dimensions.",
                "The W-beam barrier mounting height above road level shall be 700 mm.",
            ]
            retriever.metadatas = [
                {"standard_id": "IRC:119"},
                {"standard_id": "IRC:119"},
            ]
            ranks = retriever._exhaustive_scan(
                get_metric("min_w_beam_barrier_height"),
                RoadContext(barrier_type="w_beam"),
            )
        self.assertIn("wbeam", ranks)
        self.assertNotIn("general", ranks)

    def test_barrier_queries_receive_feature_specific_context(self):
        retriever = FakeRetriever([])
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(
                settings(temp), retriever, FakeLLM(LLMRuleExtraction(status="not_found"))
            )
            rag.extract_all(
                RoadContext(),
                metric_keys=["min_w_beam_barrier_height", "min_concrete_barrier_height"],
                use_cache=False,
            )
        contexts = {key: context for key, context in retriever.contexts}
        self.assertEqual(contexts["min_w_beam_barrier_height"].barrier_type, "w_beam")
        self.assertEqual(contexts["min_concrete_barrier_height"].barrier_type, "concrete")

    def test_radius_table_recovery_uses_desirable_not_absolute_minimum(self):
        hit = RetrievalHit(
            evidence_id="E-radius",
            text=(
                "2.9.4 Radii of Horizontal Curves The desirable minimum and absolute minimum "
                "radii of horizontal curves are given in Table 2.5. Table 2.5 Minimum Radii "
                "of Horizontal Curves Nature of Terrain Desirable Minimum Radius Absolute "
                "Minimum Radius Plain and Rolling 400 m 250 m Mountainous and Steep 150 m 75 m."
            ),
            source="IRC 73.pdf",
            page=26,
            section="2.9.4 Radii of Horizontal Curves",
            standard_id="IRC:SP:73",
            edition_year=2018,
            chunk_index=1,
            content_hash="radius",
            score=0.9,
        )
        extraction = StandardsRAG._explicit_clause_extraction(
            get_metric("min_radius_curvature"),
            RoadContext(terrain="plain", design_speed_kmph=80),
            [hit],
        )
        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.raw_value, 400)
        self.assertIn("Plain and Rolling 400 m 250 m", extraction.verbatim_quote)

    def test_direct_irc_radius_table_is_reported_as_non_proxy_basis(self):
        hit = RetrievalHit(
            evidence_id="E-radius-direct-basis",
            text=(
                "2.9.4 Radii of Horizontal Curves. Table 2.5 Minimum Radii of "
                "Horizontal Curves Nature of Terrain Desirable Minimum Radius Absolute "
                "Minimum Radius Plain and Rolling 400 m 250 m Mountainous and Steep "
                "150 m 75 m."
            ),
            source="IRC 73.pdf",
            page=26,
            section="2.9.4 Radii of Horizontal Curves",
            standard_id="IRC:SP:73",
            edition_year=2018,
            score=0.9,
        )
        context = RoadContext(
            road_class="National Highway",
            road_class_source="test",
            road_class_confidence=0.95,
            setting="rural",
            terrain="plain",
            carriageway="undivided",
            total_road_lanes=2,
        )
        llm = FakeLLM(LLMRuleExtraction(status="not_found", rationale="not called"))
        with tempfile.TemporaryDirectory() as temp:
            result = StandardsRAG(settings(temp), FakeRetriever([hit]), llm).extract_metric(
                "min_radius_curvature", context
            )
        self.assertEqual(result.value_m, 400)
        self.assertTrue(result.provisional)
        self.assertIn("source/edition is indexed but not reviewer-verified", result.applicability_basis)
        self.assertIn("no speed proxy used", result.applicability_basis)
        self.assertEqual(llm.calls, 0)

    def test_osm_posted_speed_radius_result_is_explicitly_provisional(self):
        hit = RetrievalHit(
            evidence_id="E-radius-osm-proxy",
            text=(
                "For a design speed of 80 km/h in plain terrain, the minimum horizontal "
                "curve radius shall be 230 m."
            ),
            source="IRC 73.pdf",
            page=35,
            section="Horizontal alignment",
            standard_id="IRC:73",
            edition_year=1980,
            score=0.9,
        )
        context = RoadContext(
            road_class="National Highway",
            road_class_source="test",
            road_class_confidence=0.95,
            terrain="plain",
            posted_speed_kmph=80,
            posted_speed_source="OpenStreetMap maxspeed tag",
        )
        with tempfile.TemporaryDirectory() as temp:
            result = StandardsRAG(
                settings(temp),
                FakeRetriever([hit]),
                FakeLLM(LLMRuleExtraction(status="not_found", rationale="not called")),
            ).extract_metric("min_radius_curvature", context)
        self.assertTrue(result.provisional)
        self.assertIn("OSM posted-speed proxy", result.applicability_basis)
        self.assertIn("PROVISIONAL", result.reason)
        self.assertLessEqual(result.evidence_quality_score, 0.78)

    def test_configuration_specific_radius_table_abstains_without_lane_context(self):
        hit = RetrievalHit(
            evidence_id="E-radius-no-speed",
            text=(
                "2.9.4 Radii of Horizontal Curves. Table 2.5 Minimum Radii of "
                "Horizontal Curves Nature of Terrain Desirable Minimum Radius Absolute "
                "Minimum Radius Plain and Rolling 400 m 250 m Mountainous and Steep "
                "150 m 75 m."
            ),
            source="IRC 73.pdf",
            page=26,
            section="2.9.4 Radii of Horizontal Curves",
            standard_id="IRC:SP:73",
            edition_year=2018,
            score=0.9,
        )
        llm = FakeLLM(LLMRuleExtraction(status="not_found", rationale="not called"))
        with tempfile.TemporaryDirectory() as temp:
            rag = StandardsRAG(settings(temp), FakeRetriever([hit]), llm)
            result = rag.extract_metric("min_radius_curvature", RoadContext(terrain="plain"))
        self.assertEqual(result.status, RuleStatus.NEEDS_CONTEXT)
        self.assertIn("verified_carriageway", result.missing_context)
        self.assertIn("verified_total_road_lanes", result.missing_context)
        self.assertEqual(llm.calls, 0)

    def test_w_beam_tolerance_is_parsed_as_a_range(self):
        hit = RetrievalHit(
            evidence_id="SE-WBEAM",
            text=(
                "Fig. 11 Typical Details of W Structural Elements. W-beam rail top "
                "height above the adjacent ground line: 730 +/- 25 mm."
            ),
            source="IRC 119.pdf",
            page=20,
            section="Fig. 11",
            standard_id="IRC:119",
            edition_year=2015,
            score=1.0,
        )
        extraction = StandardsRAG._explicit_clause_extraction(
            get_metric("min_w_beam_barrier_height"), RoadContext(), [hit]
        )
        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.comparator, "range")
        self.assertEqual(extraction.raw_value, 705)
        self.assertEqual(extraction.second_raw_value, 755)
        self.assertTrue(quote_supports_tolerance_range(hit.text, 705, 755, "mm"))

    def test_ocr_merge_keeps_novel_figure_dimension_on_text_rich_page(self):
        base = PageText(
            page=20,
            text="A long surrounding paragraph about traffic safety barriers. " * 4,
            parser="pypdf",
        )
        ocr = PageText(
            page=20,
            text="Fig. 11 Typical details\nW-beam 730 +/- 25 mm",
            parser="docling_ocr",
        )
        merged = PageExtractor._merge_page_text(base, ocr)
        self.assertIn("730 +/- 25 mm", merged.text)
        self.assertEqual(merged.parser, "pypdf+docling_ocr")

    def test_unobserved_concrete_barrier_is_not_requested(self):
        frame = pd.DataFrame(
            {
                "WBeam_crash barrier_height": [0.72],
                "Concrete_crash_barrier_height": [None],
            }
        )
        keys = RoadSafetyAuditPipeline._observed_metric_keys(frame)
        self.assertIn("min_w_beam_barrier_height", keys)
        self.assertNotIn("min_concrete_barrier_height", keys)

    def test_units(self):
        self.assertAlmostEqual(convert_to_metres(1200, "mm"), 1.2)
        self.assertAlmostEqual(convert_to_metres(75, "cm"), 0.75)
        self.assertTrue(quote_contains_measurement("height is 1,200 mm", 1200, "mm"))
        self.assertFalse(quote_contains_measurement("height is 1200 mm", 1.2, "m"))

    def test_coordinate_parser_handles_ocr_space(self):
        self.assertEqual(parse_coordinate_text("N: 17.6610 E:78. 1068"), (17.661, 78.1068))
        self.assertIsNone(parse_coordinate_text("N: 117.2 E: 78.1"))

    def test_measurement_evaluation_does_not_call_llm(self):
        rule = ThresholdResult(
            metric_key="min_lane_width",
            metric_name="minimum lane width",
            status=RuleStatus.FOUND,
            value_m=3.5,
            comparator=">=",
            citation={
                "evidence_id": "E1",
                "source": "IRC.pdf",
                "page": 1,
                "quote": "minimum 3.5 m",
            },
            evidence_quality_score=0.9,
            reason="verified",
        )
        result = evaluate_measurements([3.6, 3.4], rule)
        self.assertEqual(result.status, "FAIL")
        self.assertTrue(result.failed)

    def test_unknown_rule_never_becomes_pass(self):
        rule = ThresholdResult(
            metric_key="min_lane_width",
            metric_name="minimum lane width",
            status=RuleStatus.NEEDS_CONTEXT,
            reason="setting required",
        )
        result = evaluate_measurements([4.0], rule)
        self.assertEqual(result.status, "REVIEW_REQUIRED")
        self.assertFalse(result.failed)

    def test_number_parser_handles_multiple_measurements(self):
        self.assertEqual(extract_numbers("0.562, 0.596"), [0.562, 0.596])
        self.assertEqual(extract_numbers("3.20 | 4.34"), [3.2, 4.34])

    def test_standard_lane_dimension_is_used_as_disclosed_lower_bound_screening(self):
        hit = RetrievalHit(
            evidence_id="E-lane-standard",
            text="The standard lane width of project highway shall be 3.5 m.",
            source="IRC 84.pdf",
            page=19,
            standard_id="IRC:SP:84",
            edition_year=2019,
            score=0.9,
        )
        extraction = StandardsRAG._explicit_clause_extraction(
            get_metric("min_lane_width"),
            RoadContext(
                road_class="National Highway",
                road_class_confidence=0.95,
                setting="rural",
                carriageway="divided",
                total_road_lanes=4,
            ),
            [hit],
        )
        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.comparator, ">=")
        self.assertIn("lower-bound check", extraction.conditions[0])

    def test_generic_barrier_measurement_routes_to_configured_w_beam_rule(self):
        frame = pd.DataFrame({"Crash barrier_height": ["0.562, 0.596"]})
        keys = RoadSafetyAuditPipeline._observed_metric_keys(
            frame, RoadContext(barrier_type="w_beam")
        )
        self.assertIn("min_w_beam_barrier_height", keys)
        self.assertNotIn("min_concrete_barrier_height", keys)

    def test_project_route_configuration_is_explicitly_provisional(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config"
            config.mkdir()
            (config / "road_context_overrides.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "overrides": [
                            {
                                "id": "configured-route",
                                "highway_ref": "NH161",
                                "bounds": {
                                    "min_lat": 17.65,
                                    "max_lat": 17.67,
                                    "min_lon": 78.09,
                                    "max_lon": 78.12,
                                },
                                "carriageway": "divided",
                                "lanes_per_carriageway": 2,
                                "carriageway_count": 2,
                                "total_road_lanes": 4,
                                "barrier_type": "w_beam",
                                "review_status": "project_configured",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = RoadSafetyAuditPipeline._apply_reviewed_route_override(
                RoadContext(highway_ref="NH161"), [(17.66, 78.10)], root
            )
        self.assertEqual(result.total_road_lanes, 4)
        self.assertEqual(result.barrier_type, "w_beam")
        self.assertTrue(result.lane_configuration_provisional)
        self.assertIn("project route configuration", result.lane_count_source)

    def test_custom_report_evaluates_every_measurement_and_any_failure_wins(self):
        rule = ThresholdResult(
            metric_key="min_w_beam_barrier_height",
            metric_name="W-beam barrier mounting height",
            status=RuleStatus.FOUND,
            value_m=0.705,
            second_value_m=0.755,
            comparator="range",
            citation={
                "evidence_id": "E-test-wbeam",
                "standard_id": "IRC:119",
                "source": "IRC 119.pdf",
                "page": 20,
                "quote": "W-beam rail top height 730 +/- 25 mm.",
            },
            evidence_quality_score=0.9,
            reason="test rule",
        )
        detail, state = RoadSafetyAuditPipeline._format_measurements([0.73, 0.596], rule, "W-Beam")
        self.assertEqual(state, "FAIL")
        self.assertIn("W-Beam 1: 0.73m", detail)
        self.assertIn("W-Beam 2: 0.596m", detail)
        self.assertIn("- PASS", detail)
        self.assertIn("- FAIL", detail)

    def test_custom_report_all_measurements_pass_only_when_each_passes(self):
        rule = ThresholdResult(
            metric_key="min_radius_curvature",
            metric_name="minimum horizontal-curve radius",
            status=RuleStatus.FOUND,
            value_m=400,
            comparator=">=",
            citation={
                "evidence_id": "E-test-radius",
                "standard_id": "IRC:SP:84",
                "source": "IRC 84.pdf",
                "page": 22,
                "quote": "Plain and Rolling 400 m 250 m.",
            },
            evidence_quality_score=0.9,
            reason="test rule",
        )
        detail, state = RoadSafetyAuditPipeline._format_measurements([500, 450], rule, "Radius")
        self.assertEqual(state, "PASS")
        self.assertEqual(detail.count("- PASS"), 2)

    def test_recommendation_calculates_lane_width_shortfall(self):
        rule = ThresholdResult(
            metric_key="min_lane_width",
            metric_name="minimum lane width",
            status=RuleStatus.FOUND,
            value_m=3.5,
            comparator=">=",
            citation={
                "evidence_id": "E-test-lane",
                "source": "IRC 84.pdf",
                "page": 19,
                "quote": "The standard lane width shall be 3.5 m.",
            },
            evidence_quality_score=0.9,
            reason="test rule",
        )
        detail = "Lane 1: 3.29m (FAIL) | Lane 2: 4.09m (PASS)"
        result = RoadSafetyAuditPipeline._corrective_action(
            "Lane width",
            detail,
            "min_lane_width",
            SimpleNamespace(results={"min_lane_width": rule}),
        )
        self.assertIn("Widen Lane 1 by at least 0.21 m", result)
        self.assertNotIn("Lane 2", result)

    def test_recommendation_calculates_sign_height_shortfall(self):
        rule = ThresholdResult(
            metric_key="min_sign_height",
            metric_name="minimum sign height",
            status=RuleStatus.FOUND,
            value_m=2.5,
            comparator=">=",
            citation={
                "evidence_id": "E-test-sign",
                "source": "IRC 67.pdf",
                "page": 17,
                "quote": "The lower edge should be 2.5 m above the carriageway.",
            },
            evidence_quality_score=0.9,
            reason="test rule",
        )
        detail, _ = RoadSafetyAuditPipeline._format_measurements(
            [1.31], rule, "Sign height"
        )
        result = RoadSafetyAuditPipeline._corrective_action(
            "Traffic-sign mounting height",
            detail,
            "min_sign_height",
            SimpleNamespace(results={"min_sign_height": rule}),
        )
        self.assertIn("Raise the traffic sign by at least 1.19 m", result)
        self.assertIn("to at least 2.5 m", result)

    def test_recommendation_routes_radius_failure_to_engineering_review(self):
        rule = ThresholdResult(
            metric_key="min_radius_curvature",
            metric_name="minimum horizontal-curve radius",
            status=RuleStatus.FOUND,
            value_m=400,
            comparator=">=",
            citation={
                "evidence_id": "E-test-radius-action",
                "source": "IRC 84.pdf",
                "page": 22,
                "quote": "Desirable minimum radius 400 m.",
            },
            evidence_quality_score=0.9,
            reason="test rule",
        )
        detail, _ = RoadSafetyAuditPipeline._format_measurements(
            [350], rule, "Radius"
        )
        result = RoadSafetyAuditPipeline._corrective_action(
            "Radius of curvature",
            detail,
            "min_radius_curvature",
            SimpleNamespace(results={"min_radius_curvature": rule}),
        )
        self.assertIn("50 m below the 400 m requirement", result)
        self.assertIn("geometric-design review", result)

    def test_custom_report_retains_all_values_when_rule_needs_review(self):
        rule = ThresholdResult(
            metric_key="min_kerb_height",
            metric_name="kerb height",
            status=RuleStatus.NEEDS_CONTEXT,
            reason="kerb type required",
        )
        detail, state = RoadSafetyAuditPipeline._format_measurements([0.15, 0.18], rule, "Kerb")
        self.assertEqual(state, "REVIEW_REQUIRED")
        self.assertIn("Kerb 1: 0.15m", detail)
        self.assertIn("Kerb 2: 0.18m", detail)
        self.assertEqual(detail.count("REVIEW REQUIRED"), 2)

    def test_custom_report_marks_empty_measurement_list_absent(self):
        rule = ThresholdResult(
            metric_key="min_sign_height",
            metric_name="minimum sign height",
            status=RuleStatus.NEEDS_CONTEXT,
            reason="context required",
        )
        self.assertEqual(
            RoadSafetyAuditPipeline._format_measurements([], rule, "Sign height"),
            ("Absent", "ABSENT"),
        )

    def test_chunker_is_deterministic_and_overlaps(self):
        text = "\n\n".join(f"Section {index}. " + ("word " * 90) for index in range(6))
        chunker = PageChunker(chunk_size=600, overlap=80)
        first = chunker.split(text)
        second = chunker.split(text)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 2)
        self.assertTrue(all(len(chunk) >= 40 for chunk in first))

    def test_standard_identifier(self):
        self.assertEqual(
            identify_standard("irc.gov.in.sp.119.2018.pdf", "IRC:SP:119-2018"),
            ("IRC:SP:119", 2018),
        )
        self.assertEqual(identify_standard("IRC 73.pdf", "IRC:73-1980"), ("IRC:73", 1980))
        self.assertEqual(
            identify_standard("IRC 73.pdf", "IRC:SP:73-2018 Project Highways"),
            ("IRC:SP:73", 2018),
        )

    def test_bm25_prefers_exact_engineering_terms(self):
        documents = [
            tokenize("committee members and publication details"),
            tokenize("minimum lane width for two lane carriageway is 3.5 m"),
            tokenize("road safety barrier warrants"),
        ]
        results = BM25Index(documents).search("minimum lane width", 2)
        self.assertEqual(results[0][0], 1)

    def test_trajectory_sampling_is_bounded(self):
        points = [(float(index), float(index)) for index in range(50)]
        sampled = sample_trajectory(points, maximum=7)
        self.assertEqual(len(sampled), 7)
        self.assertEqual(sampled[0], points[0])
        self.assertEqual(sampled[-1], points[-1])


if __name__ == "__main__":
    unittest.main()
