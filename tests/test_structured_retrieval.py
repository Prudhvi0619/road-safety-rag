from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from road_safety_rag.catalog import get_metric
from road_safety_rag.config import Settings
from road_safety_rag.evaluation import GoldCase, _retrieval_matches, evaluate_gold
from road_safety_rag.models import RetrievalHit, RoadContext, RuleStatus
from road_safety_rag.retrieval import HybridRetriever
from road_safety_rag.service import StandardsRAG

METRIC = "min_w_beam_barrier_height"
SOURCE_HASH = "a" * 64
RECORD = {
    "evidence_id": "SE-WBEAM",
    "metric_key": METRIC,
    "standard_id": "IRC:119",
    "edition_year": 2015,
    "source": "IRC 119.pdf",
    "source_sha256": SOURCE_HASH,
    "page": 20,
    "section": "Fig. 11",
    "text": "W-beam rail top height above the adjacent ground line: 730 +/- 25 mm.",
    "verification": "visual_source_check",
}


class RecordingReranker:
    def __init__(self, structured_score):
        self.structured_score = structured_score
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [self.structured_score if text == RECORD["text"] else 0.0 for _, text in pairs]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = Settings(
        project_dir=tmp_path,
        corpus_dirs=(),
        persist_directory=tmp_path / "configured-db",
        require_verified_standards=False,
        dense_k=50,
        lexical_k=50,
    )
    config = tmp_path / "config"
    config.mkdir()
    (config / "structured_evidence.json").write_text(
        json.dumps({"records": [RECORD]}), encoding="utf-8"
    )
    settings.persist_directory.mkdir()
    manifest = settings.persist_directory / "index_manifest.json"
    manifest.write_text(
        json.dumps({"documents": {"IRC 119.pdf": {"sha256": SOURCE_HASH}}}),
        encoding="utf-8",
    )
    texts = [
        "The W-beam barrier mounting height shall be 700 mm.",
        "Road safety introduction and contents.",
    ]

    def vector_store(_self):
        metadata = [
            {"source": f"source-{i}.pdf", "page": 1, "content_hash": f"chunk-{i}"}
            for i in range(len(texts))
        ]
        return SimpleNamespace(
            _collection=SimpleNamespace(
                get=lambda **kwargs: {
                    "ids": [f"document-{i:03}" for i in range(len(texts))],
                    "documents": list(texts),
                    "metadatas": metadata,
                }
            ),
            similarity_search=lambda query, k: [
                SimpleNamespace(page_content=text, metadata=meta)
                for text, meta in list(zip(texts, metadata))[:k]
            ],
        )

    monkeypatch.setattr(HybridRetriever, "_vector_store", vector_store)
    return settings, manifest, texts


def retrieve(settings):
    return HybridRetriever(settings).retrieve(get_metric(METRIC), RoadContext())


def test_verified_wbeam_is_a_raw_retrieval_candidate_with_bounded_prior(setup):
    settings, _, _ = setup
    hits = retrieve(settings)
    structured = next(hit for hit in hits if hit.evidence_id == "SE-WBEAM")
    corpus_scores = [hit.score for hit in hits if hit.evidence_id != "SE-WBEAM"]
    assert min(corpus_scores) <= structured.score <= max(corpus_scores)
    assert structured.content_hash == SOURCE_HASH
    assert structured.metadata["source_sha256"] == SOURCE_HASH
    assert structured.score != 1.0
    assert structured.dense_rank is None and structured.lexical_rank is None
    assert hits[0].evidence_id != "SE-WBEAM"


def test_structured_prior_depends_on_query_relevance_not_verification(setup):
    settings, _, _ = setup
    retriever = HybridRetriever(settings)
    corpus = retriever.retrieve(get_metric(METRIC), RoadContext())
    corpus = [hit for hit in corpus if hit.evidence_id != "SE-WBEAM"]
    relevant = retriever._structured_candidates(get_metric(METRIC), corpus, ["rail top height"])
    irrelevant = retriever._structured_candidates(get_metric(METRIC), corpus, ["introduction contents"])
    assert relevant[0].score > irrelevant[0].score


@pytest.mark.parametrize("invalid", ["hash", "empty_hash", "missing", "empty", "bad_entry"])
def test_unvalidated_structured_evidence_is_excluded(setup, invalid):
    settings, manifest, _ = setup
    if invalid == "missing":
        manifest.unlink()
    elif invalid == "hash":
        manifest.write_text(json.dumps({"documents": {"source": {"sha256": "b" * 64}}}))
    elif invalid == "bad_entry":
        manifest.write_text(json.dumps({"documents": {"source": "not a hash record"}}))
    elif invalid == "empty":
        manifest.write_text(json.dumps({"documents": {}}))
    else:
        (settings.project_dir / "config" / "structured_evidence.json").write_text(
            json.dumps({"records": [{**RECORD, "source_sha256": ""}]})
        )
    assert all(hit.evidence_id != "SE-WBEAM" for hit in retrieve(settings))


def test_manifest_is_revalidated_on_each_retrieval(setup):
    settings, manifest, _ = setup
    retriever = HybridRetriever(settings)
    assert any(
        hit.evidence_id == "SE-WBEAM"
        for hit in retriever.retrieve(get_metric(METRIC), RoadContext())
    )
    manifest.write_text(json.dumps({"documents": {}}))
    assert all(
        hit.evidence_id != "SE-WBEAM"
        for hit in retriever.retrieve(get_metric(METRIC), RoadContext())
    )


@pytest.mark.parametrize("semantic", [-8.0, 8.0])
def test_structured_candidate_participates_in_reranking_without_forced_first_place(setup, semantic):
    settings, _, _ = setup
    retriever = HybridRetriever(settings)
    reranker = RecordingReranker(semantic)
    retriever.reranker = reranker
    hits = retriever.retrieve(get_metric(METRIC), RoadContext())
    structured = next(hit for hit in hits if hit.evidence_id == "SE-WBEAM")
    canonical = retriever._queries(get_metric(METRIC), RoadContext())[0]
    assert (canonical, RECORD["text"]) in reranker.pairs
    assert structured.metadata["reranker_queries_evaluated"] == [canonical]
    prior = structured.metadata["structured_normalized_prior"]
    assert structured.score == pytest.approx(0.88 / (1 + math.exp(-semantic)) + 0.12 * prior)
    assert (hits[0].evidence_id == "SE-WBEAM") == (semantic > 0)


def test_structured_candidate_reaches_reranker_beyond_corpus_budget(setup):
    settings, _, texts = setup
    texts[:] = [
        f"The W-beam barrier mounting height shall be 700 mm. Clause {i}." for i in range(40)
    ]
    texts.append("Introduction")
    retriever = HybridRetriever(settings)
    retriever.reranker = RecordingReranker(8.0)
    hits = retriever.retrieve(get_metric(METRIC), RoadContext())
    assert hits[0].evidence_id == "SE-WBEAM"
    assert len(retriever.reranker.pairs) == 31


@pytest.mark.parametrize("duplicate", [RECORD, {**RECORD, "evidence_id": "SE-ALIAS"}])
def test_duplicate_registry_records_are_returned_and_reranked_once(setup, duplicate):
    settings, _, _ = setup
    (settings.project_dir / "config" / "structured_evidence.json").write_text(
        json.dumps({"records": [RECORD, duplicate]})
    )
    retriever = HybridRetriever(settings)
    retriever.reranker = RecordingReranker(8.0)
    hits = retriever.retrieve(get_metric(METRIC), RoadContext())
    assert sum(hit.content_hash == SOURCE_HASH for hit in hits) == 1
    assert sum(text == RECORD["text"] for _, text in retriever.reranker.pairs) == 1


def test_deduplication_checks_id_before_secondary_hash_and_keeps_best():
    def hit(evidence_id, content_hash, score):
        return RetrievalHit(
            evidence_id=evidence_id,
            content_hash=content_hash,
            score=score,
            text=evidence_id,
            source="test",
        )

    hits = [
        hit("E-1", "h1", 0.8),
        hit("E-1", "h2", 0.7),
        hit("E-2", "h1", 0.6),
        hit("E-3", "h3", 0.5),
    ]
    assert [item.evidence_id for item in HybridRetriever._deduplicate(hits)] == ["E-1", "E-3"]


def test_missing_or_unrelated_registry_leaves_dense_bm25_behavior_unchanged(setup):
    settings, _, _ = setup
    path = settings.project_dir / "config" / "structured_evidence.json"
    path.unlink()
    before = [hit.model_dump() for hit in retrieve(settings)]
    path.write_text(json.dumps({"records": [{**RECORD, "metric_key": "min_lane_width"}]}))
    assert [hit.model_dump() for hit in retrieve(settings)] == before
    assert before[0]["dense_rank"] == 1
    assert before[0]["lexical_rank"] == 1


def test_structured_only_candidates_have_a_finite_rrf_scale(setup):
    settings, _, _ = setup
    retriever = HybridRetriever(settings)
    retriever.vector_store.similarity_search = lambda *args, **kwargs: []
    retriever.bm25.search = lambda *args, **kwargs: []
    hits = retriever.retrieve(get_metric(METRIC), RoadContext())
    assert len(hits) == 1 and 0 < hits[0].score <= 1 / 61


def test_service_preserves_hybrid_ranking_and_does_not_reinsert_filtered_evidence(
    setup, monkeypatch
):
    settings, _, _ = setup
    retriever = HybridRetriever(settings)
    rag = StandardsRAG(settings, retriever=retriever)
    hits = retriever.retrieve(get_metric(METRIC), RoadContext())
    captured = []
    original = rag._explicit_clause_extraction

    def capture(metric, context, evidence):
        captured.append(list(evidence))
        return original(metric, context, evidence)

    monkeypatch.setattr(rag, "_explicit_clause_extraction", capture)
    monkeypatch.setattr(
        rag.structured_evidence,
        "hits",
        lambda *args: pytest.fail("Service reinjected structured evidence"),
    )
    result = rag.extract_metric(METRIC, RoadContext(), retrieved_hits=[*hits, hits[-1]])
    assert result.citation.evidence_id == "SE-WBEAM"
    assert captured[0] == hits
    assert (
        rag.extract_metric(METRIC, RoadContext(), retrieved_hits=[]).status == RuleStatus.NOT_FOUND
    )


def test_custom_retriever_fallback_validates_and_deduplicates(setup):
    settings, manifest, _ = setup
    hits = retrieve(settings)
    retriever = SimpleNamespace(retrieve=lambda *args: [*hits, hits[0]])
    result = StandardsRAG(settings, retriever=retriever).extract_metric(METRIC, RoadContext())
    assert result.citation.evidence_id == "SE-WBEAM"
    assert all(item.content_hash != SOURCE_HASH for item in result.alternatives)
    retriever.retrieve = lambda *args: []
    assert (
        StandardsRAG(settings, retriever=retriever)
        .extract_metric(METRIC, RoadContext())
        .citation.evidence_id
        == "SE-WBEAM"
    )
    manifest.write_text(json.dumps({"documents": {}}))
    assert (
        StandardsRAG(settings, retriever=retriever).extract_metric(METRIC, RoadContext()).status
        == RuleStatus.NOT_FOUND
    )


def test_evaluator_matches_wbeam_through_hybrid_retriever(setup, tmp_path):
    settings, _, _ = setup
    case = GoldCase(
        case_id="retrieval_wbeam",
        metric_key=METRIC,
        road_context=RoadContext(),
        expected_standard_id="IRC:119",
        expected_edition_year=2015,
        expected_page=20,
        expected_evidence_id="SE-WBEAM",
        expected_content_hash=SOURCE_HASH,
        expected_quote=RECORD["text"],
    )
    assert any(_retrieval_matches(hit, case) for hit in retrieve(settings))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([case.model_dump(mode="json")]))
    result = evaluate_gold(settings, gold)
    assert result["results"][0]["first_matching_rank"] <= 5
    assert result["summary"]["recall_at_5"] == 1.0
