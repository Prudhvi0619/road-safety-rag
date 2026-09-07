from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from road_safety_rag.models import RetrievalHit, RoadContext
from road_safety_rag.retrieval import HybridRetriever

QUERIES = ["Q1 canonical", "Q2 dense", "Q3 alternate"]


class RecordingReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.calls += 1
        self.pairs = list(pairs)
        if callable(self.scores):
            return self.scores(self.pairs)
        return list(self.scores)


class FakeVectorStore:
    def __init__(self, retriever, rankings):
        self.retriever = retriever
        self.rankings = rankings

    def similarity_search(self, query, k):
        results = []
        for document_id in self.rankings.get(query, [])[:k]:
            index = self.retriever.id_to_index[document_id]
            results.append(
                SimpleNamespace(
                    page_content=self.retriever.documents[index],
                    metadata=self.retriever.metadatas[index],
                )
            )
        return results


class FakeBM25:
    def __init__(self, retriever, rankings):
        self.retriever = retriever
        self.rankings = rankings

    def search(self, query, k):
        return [
            (self.retriever.id_to_index[document_id], 1.0 / rank)
            for rank, document_id in enumerate(self.rankings.get(query, [])[:k], start=1)
        ]


def _order_with_target(rank: int, size: int = 10) -> list[str]:
    order = [f"doc-{index}" for index in range(1, size + 1)]
    order.insert(rank - 1, "target")
    return order[:size]


def _retriever(dense_rankings=None, lexical_rankings=None, reranker=None):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.ids = ["target", *[f"doc-{index}" for index in range(1, 11)]]
    retriever.documents = [
        f"Road-safety requirement for {document_id}: minimum value 3.5 m."
        for document_id in retriever.ids
    ]
    retriever.metadatas = [
        {
            "source": f"{document_id}.pdf",
            "source_path": f"/{document_id}.pdf",
            "standard_id": "IRC:TEST",
            "content_hash": document_id,
        }
        for document_id in retriever.ids
    ]
    retriever.id_to_index = {
        document_id: index for index, document_id in enumerate(retriever.ids)
    }
    retriever.content_hash_to_id = {document_id: document_id for document_id in retriever.ids}
    retriever.compound_to_id = {}
    retriever.text_to_id = {
        text: retriever.ids[index] for index, text in enumerate(retriever.documents)
    }
    retriever.source_chunk_lookup = {}
    retriever.settings = SimpleNamespace(
        dense_k=20,
        lexical_k=20,
        exhaustive_retrieval=False,
        final_k=20,
        neighbor_window=1,
    )
    retriever.vector_store = FakeVectorStore(retriever, dense_rankings or {})
    retriever.bm25 = FakeBM25(retriever, lexical_rankings or {})
    retriever.reranker = reranker
    retriever._queries = lambda metric, context: list(QUERIES)
    return retriever


def _metric():
    return SimpleNamespace(
        key="min_sign_height",
        preferred_standards=(),
        search_phrases=lambda: ["minimum sign height", "mounting height"],
    )


def _target(hits):
    return next(hit for hit in hits if hit.content_hash == "target")


def _hit(*, dense=None, lexical=None, score=0.5, evidence_id="E-1", text="chunk"):
    metadata = {}
    if dense is not None:
        metadata["dense_best_query"] = dense
    if lexical is not None:
        metadata["lexical_best_query"] = lexical
    return RetrievalHit(
        evidence_id=evidence_id,
        text=text,
        source="standard.pdf",
        score=score,
        metadata=metadata,
    )


def test_dense_provenance_retains_query_with_best_rank():
    retriever = _retriever(
        dense_rankings={
            QUERIES[0]: _order_with_target(8),
            QUERIES[1]: _order_with_target(2),
            QUERIES[2]: _order_with_target(5),
        }
    )

    target = _target(retriever.retrieve(_metric(), RoadContext()))

    assert target.dense_rank == 2
    assert target.metadata["dense_best_query"] == QUERIES[1]


def test_lexical_provenance_retains_query_with_best_rank():
    retriever = _retriever(
        lexical_rankings={
            QUERIES[0]: _order_with_target(3),
            QUERIES[1]: _order_with_target(7),
            QUERIES[2]: _order_with_target(6),
        }
    )

    target = _target(retriever.retrieve(_metric(), RoadContext()))

    assert target.lexical_rank == 3
    assert target.metadata["lexical_best_query"] == QUERIES[0]


def test_same_best_query_creates_one_reranker_pair():
    reranker = RecordingReranker([0.4])
    retriever = _retriever(reranker=reranker)
    hit = _hit(dense=QUERIES[1], lexical=QUERIES[1])

    retriever._rerank(QUERIES, [hit])

    assert reranker.calls == 1
    assert reranker.pairs == [(QUERIES[1], "chunk")]
    assert hit.metadata["reranker_queries_evaluated"] == [QUERIES[1]]


def test_different_best_queries_are_batched_once_and_both_evaluated():
    reranker = RecordingReranker([0.4, 0.7])
    retriever = _retriever(reranker=reranker)
    hit = _hit(dense=QUERIES[1], lexical=QUERIES[0])

    retriever._rerank(QUERIES, [hit])

    assert reranker.calls == 1
    assert reranker.pairs == [(QUERIES[1], "chunk"), (QUERIES[0], "chunk")]


def test_maximum_semantic_relevance_and_winning_raw_score_are_retained():
    reranker = RecordingReranker([-1.0, 2.0])
    retriever = _retriever(reranker=reranker)
    hit = _hit(dense=QUERIES[1], lexical=QUERIES[0])

    retriever._rerank(QUERIES, [hit])

    expected_semantic = 1.0 / (1.0 + math.exp(-2.0))
    assert hit.score == pytest.approx(0.88 * expected_semantic + 0.12 * 0.5)
    assert hit.metadata["reranker_score"] == 2.0
    assert hit.metadata["reranker_winning_query"] == QUERIES[0]


def test_only_dense_provenance_uses_only_dense_query():
    reranker = RecordingReranker([0.3])
    retriever = _retriever(reranker=reranker)

    retriever._rerank(QUERIES, [_hit(dense=QUERIES[1])])

    assert reranker.pairs == [(QUERIES[1], "chunk")]


def test_only_lexical_provenance_uses_only_lexical_query():
    reranker = RecordingReranker([0.3])
    retriever = _retriever(reranker=reranker)

    retriever._rerank(QUERIES, [_hit(lexical=QUERIES[0])])

    assert reranker.pairs == [(QUERIES[0], "chunk")]


def test_provenance_less_neighbor_falls_back_to_canonical_query():
    reranker = RecordingReranker([0.3])
    retriever = _retriever(reranker=reranker)
    neighbor = _hit()

    retriever._rerank(QUERIES, [neighbor])

    assert reranker.pairs == [(QUERIES[0], "chunk")]
    assert neighbor.metadata["reranker_winning_query"] == QUERIES[0]


def test_best_rank_tie_retains_earliest_expanded_query():
    retriever = _retriever(
        dense_rankings={
            QUERIES[0]: _order_with_target(2),
            QUERIES[1]: _order_with_target(2),
            QUERIES[2]: _order_with_target(4),
        },
        lexical_rankings={
            QUERIES[0]: _order_with_target(2),
            QUERIES[1]: _order_with_target(2),
            QUERIES[2]: _order_with_target(4),
        },
    )

    target = _target(retriever.retrieve(_metric(), RoadContext()))

    assert target.dense_rank == 2
    assert target.metadata["dense_best_query"] == QUERIES[0]
    assert target.lexical_rank == 2
    assert target.metadata["lexical_best_query"] == QUERIES[0]


def test_duplicate_query_chunk_pairs_are_globally_evaluated_once():
    reranker = RecordingReranker([0.3])
    retriever = _retriever(reranker=reranker)
    first = _hit(dense=QUERIES[1], evidence_id="E-first", text="same chunk")
    second = _hit(lexical=QUERIES[1], evidence_id="E-second", text="same chunk")

    retriever._rerank(QUERIES, [first, second])

    assert reranker.calls == 1
    assert reranker.pairs == [(QUERIES[1], "same chunk")]
    assert first.metadata["reranker_score"] == second.metadata["reranker_score"] == 0.3


def test_reranker_disabled_preserves_fusion_ranking():
    retriever = _retriever(
        dense_rankings={
            QUERIES[0]: ["doc-1", "target"],
            QUERIES[1]: ["doc-1", "target"],
            QUERIES[2]: ["doc-1", "target"],
        },
        reranker=None,
    )

    hits = retriever.retrieve(_metric(), RoadContext())

    assert hits[0].content_hash == "doc-1"
    assert "dense_best_query" in hits[0].metadata
    assert "reranker_score" not in hits[0].metadata


def test_existing_fusion_normalization_and_weighting_are_unchanged():
    reranker = RecordingReranker([0.0, 0.0])
    retriever = _retriever(reranker=reranker)
    low_fusion = _hit(score=0.2, evidence_id="E-low", text="low")
    high_fusion = _hit(score=0.8, evidence_id="E-high", text="high")

    ranked = retriever._rerank(QUERIES, [low_fusion, high_fusion])
    by_id = {hit.evidence_id: hit for hit in ranked}

    assert by_id["E-low"].score == pytest.approx(0.88 * 0.5 + 0.12 * 0.0)
    assert by_id["E-high"].score == pytest.approx(0.88 * 0.5 + 0.12 * 1.0)
    assert by_id["E-low"].metadata["fusion_score"] == 0.2
    assert by_id["E-high"].metadata["fusion_score"] == 0.8
    assert reranker.calls == 1
    assert len(reranker.pairs) == 2
