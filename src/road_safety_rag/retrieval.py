from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .catalog import MetricSpec
from .config import Settings
from .models import RetrievalHit, RoadContext

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.:/-][a-z0-9]+)*", re.IGNORECASE)
NUMERIC_WITH_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|metres?|meters?|km|km/h)\b", re.I)

PROJECT_HIGHWAY_MANUALS = {
    "IRC:SP:73": ("undivided", 2),
    "IRC:SP:84": ("divided", 4),
    "IRC:SP:87": ("divided", 6),
}


def standard_configuration_status(
    metric: MetricSpec, context: RoadContext, standard_id: str
) -> str:
    """Return applicable, inapplicable, unknown, or not_scoped."""

    if metric.key not in {"min_lane_width", "min_radius_curvature"}:
        return "not_scoped"
    normalized = standard_id.upper().strip()
    manual = next(
        (
            key
            for key in PROJECT_HIGHWAY_MANUALS
            if normalized == key or normalized.startswith(f"{key}-")
        ),
        None,
    )
    if manual is not None:
        expected_carriageway, expected_total = PROJECT_HIGHWAY_MANUALS[manual]
        if context.carriageway not in {"unknown", expected_carriageway}:
            return "inapplicable"
        total = context.total_road_lanes
        if total is not None and total != expected_total:
            return "inapplicable"
        if context.carriageway == "unknown" or total is None:
            return "unknown"
        return "applicable"

    if normalized == "IRC:SP:99" or normalized.startswith("IRC:SP:99-"):
        road_class = (context.road_class or "").casefold()
        if not road_class:
            return "unknown"
        return "applicable" if "expressway" in road_class else "inapplicable"
    # IRC:86 is an urban-roads geometric-design standard. It must not become
    # the fallback source for a rural NH mainline after a project manual is
    # correctly excluded.
    if normalized == "IRC:86" or normalized.startswith("IRC:86-"):
        return "inapplicable" if context.setting == "rural" else "not_scoped"
    return "not_scoped"


def configuration_missing_context(context: RoadContext) -> list[str]:
    missing: list[str] = []
    if context.carriageway == "unknown":
        missing.append("verified_carriageway")
    if context.total_road_lanes is None:
        missing.append("verified_total_road_lanes")
    return missing


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple[str, ...]:
    tokens = [token.casefold().strip(".:/-") for token in TOKEN_RE.findall(text)]
    return tuple(token for token in tokens if len(token) > 1 or token.isdigit())


@dataclass
class BM25Index:
    tokenized_documents: list[tuple[str, ...]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_lengths = [len(document) for document in self.tokenized_documents]
        self.average_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        document_frequency: Counter[str] = Counter()
        for document in self.tokenized_documents:
            document_frequency.update(set(document))
        total = len(self.tokenized_documents)
        self.idf = {
            token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.term_frequencies = [Counter(document) for document in self.tokenized_documents]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        query_tokens = Counter(tokenize(query))
        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.doc_lengths[index] / max(self.average_length, 1.0)
            )
            for token, query_frequency in query_tokens.items():
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                score += (
                    self.idf.get(token, 0.0)
                    * (frequency * (self.k1 + 1.0) / (frequency + length_norm))
                    * min(query_frequency, 2)
                )
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


class HybridRetriever:
    """Dense + lexical retrieval with RRF, source priors, and diversity."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_store = self._vector_store()
        raw = self.vector_store._collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = list(raw.get("ids") or [])
        self.documents: list[str] = list(raw.get("documents") or [])
        self.metadatas: list[dict] = list(raw.get("metadatas") or [])
        if not self.ids:
            raise RuntimeError(
                f"Collection '{settings.collection_name}' is empty. Run the index command first."
            )
        self.id_to_index = {document_id: index for index, document_id in enumerate(self.ids)}
        self.content_hash_to_id = {
            str(metadata["content_hash"]): self.ids[index]
            for index, metadata in enumerate(self.metadatas)
            if metadata.get("content_hash")
        }
        self.compound_to_id = {
            (
                str(metadata.get("source_path", "")),
                metadata.get("page"),
                metadata.get("chunk_index"),
            ): self.ids[index]
            for index, metadata in enumerate(self.metadatas)
        }
        self.text_to_id = {text: self.ids[index] for index, text in enumerate(self.documents)}
        self.bm25 = BM25Index([tokenize(document) for document in self.documents])
        self.source_chunk_lookup: dict[tuple[str, int], int] = {}
        editions: defaultdict[str, set[int]] = defaultdict(set)
        for index, metadata in enumerate(self.metadatas):
            source_path = str(metadata.get("source_path", metadata.get("source", "")))
            chunk_index = metadata.get("chunk_index")
            if isinstance(chunk_index, int):
                self.source_chunk_lookup[(source_path, chunk_index)] = index
            standard_id = metadata.get("standard_id")
            edition_year = metadata.get("edition_year")
            if standard_id and isinstance(edition_year, int):
                editions[str(standard_id)].add(edition_year)
        self.standard_editions = {key: frozenset(value) for key, value in editions.items()}
        self.reranker = self._load_reranker() if settings.enable_reranker else None

    def retrieve(self, metric: MetricSpec, context: RoadContext) -> list[RetrievalHit]:
        queries = self._queries(metric, context)
        dense_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}
        scan_ranks: dict[str, int] = {}

        for query in queries:
            for rank, document in enumerate(
                self.vector_store.similarity_search(query, k=self.settings.dense_k), start=1
            ):
                document_id = self._resolve_id(document.page_content, document.metadata)
                if document_id:
                    dense_ranks[document_id] = min(rank, dense_ranks.get(document_id, 10**9))

            for rank, (index, _score) in enumerate(
                self.bm25.search(query, self.settings.lexical_k), start=1
            ):
                document_id = self.ids[index]
                lexical_ranks[document_id] = min(rank, lexical_ranks.get(document_id, 10**9))

        if self.settings.exhaustive_retrieval:
            scan_ranks = self._exhaustive_scan(metric, context)

        candidate_ids = set(dense_ranks) | set(lexical_ranks) | set(scan_ranks)
        scored: list[RetrievalHit] = []
        query_tokens = set(tokenize(" ".join(queries)))
        exact_phrases = {
            " ".join(tokenize(phrase))
            for phrase in metric.search_phrases()
            if len(tokenize(phrase)) >= 2
        }
        for document_id in candidate_ids:
            index = self.id_to_index.get(document_id)
            if index is None:
                continue
            text = self.documents[index]
            metadata = self.metadatas[index]
            dense_rank = dense_ranks.get(document_id)
            lexical_rank = lexical_ranks.get(document_id)
            scan_rank = scan_ranks.get(document_id)
            score = 0.0
            if dense_rank:
                score += (0.46 if scan_ranks else 0.56) / (60 + dense_rank)
            if lexical_rank:
                score += (0.34 if scan_ranks else 0.44) / (60 + lexical_rank)
            if scan_rank:
                score += 0.20 / (60 + scan_rank)
            standard_id = str(metadata.get("standard_id", ""))
            if standard_configuration_status(metric, context, standard_id) == "inapplicable":
                continue
            if any(
                standard_id.upper().startswith(item.upper()) for item in metric.preferred_standards
            ):
                score *= 1.18
            score *= self._context_standard_multiplier(metric, context, standard_id)
            overlap = len(query_tokens.intersection(tokenize(text))) / max(len(query_tokens), 1)
            score *= 1.0 + min(overlap, 0.35)
            normalized_text = " ".join(tokenize(text))
            exact_match = any(phrase in normalized_text for phrase in exact_phrases)
            has_value_with_unit = bool(NUMERIC_WITH_UNIT_RE.search(text))
            if exact_match:
                score *= 1.28
            if has_value_with_unit:
                score *= 1.12
            else:
                score *= 0.82
            # Exact requirement language plus an explicit unit is a stronger
            # signal than a semantically similar discussion or contents page.
            if exact_match and has_value_with_unit:
                score += 0.022
            scored.append(self._hit(index, score, dense_rank, lexical_rank))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        expanded = self._expand_neighbors(scored[: max(self.settings.final_k, 6)])
        combined = self._deduplicate(scored + expanded)
        if self.reranker:
            combined = self._rerank(queries[0], combined[:30])
        return self._diversify(combined, self.settings.final_k)

    def _exhaustive_scan(self, metric: MetricSpec, context: RoadContext) -> dict[str, int]:
        """Scan every indexed chunk for explicit feature + measurement evidence.

        This slower path complements approximate vector search.  It is useful
        for OCR'd standards, where a rare table label or hyphenated feature can
        have a weak embedding despite containing the exact dimension.
        """

        phrases = [set(tokenize(phrase)) for phrase in metric.search_phrases()]
        preferred = tuple(item.upper() for item in metric.preferred_standards)
        feature_patterns = {
            "min_lane_width": r"\b(?:lane|carriageway)\b",
            "min_sign_height": r"\b(?:sign|mounting|clearance)\b",
            "traffic_sign_width": r"\b(?:sign|diameter|width)\b",
            "traffic_sign_height": r"\b(?:sign|diameter|height)\b",
            "min_kerb_height": r"\b(?:kerb|curb)\b",
            "min_w_beam_barrier_height": r"\bw[\s-]?beam\b",
            "min_concrete_barrier_height": r"\b(?:concrete|new\s+jersey|rigid)\b",
            "min_radius_curvature": r"\b(?:radius|radii|curve)\b",
        }
        pattern = re.compile(feature_patterns[metric.key], re.I)
        scored: list[tuple[str, float]] = []
        context_tokens = set(tokenize(context.compact_description()))
        for index, text in enumerate(self.documents):
            if not pattern.search(text) or not NUMERIC_WITH_UNIT_RE.search(text):
                continue
            tokens = set(tokenize(text))
            phrase_overlap = max(
                (len(tokens.intersection(item)) / max(len(item), 1) for item in phrases),
                default=0.0,
            )
            if phrase_overlap < 0.34:
                continue
            metadata = self.metadatas[index]
            standard_id = str(metadata.get("standard_id", "")).upper()
            standard_bonus = (
                0.35 if any(standard_id.startswith(item) for item in preferred) else 0.0
            )
            context_overlap = len(tokens.intersection(context_tokens)) / max(len(context_tokens), 1)
            score = phrase_overlap + standard_bonus + min(context_overlap, 0.25)
            scored.append((self.ids[index], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        limit = max(self.settings.lexical_k, self.settings.final_k * 4, 60)
        return {document_id: rank for rank, (document_id, _score) in enumerate(scored[:limit], 1)}

    @staticmethod
    def _context_standard_multiplier(
        metric: MetricSpec, context: RoadContext, standard_id: str
    ) -> float:
        if metric.key not in {"min_lane_width", "min_radius_curvature"}:
            return 1.0
        normalized = standard_id.upper()
        road_class = (context.road_class or "").casefold()
        if "expressway" in road_class:
            return 1.4 if normalized.startswith("IRC:SP:99") else 0.9
        status = standard_configuration_status(metric, context, standard_id)
        if status == "applicable":
            return 1.65
        if status == "unknown":
            return 0.72
        return 1.0

    def _queries(self, metric: MetricSpec, context: RoadContext) -> list[str]:
        standard_hint = " ".join(metric.preferred_standards)
        base_context = context.compact_description()
        phrases = metric.search_phrases()[:3]
        return [
            f"{phrase}; {base_context}; applicable requirement table standard {standard_hint}"
            for phrase in phrases
        ]

    def _resolve_id(self, text: str, metadata: dict) -> str | None:
        content_hash = metadata.get("content_hash")
        source_path = metadata.get("source_path")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        if content_hash and str(content_hash) in self.content_hash_to_id:
            return self.content_hash_to_id[str(content_hash)]
        compound = (str(source_path or ""), page, chunk_index)
        if source_path and compound in self.compound_to_id:
            return self.compound_to_id[compound]
        return self.text_to_id.get(text)

    def _hit(
        self,
        index: int,
        score: float,
        dense_rank: int | None = None,
        lexical_rank: int | None = None,
    ) -> RetrievalHit:
        metadata = dict(self.metadatas[index])
        return RetrievalHit(
            evidence_id=f"E-{self.ids[index][:12]}",
            text=self.documents[index],
            source=str(metadata.get("source", "unknown")),
            page=_int_or_none(metadata.get("page")),
            section=_str_or_none(metadata.get("section")),
            standard_id=_str_or_none(metadata.get("standard_id")),
            edition_year=_int_or_none(metadata.get("edition_year")),
            chunk_index=_int_or_none(metadata.get("chunk_index")),
            content_hash=_str_or_none(metadata.get("content_hash")),
            score=score,
            dense_rank=dense_rank,
            lexical_rank=lexical_rank,
            metadata=metadata,
        )

    def _expand_neighbors(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        expanded: list[RetrievalHit] = []
        hit_limit = 10 if self.settings.exhaustive_retrieval else 5
        window = max(1, self.settings.neighbor_window)
        for hit in hits[:hit_limit]:
            source_path = str(hit.metadata.get("source_path", hit.source))
            if hit.chunk_index is None:
                continue
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_index = hit.chunk_index + offset
                corpus_index = self.source_chunk_lookup.get((source_path, neighbor_index))
                if corpus_index is None:
                    continue
                neighbor = self._hit(corpus_index, hit.score * 0.92)
                expanded.append(neighbor)
        return expanded

    @staticmethod
    def _deduplicate(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        best: dict[str, RetrievalHit] = {}
        for hit in hits:
            key = hit.content_hash or re.sub(r"\s+", " ", hit.text.casefold())[:500]
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return sorted(best.values(), key=lambda hit: hit.score, reverse=True)

    @staticmethod
    def _diversify(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
        selected: list[RetrievalHit] = []
        per_source: defaultdict[str, int] = defaultdict(int)
        per_page: defaultdict[tuple[str, int | None], int] = defaultdict(int)
        for hit in hits:
            page_key = (hit.source, hit.page)
            if per_source[hit.source] >= 4 or per_page[page_key] >= 2:
                continue
            selected.append(hit)
            per_source[hit.source] += 1
            per_page[page_key] += 1
            if len(selected) >= limit:
                break
        return selected

    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(
                self.settings.reranker_model,
                device="cpu",
                local_files_only=not self.settings.allow_model_download,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reranker was enabled but '{self.settings.reranker_model}' could not load: {exc}"
                " Cache the model first or temporarily set ROAD_RAG_ALLOW_MODEL_DOWNLOAD=true."
            ) from exc

    def _rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        scores = self.reranker.predict([(query, hit.text) for hit in hits])
        fusion_scores = [hit.score for hit in hits]
        low = min(fusion_scores, default=0.0)
        high = max(fusion_scores, default=0.0)
        span = high - low
        for hit, raw_score in zip(hits, scores):
            reranker_score = float(raw_score)
            semantic_score = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, reranker_score))))
            fusion_score = (hit.score - low) / span if span > 0 else 0.5
            hit.metadata["fusion_score"] = round(hit.score, 8)
            hit.metadata["reranker_score"] = round(reranker_score, 8)
            # Cross-encoder relevance dominates, while a small RRF/source-prior
            # contribution prevents near-ties from discarding applicability signals.
            hit.score = 0.88 * semantic_score + 0.12 * fusion_score
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def _vector_store(self):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Retrieval requires langchain-chroma and langchain-huggingface."
            ) from exc
        embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={
                "device": "cpu",
                "local_files_only": not self.settings.allow_model_download,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.settings.persist_directory),
        )


def format_evidence(hits: list[RetrievalHit], max_chars: int = 18000) -> str:
    blocks: list[str] = []
    used = 0
    for hit in hits:
        header = (
            f"[{hit.evidence_id}] source={hit.source}; standard={hit.standard_id or 'unknown'}; "
            f"edition={hit.edition_year or 'unknown'}; page={hit.page or 'unknown'}; "
            f"section={hit.section or 'unknown'}"
        )
        block = f"{header}\n{hit.text.strip()}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _int_or_none(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _str_or_none(value) -> str | None:
    return str(value) if value not in (None, "") else None
