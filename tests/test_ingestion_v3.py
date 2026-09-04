from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from road_safety_rag.cli import retrieval_mode_settings
from road_safety_rag.config import Settings
from road_safety_rag.ingestion import (
    CHUNKING_VERSION,
    IndexBuilder,
    PageChunker,
    PageExtractor,
    resolve_standard_identity,
)
from road_safety_rag.models import RetrievalHit
from road_safety_rag.registry import StandardsRegistry
from road_safety_rag.retrieval import HybridRetriever


def _settings(root: Path) -> Settings:
    return Settings(
        project_dir=root,
        corpus_dirs=(),
        persist_directory=root / "db",
        standards_registry=root / "registry.json",
        require_verified_standards=False,
    )


def _registry(tmp_path: Path, standard: dict | None = None) -> StandardsRegistry:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "standards": {
                    "IRC:SP:84": standard
                    or {
                        "active_edition_year": None,
                        "approved_sources": [],
                        "source_aliases": [],
                        "approved_sha256": [],
                        "official_source_url": None,
                        "licence_basis": None,
                        "amendments": [],
                        "supersedes": None,
                        "reviewed_by": None,
                        "reviewed_on": None,
                        "notes": "",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return StandardsRegistry.load(path)


def test_large_markdown_table_repeats_header_without_splitting_rows():
    rows = [
        f"| {speed} km/h | {speed * 3}.0 m | Plain terrain requirement row {speed} |"
        for speed in range(20, 121, 10)
    ]
    table = "\n".join(
        [
            "4.3 Horizontal Curves",
            "Table 4.2 Minimum curve radius",
            "| Design speed | Minimum radius | Applicability |",
            "|---|---:|---|",
            *rows,
        ]
    )
    parts = PageChunker(chunk_size=400, overlap=60).split_with_metadata(table)
    table_parts = [part for part in parts if part.content_type == "table"]
    assert len(table_parts) > 1
    assert len({part.table_id for part in table_parts}) == 1
    for part in table_parts:
        assert "| Design speed | Minimum radius | Applicability |" in part.text
        assert "|---|---:|---|" in part.text
        assert part.table_row_start is not None
        assert part.table_row_end is not None
        for line in part.text.splitlines():
            if line.startswith("|"):
                assert line.endswith("|")
                assert line.count("|") == 4
    covered = [
        row
        for part in table_parts
        for row in range(part.table_row_start or 0, (part.table_row_end or -1) + 1)
    ]
    assert covered == list(range(1, len(rows) + 1))


def test_registry_hash_has_precedence_over_damaged_ocr(tmp_path: Path):
    digest = "a" * 64
    registry = _registry(
        tmp_path,
        {
            "active_edition_year": 2019,
            "approved_sources": ["official.pdf"],
            "source_aliases": [],
            "approved_sha256": [digest],
            "official_source_url": "https://example.invalid/official.pdf",
            "licence_basis": "institutional copy",
            "amendments": [],
            "supersedes": None,
            "reviewed_by": "Reviewer",
            "reviewed_on": "2026-01-01",
            "notes": "",
        },
    )
    identity = resolve_standard_identity(
        "damaged-name.pdf", "unreadable title page", digest, registry
    )
    assert identity.standard_id == "IRC:SP:84"
    assert identity.edition_year == 2019
    assert identity.method == "registry_sha256"


def test_ocr_normalization_accepts_narrow_irc_confusion(tmp_path: Path):
    identity = resolve_standard_identity(
        "scan.pdf", "lRC : SP - 84 - 2019", None, _registry(tmp_path)
    )
    assert identity.standard_id == "IRC:SP:84"
    assert identity.method == "ocr_normalized_cover"


def test_fuzzy_identity_remains_candidate_not_truth(tmp_path: Path):
    identity = resolve_standard_identity(
        "damaged.pdf", "lXQ SP 84 engineering manual", None, _registry(tmp_path)
    )
    assert identity.standard_id is None
    if identity.candidate_standard_id is not None:
        assert identity.method == "fuzzy_candidate"
        assert identity.candidate_score is not None


def test_old_index_manifest_is_rejected_instead_of_mixed(tmp_path: Path):
    database = tmp_path / "db"
    database.mkdir()
    (database / "index_manifest.json").write_text(
        json.dumps({"schema_version": 2, "documents": {"old.pdf": {"chunks": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="new empty v3"):
        IndexBuilder(_settings(tmp_path))._load_manifest()


def test_retrieval_modes_enable_reranking_only_when_requested(tmp_path: Path):
    base = _settings(tmp_path)
    assert not retrieval_mode_settings(base, "fast").enable_reranker
    assert retrieval_mode_settings(base, "balanced").enable_reranker
    audit = retrieval_mode_settings(base, "audit")
    assert audit.enable_reranker
    assert audit.exhaustive_retrieval
    assert audit.final_k >= 12


def test_reranker_records_both_semantic_and_fusion_scores():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.reranker = SimpleNamespace(predict=lambda pairs: [-1.0, 2.0])
    hits = [
        RetrievalHit(evidence_id="E-1", text="weak", source="a", score=0.9),
        RetrievalHit(evidence_id="E-2", text="strong", source="b", score=0.2),
    ]
    ranked = retriever._rerank("minimum lane width", hits)
    assert ranked[0].evidence_id == "E-2"
    assert "fusion_score" in ranked[0].metadata
    assert "reranker_score" in ranked[0].metadata


def test_chunking_version_is_explicit():
    assert CHUNKING_VERSION == "table-aware-v1"


def test_page_extractor_keeps_one_lazy_docling_converter_per_run():
    extractor = PageExtractor()
    assert extractor._docling_converter is None
