from __future__ import annotations

import json
from pathlib import Path

from .models import RetrievalHit


class StructuredEvidenceRegistry:
    """Source-hash-bound transcriptions for facts printed only in figures."""

    def __init__(self, records: list[dict[str, object]] | None = None):
        self.records = records or []

    @classmethod
    def load(cls, path: Path) -> "StructuredEvidenceRegistry":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(list(payload.get("records", [])))

    def hits(self, metric_key: str, manifest_path: Path) -> list[RetrievalHit]:
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        indexed_hashes = {
            str(info.get("sha256", "")).casefold()
            for info in manifest.get("documents", {}).values()
            if isinstance(info, dict)
        }
        hits: list[RetrievalHit] = []
        for record in self.records:
            if record.get("metric_key") != metric_key:
                continue
            source_hash = str(record.get("source_sha256", "")).casefold()
            if not source_hash or source_hash not in indexed_hashes:
                continue
            hits.append(
                RetrievalHit(
                    evidence_id=str(record["evidence_id"]),
                    text=str(record["text"]),
                    source=str(record["source"]),
                    page=int(record["page"]),
                    section=str(record.get("section") or "") or None,
                    standard_id=str(record.get("standard_id") or "") or None,
                    edition_year=(
                        int(record["edition_year"])
                        if record.get("edition_year") is not None
                        else None
                    ),
                    content_hash=source_hash,
                    score=1.0,
                    metadata={
                        "evidence_kind": "source_hash_bound_figure_transcription",
                        "source_sha256": source_hash,
                        "verification": str(record.get("verification", "visual_source_check")),
                    },
                )
            )
        return hits
