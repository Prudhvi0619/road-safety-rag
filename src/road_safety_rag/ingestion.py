from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .config import Settings
from .registry import StandardsRegistry

INDEX_SCHEMA_VERSION = 3
CHUNKING_VERSION = "table-aware-v1"
IDENTITY_VERSION = "registry-first-v1"
STANDARD_PATTERN = re.compile(
    r"\bIRC\s*[:._-]?\s*(?:(SP)\s*[:._-]?\s*)?(\d{1,3})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    parser: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class ChunkPart:
    text: str
    content_type: str = "prose"
    table_id: str | None = None
    table_caption: str | None = None
    table_header: str | None = None
    table_row_start: int | None = None
    table_row_end: int | None = None


@dataclass(frozen=True)
class StandardIdentity:
    standard_id: str | None
    edition_year: int | None
    method: str
    confidence: float
    candidate_standard_id: str | None = None
    candidate_score: float | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\x08", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _identity_from_sample(sample: str, fallback_text: str) -> tuple[str | None, int | None]:
    match = STANDARD_PATTERN.search(sample)
    if not match:
        return None, None
    standard_id = f"IRC:{'SP:' if match.group(1) else ''}{int(match.group(2))}"
    tail = sample[match.start() : match.start() + 80]
    year_match = re.search(r"\b(?:19|20)\d{2}\b", tail)
    edition_year = int(year_match.group()) if year_match else None
    if edition_year is None:
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", fallback_text[:5000])]
        plausible = [year for year in years if 1950 <= year <= datetime.now().year]
        if plausible:
            edition_year = plausible[0]
    return standard_id, edition_year


def _normalize_ocr_standard_text(text: str) -> str:
    """Repair only narrow, well-known OCR confusions around the IRC token."""

    return re.sub(
        r"(?i)(?<![A-Z0-9])[1IL]\s*R\s*C(?=\s*[:._-]?\s*(?:SP\b|\d))",
        "IRC",
        text,
    )


def identify_standard(filename: str, first_pages_text: str) -> tuple[str | None, int | None]:
    """Backward-compatible strict identity extraction.

    Cover text remains preferable to the filename. A narrowly normalized OCR
    pass handles ``1RC``/``lRC`` without using fuzzy matching as truth.
    """

    content_sample = first_pages_text[:12000]
    for sample, fallback in (
        (content_sample, first_pages_text),
        (filename, first_pages_text),
        (_normalize_ocr_standard_text(content_sample), first_pages_text),
        (_normalize_ocr_standard_text(filename), first_pages_text),
    ):
        standard_id, edition_year = _identity_from_sample(sample, fallback)
        if standard_id:
            return standard_id, edition_year
    return None, None


def _fuzzy_identity_candidate(
    filename: str, first_pages_text: str, known_standard_ids: Iterable[str]
) -> tuple[str | None, float | None]:
    """Suggest a candidate identity; callers must never promote it automatically."""

    haystack = re.sub(
        r"[^A-Z0-9]",
        "",
        _normalize_ocr_standard_text(f"{filename} {first_pages_text[:3000]}").upper(),
    )
    known = {item.upper(): item for item in known_standard_ids}
    explicit_sp_numbers = {int(value) for value in re.findall(r"SP0*(\d{1,3})", haystack)}
    explicit_candidates = [
        known[key] for value in explicit_sp_numbers if (key := f"IRC:SP:{value}") in known
    ]
    if len(explicit_candidates) == 1:
        return explicit_candidates[0], 0.96
    best_id: str | None = None
    best_score = 0.0
    for standard_id in known.values():
        needle = re.sub(r"[^A-Z0-9]", "", standard_id.upper())
        if not needle:
            continue
        if needle in haystack:
            score = 1.0
        else:
            widths = range(max(3, len(needle) - 2), len(needle) + 3)
            score = max(
                (
                    SequenceMatcher(None, needle, haystack[start : start + width]).ratio()
                    for width in widths
                    for start in range(max(1, len(haystack) - width + 1))
                ),
                default=0.0,
            )
        if score > best_score:
            best_id, best_score = standard_id, score
    if best_score < 0.78:
        return None, None
    return best_id, round(best_score, 4)


def resolve_standard_identity(
    filename: str,
    first_pages_text: str,
    document_sha256: str | None,
    registry: StandardsRegistry,
) -> StandardIdentity:
    """Resolve a standard with registry facts first and safe OCR fallbacks."""

    policy, registry_method = registry.resolve_document(filename, document_sha256)
    if policy is not None:
        return StandardIdentity(
            standard_id=policy.standard_id,
            edition_year=policy.active_edition_year,
            method=registry_method or "registry",
            confidence=1.0 if registry_method == "registry_sha256" else 0.98,
        )

    content_sample = first_pages_text[:12000]
    standard_id, edition_year = _identity_from_sample(content_sample, first_pages_text)
    if standard_id:
        return StandardIdentity(standard_id, edition_year, "cover_regex", 0.95)
    standard_id, edition_year = _identity_from_sample(filename, first_pages_text)
    if standard_id:
        return StandardIdentity(standard_id, edition_year, "filename_regex", 0.82)

    normalized_cover = _normalize_ocr_standard_text(content_sample)
    standard_id, edition_year = _identity_from_sample(normalized_cover, first_pages_text)
    if standard_id:
        return StandardIdentity(standard_id, edition_year, "ocr_normalized_cover", 0.88)
    normalized_filename = _normalize_ocr_standard_text(filename)
    standard_id, edition_year = _identity_from_sample(normalized_filename, first_pages_text)
    if standard_id:
        return StandardIdentity(standard_id, edition_year, "ocr_normalized_filename", 0.78)

    candidate, score = _fuzzy_identity_candidate(
        filename, first_pages_text, registry.policies.keys()
    )
    return StandardIdentity(
        standard_id=None,
        edition_year=None,
        method="fuzzy_candidate" if candidate else "unknown",
        confidence=0.0,
        candidate_standard_id=candidate,
        candidate_score=score,
    )


def detect_section(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip(" #\t")
        if not stripped or len(stripped) > 160:
            continue
        if re.match(r"^(?:section\s+)?\d+(?:\.\d+){0,4}\s+[A-Za-z]", stripped, re.I):
            return stripped
        if line.lstrip().startswith("#"):
            return stripped
    return None


class PageExtractor:
    """Extract page-addressable content; use Docling OCR only when requested/needed."""

    def __init__(self) -> None:
        self._docling_converter = None

    def extract(self, pdf_path: Path, use_ocr: bool = False) -> tuple[list[PageText], list[str]]:
        pages, warnings = self._extract_pypdf(pdf_path)
        weak = [page for page in pages if len(re.sub(r"\s+", "", page.text)) < 80]
        weak_ratio = len(weak) / max(len(pages), 1)
        if use_ocr or weak_ratio > 0.25:
            try:
                docling_pages = self._extract_docling(pdf_path)
                by_page = {page.page: page for page in docling_pages}
                merged: list[PageText] = []
                for page in pages:
                    replacement = by_page.get(page.page)
                    if replacement:
                        merged.append(self._merge_page_text(page, replacement))
                    else:
                        merged.append(page)
                for page_num in sorted(set(by_page) - {page.page for page in merged}):
                    merged.append(by_page[page_num])
                pages = sorted(merged, key=lambda page: page.page)
            except Exception as exc:  # OCR is an optional recovery path.
                warnings.append(f"Docling OCR unavailable/failed for {pdf_path.name}: {exc}")

        remaining_weak = [page.page for page in pages if len(re.sub(r"\s+", "", page.text)) < 80]
        if remaining_weak:
            preview = ", ".join(map(str, remaining_weak[:12]))
            suffix = "..." if len(remaining_weak) > 12 else ""
            warnings.append(
                f"{pdf_path.name} has {len(remaining_weak)} low-text page(s): {preview}{suffix}. "
                "Rules from those pages may be unretrievable until OCR is improved."
            )
        return pages, warnings

    @staticmethod
    def _merge_page_text(base: PageText, ocr: PageText) -> PageText:
        """Keep body text while adding novel numeric content from figures/tables."""

        if len(re.sub(r"\s+", "", base.text)) < 80:
            return ocr if len(ocr.text) > len(base.text) else base
        base_flat = normalize_text(base.text).casefold()
        novel: list[str] = []
        for line in ocr.text.splitlines():
            candidate = normalize_text(line)
            folded = candidate.casefold()
            if len(folded) < 4 or folded in base_flat:
                continue
            # Diagram annotations are easily lost even when surrounding prose
            # gives the page a healthy text count. Preserve dimension-bearing
            # OCR lines and figure/table labels without duplicating all prose.
            has_dimension = bool(
                re.search(
                    r"\d(?:[\d., ]*\d)?\s*(?:±|\+\s*/\s*-)?\s*"
                    r"(?:mm|cm|m|metres?|meters?|km)\b",
                    candidate,
                    re.IGNORECASE,
                )
            )
            is_locator = bool(re.search(r"\b(?:fig(?:ure)?|table)\s*\.?\s*\d+", candidate, re.I))
            if has_dimension or is_locator:
                novel.append(candidate)
        if not novel:
            return base
        supplement = "\n".join(dict.fromkeys(novel))
        return PageText(
            page=base.page,
            text=normalize_text(f"{base.text}\n\n[OCR figure/table supplement]\n{supplement}"),
            parser=f"{base.parser}+{ocr.parser}",
        )

    @staticmethod
    def _extract_pypdf(pdf_path: Path) -> tuple[list[PageText], list[str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for page-aware ingestion") from exc

        reader = PdfReader(str(pdf_path))
        pages: list[PageText] = []
        warnings: list[str] = []
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise RuntimeError(f"Encrypted PDF cannot be opened: {pdf_path}") from exc
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"pypdf failed on {pdf_path.name} page {page_number}: {exc}")
                text = ""
            pages.append(PageText(page=page_number, text=text, parser="pypdf"))
        return pages, warnings

    def _extract_docling(self, pdf_path: Path) -> list[PageText]:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if self._docling_converter is None:
            options = PdfPipelineOptions()
            options.do_ocr = True
            self._docling_converter = DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=options)}
            )
        result = self._docling_converter.convert(str(pdf_path))
        page_parts: dict[int, list[str]] = defaultdict(list)
        for item, _level in result.document.iterate_items():
            provenance = getattr(item, "prov", None) or []
            if not provenance:
                continue
            page_number = int(provenance[0].page_no)
            text = getattr(item, "text", None)
            if not text and hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=result.document)
                except TypeError:
                    text = item.export_to_markdown()
            if text:
                page_parts[page_number].append(str(text))
        return [
            PageText(page=page, text=normalize_text("\n\n".join(parts)), parser="docling_ocr")
            for page, parts in sorted(page_parts.items())
        ]


class PageChunker:
    def __init__(self, chunk_size: int = 1400, overlap: int = 220):
        if chunk_size < 400:
            raise ValueError("chunk_size must be at least 400 characters")
        if not 0 <= overlap < chunk_size // 2:
            raise ValueError("overlap must be non-negative and less than half chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        return [part.text for part in self.split_with_metadata(text)]

    def split_with_metadata(self, text: str) -> list[ChunkPart]:
        text = normalize_text(text)
        if not text:
            return []
        parts: list[ChunkPart] = []
        prose_lines: list[str] = []
        active_heading: str | None = None
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            if self._is_table_line(lines[index]):
                end = index
                while end < len(lines) and self._is_table_line(lines[end]):
                    end += 1
                table_lines = [line.strip() for line in lines[index:end] if line.strip()]
                if self._is_markdown_table(table_lines):
                    prose = normalize_text("\n".join(prose_lines))
                    if prose:
                        parts.extend(ChunkPart(piece) for piece in self._split_prose(prose))
                        active_heading = self._last_heading(prose) or active_heading
                    caption = self._last_table_caption(prose)
                    parts.extend(
                        self._split_table(table_lines, caption=caption, heading=active_heading)
                    )
                    prose_lines = []
                    index = end
                    continue
            prose_lines.append(lines[index])
            index += 1

        prose = normalize_text("\n".join(prose_lines))
        if prose:
            parts.extend(ChunkPart(piece) for piece in self._split_prose(prose))
        return [part for part in parts if len(part.text) >= 40]

    def _split_prose(self, text: str) -> list[str]:
        paragraphs = [
            paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
        ]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            pieces = self._split_long(paragraph)
            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if current and len(candidate) > self.chunk_size:
                    chunks.append(current.strip())
                    tail = current[-self.overlap :]
                    boundary = max(tail.find(". "), tail.find("\n"))
                    if boundary >= 0:
                        tail = tail[boundary + 1 :]
                    current = f"{tail.strip()}\n\n{piece}".strip()
                else:
                    current = candidate
        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if len(chunk) >= 40]

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3

    @staticmethod
    def _is_markdown_table(lines: list[str]) -> bool:
        if len(lines) < 2:
            return False
        for line in lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                return True
        return False

    @staticmethod
    def _last_heading(prose: str) -> str | None:
        for line in reversed(prose.splitlines()):
            stripped = line.strip(" #\t")
            if not stripped or len(stripped) > 180:
                continue
            if line.lstrip().startswith("#") or re.match(
                r"^(?:section\s+)?\d+(?:\.\d+){0,4}\s+[A-Za-z]", stripped, re.I
            ):
                return stripped
        return None

    @staticmethod
    def _last_table_caption(prose: str) -> str | None:
        lines = [line.strip(" #\t") for line in prose.splitlines() if line.strip()]
        for line in reversed(lines[-6:]):
            if len(line) <= 240 and re.search(r"\b(?:table|schedule|annex)\b", line, re.I):
                return line
        return None

    def _split_table(
        self, table_lines: list[str], caption: str | None, heading: str | None
    ) -> list[ChunkPart]:
        separator_index = next(
            index
            for index, line in enumerate(table_lines)
            if all(
                re.fullmatch(r":?-{3,}:?", cell.strip())
                for cell in line.strip().strip("|").split("|")
            )
        )
        header_lines = table_lines[: separator_index + 1]
        data_rows = table_lines[separator_index + 1 :]
        table_hash = hashlib.sha256("\n".join(table_lines).encode("utf-8")).hexdigest()[:16]
        table_id = f"T-{table_hash}"
        context_lines: list[str] = []
        if heading and heading != caption:
            context_lines.append(f"Section: {heading}")
        if caption:
            context_lines.append(f"Table: {caption}")
        prefix_lines = context_lines + header_lines
        header_text = "\n".join(header_lines)
        target = max(self.chunk_size, int(self.chunk_size * 1.35))

        if not data_rows:
            text = normalize_text("\n".join(prefix_lines))
            return [
                ChunkPart(
                    text=text,
                    content_type="table",
                    table_id=table_id,
                    table_caption=caption,
                    table_header=header_text,
                    table_row_start=0,
                    table_row_end=0,
                )
            ]

        chunks: list[ChunkPart] = []
        selected_rows: list[str] = []
        row_start = 1
        for row_number, row in enumerate(data_rows, start=1):
            candidate = normalize_text("\n".join(prefix_lines + selected_rows + [row]))
            if selected_rows and len(candidate) > target:
                chunks.append(
                    self._table_part(
                        prefix_lines,
                        selected_rows,
                        table_id,
                        caption,
                        header_text,
                        row_start,
                        row_number - 1,
                    )
                )
                selected_rows = [row]
                row_start = row_number
            else:
                selected_rows.append(row)
        if selected_rows:
            chunks.append(
                self._table_part(
                    prefix_lines,
                    selected_rows,
                    table_id,
                    caption,
                    header_text,
                    row_start,
                    len(data_rows),
                )
            )
        return chunks

    @staticmethod
    def _table_part(
        prefix_lines: list[str],
        rows: list[str],
        table_id: str,
        caption: str | None,
        header_text: str,
        row_start: int,
        row_end: int,
    ) -> ChunkPart:
        return ChunkPart(
            text=normalize_text("\n".join(prefix_lines + rows)),
            content_type="table",
            table_id=table_id,
            table_caption=caption,
            table_header=header_text,
            table_row_start=row_start,
            table_row_end=row_end,
        )

    def _split_long(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        pieces: list[str] = []
        start = 0
        while start < len(paragraph):
            end = min(start + self.chunk_size, len(paragraph))
            if end < len(paragraph):
                candidates = [
                    paragraph.rfind(token, start + 300, end) for token in (". ", "; ", "\n", " ")
                ]
                best = max(candidates)
                if best > start:
                    end = best + 1
            pieces.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(end - self.overlap, start + 1)
        return pieces


class IndexBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.extractor = PageExtractor()
        self.chunker = PageChunker(settings.chunk_size, settings.chunk_overlap)
        registry_path = (
            settings.standards_registry
            or settings.project_dir / "config" / "standards_registry.json"
        )
        self.registry = StandardsRegistry.load(registry_path)

    def discover_pdfs(self) -> list[Path]:
        paths: dict[str, Path] = {}
        for corpus_dir in self.settings.corpus_dirs:
            if not corpus_dir.exists():
                continue
            for path in corpus_dir.rglob("*.pdf"):
                paths[str(path.resolve()).casefold()] = path.resolve()
        return sorted(paths.values(), key=lambda path: str(path).casefold())

    def build(
        self, use_ocr: bool = False, force: bool = False, prune: bool = False
    ) -> dict[str, object]:
        vector_store = self._vector_store()
        manifest = self._load_manifest()
        pdf_paths = self.discover_pdfs()
        summary: dict[str, object] = {
            "indexed_documents": 0,
            "skipped_documents": 0,
            "pruned_documents": 0,
            "pruned_sources": [],
            "chunks": 0,
            "warnings": [],
        }
        warnings: list[str] = summary["warnings"]  # type: ignore[assignment]

        if prune:
            active_sources = {str(path).casefold() for path in pdf_paths}
            stale_sources = [
                source
                for source in list(manifest.get("documents", {}))
                if self._source_is_in_scope(source) and source.casefold() not in active_sources
            ]
            for source in stale_sources:
                vector_store._collection.delete(where={"source_path": source})
                manifest.get("documents", {}).pop(source, None)
                manifest.get("failed_documents", {}).pop(source, None)
            summary["pruned_documents"] = len(stale_sources)
            summary["pruned_sources"] = stale_sources

        for pdf_path in pdf_paths:
            file_hash = sha256_file(pdf_path)
            existing = manifest.get("documents", {}).get(str(pdf_path))
            if (
                not force
                and existing
                and existing.get("sha256") == file_hash
                and existing.get("chunking_version") == CHUNKING_VERSION
                and existing.get("identity_version") == IDENTITY_VERSION
            ):
                summary["skipped_documents"] = int(summary["skipped_documents"]) + 1
                continue
            duplicate_sources = [
                source
                for source, info in manifest.get("documents", {}).items()
                if source != str(pdf_path) and info.get("sha256") == file_hash
            ]
            # --force means reprocess the same source with the improved parser;
            # it must not add a second copy of identical content from a new
            # folder into the existing collection.
            if duplicate_sources:
                warnings.append(
                    f"Skipped exact duplicate PDF: {pdf_path} (already indexed from {duplicate_sources[0]})"
                )
                summary["skipped_documents"] = int(summary["skipped_documents"]) + 1
                continue

            try:
                pages, extraction_warnings = self.extractor.extract(pdf_path, use_ocr=use_ocr)
            except Exception as exc:
                warning = f"Extraction failed for {pdf_path}: {exc}"
                warnings.append(warning)
                manifest.setdefault("failed_documents", {})[str(pdf_path)] = {
                    "sha256": file_hash,
                    "attempted_ocr": use_ocr,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "reason": str(exc),
                }
                self._save_manifest(manifest)
                continue
            warnings.extend(extraction_warnings)
            chunks = self._make_chunks(pdf_path, file_hash, pages)
            if not chunks:
                warnings.append(f"No indexable text found in {pdf_path}")
                manifest.setdefault("failed_documents", {})[str(pdf_path)] = {
                    "sha256": file_hash,
                    "pages": len(pages),
                    "attempted_ocr": use_ocr,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "No indexable text after extraction",
                }
                self._save_manifest(manifest)
                continue

            self._delete_previous_source(vector_store, pdf_path)
            self._add_chunks(vector_store, chunks)
            low_text_pages = [
                page.page for page in pages if len(re.sub(r"\s+", "", page.text)) < 80
            ]
            identity = self._document_identity(pdf_path, file_hash, pages)
            document_record: dict[str, object] = {
                "sha256": file_hash,
                "pages": len(pages),
                "chunks": len(chunks),
                "chunking_version": CHUNKING_VERSION,
                "identity_version": IDENTITY_VERSION,
                "identity_method": identity.method,
                "identity_confidence": identity.confidence,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "parsers": sorted({page.parser for page in pages}),
                "low_text_pages": low_text_pages,
                "page_text_coverage": round(
                    (len(pages) - len(low_text_pages)) / max(len(pages), 1), 4
                ),
            }
            if identity.standard_id:
                document_record["standard_id"] = identity.standard_id
            if identity.edition_year:
                document_record["edition_year"] = identity.edition_year
            if identity.candidate_standard_id:
                document_record["candidate_standard_id"] = identity.candidate_standard_id
                document_record["candidate_score"] = identity.candidate_score
            manifest.setdefault("documents", {})[str(pdf_path)] = document_record
            manifest.setdefault("failed_documents", {}).pop(str(pdf_path), None)
            summary["indexed_documents"] = int(summary["indexed_documents"]) + 1
            summary["chunks"] = int(summary["chunks"]) + len(chunks)
            self._save_manifest(manifest)

        manifest["schema_version"] = INDEX_SCHEMA_VERSION
        manifest["chunking_version"] = CHUNKING_VERSION
        manifest["identity_version"] = IDENTITY_VERSION
        manifest["collection_name"] = self.settings.collection_name
        manifest["embedding_model"] = self.settings.embedding_model
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["last_run"] = {
            "indexed_documents": summary["indexed_documents"],
            "skipped_documents": summary["skipped_documents"],
            "chunks": summary["chunks"],
            "warning_count": len(warnings),
            "used_ocr": use_ocr,
        }
        self._save_manifest(manifest)
        return summary

    def _source_is_in_scope(self, source: str) -> bool:
        try:
            source_path = Path(source).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        for corpus_dir in self.settings.corpus_dirs:
            try:
                source_path.relative_to(corpus_dir.expanduser().resolve())
                return True
            except ValueError:
                continue
        return False

    def repair_standard_metadata(self) -> dict[str, object]:
        """Re-identify standards from indexed cover pages without re-embedding text."""

        vector_store = self._vector_store()
        collection = vector_store._collection
        payload = collection.get(include=["documents", "metadatas"])
        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        grouped: dict[str, list[tuple[str, str, dict[str, object]]]] = defaultdict(list)
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = dict(metadata or {})
            source_path = str(metadata.get("source_path", ""))
            if source_path:
                grouped[source_path].append((chunk_id, document or "", metadata))

        changed_chunks = 0
        changed_documents: list[dict[str, object]] = []
        manifest = self._load_manifest()
        for source_path, chunks in grouped.items():
            cover_chunks = sorted(
                (item for item in chunks if int(item[2].get("page", 10_000)) <= 4),
                key=lambda item: (
                    int(item[2].get("page", 10_000)),
                    int(item[2].get("page_chunk_index", 10_000)),
                ),
            )
            cover_text = "\n".join(item[1] for item in cover_chunks)
            document_hashes = {
                str(item[2].get("document_sha256"))
                for item in chunks
                if item[2].get("document_sha256")
            }
            identity = resolve_standard_identity(
                Path(source_path).name,
                cover_text,
                next(iter(document_hashes)) if len(document_hashes) == 1 else None,
                self.registry,
            )
            old_ids = sorted(
                {str(item[2].get("standard_id")) for item in chunks if item[2].get("standard_id")}
            )
            old_years = sorted(
                {int(item[2]["edition_year"]) for item in chunks if item[2].get("edition_year")}
            )
            new_id = identity.standard_id or (old_ids[0] if len(old_ids) == 1 else None)
            new_year = identity.edition_year or (old_years[0] if len(old_years) == 1 else None)
            updates: list[tuple[str, dict[str, object]]] = []
            for chunk_id, _document, metadata in chunks:
                updated = dict(metadata)
                updated["identity_method"] = identity.method
                updated["identity_confidence"] = identity.confidence
                if new_id:
                    updated["standard_id"] = new_id
                if new_year:
                    updated["edition_year"] = new_year
                if identity.candidate_standard_id:
                    updated["candidate_standard_id"] = identity.candidate_standard_id
                    updated["candidate_score"] = identity.candidate_score or 0.0
                if updated != metadata:
                    updates.append((chunk_id, updated))
            for start in range(0, len(updates), 500):
                batch = updates[start : start + 500]
                collection.update(
                    ids=[item[0] for item in batch],
                    metadatas=[item[1] for item in batch],
                )
            if updates:
                changed_chunks += len(updates)
                changed_documents.append(
                    {
                        "source": source_path,
                        "old_standard_ids": old_ids,
                        "standard_id": new_id,
                        "edition_year": new_year,
                        "identity_method": identity.method,
                        "candidate_standard_id": identity.candidate_standard_id,
                        "changed_chunks": len(updates),
                    }
                )
            manifest_entry = manifest.get("documents", {}).get(source_path)
            if manifest_entry is not None:
                manifest_entry["standard_id"] = new_id
                manifest_entry["edition_year"] = new_year
                manifest_entry["identity_method"] = identity.method
                manifest_entry["identity_confidence"] = identity.confidence
                if identity.candidate_standard_id:
                    manifest_entry["candidate_standard_id"] = identity.candidate_standard_id
                    manifest_entry["candidate_score"] = identity.candidate_score

        manifest["metadata_repaired_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)
        return {
            "documents_checked": len(grouped),
            "documents_changed": len(changed_documents),
            "chunks_changed": changed_chunks,
            "changes": changed_documents,
        }

    def _document_identity(
        self, pdf_path: Path, file_hash: str, pages: list[PageText]
    ) -> StandardIdentity:
        first_pages = "\n".join(page.text for page in pages[:4])
        return resolve_standard_identity(pdf_path.name, first_pages, file_hash, self.registry)

    def _make_chunks(self, pdf_path: Path, file_hash: str, pages: list[PageText]) -> list[Chunk]:
        identity = self._document_identity(pdf_path, file_hash, pages)
        chunks: list[Chunk] = []
        document_chunk_index = 0
        for page in pages:
            for page_chunk_index, part in enumerate(self.chunker.split_with_metadata(page.text)):
                text = part.text
                content_hash = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                chunk_id = hashlib.sha256(
                    f"{file_hash}:{page.page}:{page_chunk_index}:{content_hash}".encode("utf-8")
                ).hexdigest()
                metadata: dict[str, str | int | float | bool] = {
                    "source": pdf_path.name,
                    "source_path": str(pdf_path.resolve()),
                    "document_sha256": file_hash,
                    "content_hash": content_hash,
                    "page": page.page,
                    "page_chunk_index": page_chunk_index,
                    "chunk_index": document_chunk_index,
                    "parser": page.parser,
                    "citation_quality": "page",
                    "content_type": part.content_type,
                    "chunking_version": CHUNKING_VERSION,
                    "identity_method": identity.method,
                    "identity_confidence": identity.confidence,
                }
                if part.table_id:
                    metadata["table_id"] = part.table_id
                if part.table_caption:
                    metadata["table_caption"] = part.table_caption
                if part.table_header:
                    metadata["table_header"] = part.table_header
                if part.table_row_start is not None:
                    metadata["table_row_start"] = part.table_row_start
                if part.table_row_end is not None:
                    metadata["table_row_end"] = part.table_row_end
                section = detect_section(text)
                if section:
                    metadata["section"] = section
                if identity.standard_id:
                    metadata["standard_id"] = identity.standard_id
                if identity.edition_year:
                    metadata["edition_year"] = identity.edition_year
                if identity.candidate_standard_id:
                    metadata["candidate_standard_id"] = identity.candidate_standard_id
                    metadata["candidate_score"] = identity.candidate_score or 0.0
                chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
                document_chunk_index += 1
        return chunks

    def _vector_store(self):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Indexing requires langchain-chroma and langchain-huggingface. "
                "Install requirements.txt in your project environment."
            ) from exc
        embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={
                "device": "cpu",
                "local_files_only": not self.settings.allow_model_download,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        self.settings.persist_directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.settings.persist_directory),
            collection_metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.settings.embedding_model,
                "schema_version": str(INDEX_SCHEMA_VERSION),
                "chunking_version": CHUNKING_VERSION,
            },
        )

    @staticmethod
    def _delete_previous_source(vector_store, pdf_path: Path) -> None:
        # Chroma's collection-level metadata delete is atomic and avoids stale
        # chunks when a document is replaced with a new edition/file content.
        vector_store._collection.delete(where={"source_path": str(pdf_path.resolve())})

    @staticmethod
    def _add_chunks(vector_store, chunks: list[Chunk]) -> None:
        from langchain_core.documents import Document

        batch_size = 96
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vector_store.add_documents(
                [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in batch],
                ids=[chunk.chunk_id for chunk in batch],
            )

    @property
    def manifest_path(self) -> Path:
        return self.settings.persist_directory / "index_manifest.json"

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunking_version": CHUNKING_VERSION,
                "identity_version": IDENTITY_VERSION,
                "documents": {},
            }
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        documents = manifest.get("documents", {})
        schema_version = manifest.get("schema_version")
        chunking_version = manifest.get("chunking_version")
        if documents and (
            schema_version != INDEX_SCHEMA_VERSION or chunking_version != CHUNKING_VERSION
        ):
            raise RuntimeError(
                "Existing index uses an incompatible ingestion schema. "
                "Keep it for comparison and configure a new empty v3 database directory/collection."
            )
        return manifest

    def _save_manifest(self, manifest: dict[str, object]) -> None:
        self.settings.persist_directory.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.manifest_path)


def corpus_conflicts(paths: Iterable[Path]) -> dict[str, list[str]]:
    """Group possible duplicate/edition conflicts for a doctor report."""

    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        standard_id, _ = identify_standard(path.name, "")
        if standard_id:
            groups[standard_id].append(str(path))
    return {key: values for key, values in groups.items() if len(values) > 1}
