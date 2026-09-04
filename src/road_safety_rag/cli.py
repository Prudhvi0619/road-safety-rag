from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .audit import RoadSafetyAuditPipeline
from .catalog import METRICS, get_metric
from .config import Settings
from .evaluation import evaluate_gold
from .ingestion import CHUNKING_VERSION, INDEX_SCHEMA_VERSION, IndexBuilder, corpus_conflicts
from .models import RoadContext
from .ollama_client import OllamaClient, OllamaUnavailable
from .registry import StandardsRegistry
from .retrieval import HybridRetriever
from .service import StandardsRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="road-safety-rag",
        description="Evidence-first offline RAG for IRC/Indian road-safety audits.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "wizard",
        help="Guided PDF ingestion followed by an Excel road-safety audit.",
    )

    subparsers.add_parser(
        "doctor", help="Check corpus, index, dependencies, duplicates, and Ollama."
    )
    index = subparsers.add_parser("index", help="Build/update the table-aware v3 vector index.")
    index.add_argument("--ocr", action="store_true", help="Use Docling OCR to recover weak pages.")
    index.add_argument("--force", action="store_true", help="Re-index every PDF.")
    index.add_argument(
        "--prune",
        action="store_true",
        help="Remove indexed sources that no longer exist under the configured corpus folders.",
    )
    index.add_argument(
        "--folder",
        action="append",
        type=Path,
        help="Ingest PDFs from this folder into the existing vector database (repeatable).",
    )
    subparsers.add_parser(
        "repair-metadata",
        help="Correct standard IDs from indexed cover pages without re-embedding.",
    )

    query = subparsers.add_parser("query", help="Extract verified standard thresholds.")
    _add_context_arguments(query)
    query.add_argument("--metric", action="append", choices=sorted(METRICS))
    query.add_argument("--no-cache", action="store_true")
    _add_deep_retrieval_argument(query)

    retrieve = subparsers.add_parser(
        "retrieve", help="Inspect ranked evidence without calling Ollama."
    )
    _add_context_arguments(retrieve)
    retrieve.add_argument("--metric", required=True, choices=sorted(METRICS))
    _add_deep_retrieval_argument(retrieve)

    audit = subparsers.add_parser("audit", help="Run the combined measurement + standards audit.")
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", type=Path, default=Path("Final_Road_Safety_Audit_Robust.xlsx"))
    audit.add_argument(
        "--html-output", type=Path, help="Also write an interactive severity-coded HTML map."
    )
    audit.add_argument(
        "--lookup-road", action="store_true", help="Use Overpass to infer road context."
    )
    _add_deep_retrieval_argument(audit)
    _add_context_arguments(audit)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Measure retrieval, evidence, applicability, abstention, decision, and latency quality.",
    )
    evaluate.add_argument("gold", type=Path)
    evaluate.add_argument("--with-llm", action="store_true")
    evaluate.add_argument("--output", type=Path, default=Path("rag_evaluation_results.json"))
    return parser


def _add_deep_retrieval_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("fast", "balanced", "audit"),
        help=(
            "fast: hybrid retrieval; balanced: hybrid + cross-encoder; "
            "audit: exhaustive retrieval + cross-encoder."
        ),
    )
    parser.add_argument(
        "--deep-retrieval",
        action="store_true",
        help=("Deprecated alias for --retrieval-mode audit."),
    )


def deep_retrieval_settings(settings: Settings, enabled: bool) -> Settings:
    return retrieval_mode_settings(settings, "audit" if enabled else "fast")


def retrieval_mode_settings(settings: Settings, mode: str) -> Settings:
    if mode == "fast":
        return replace(
            settings,
            enable_reranker=False,
            exhaustive_retrieval=False,
            neighbor_window=max(1, settings.neighbor_window),
        )
    if mode == "balanced":
        return replace(
            settings,
            dense_k=max(settings.dense_k, 40),
            lexical_k=max(settings.lexical_k, 50),
            final_k=max(settings.final_k, 10),
            exhaustive_retrieval=False,
            neighbor_window=max(settings.neighbor_window, 1),
            enable_reranker=True,
        )
    if mode == "audit":
        return replace(
            settings,
            dense_k=max(settings.dense_k, 80),
            lexical_k=max(settings.lexical_k, 100),
            final_k=max(settings.final_k, 12),
            exhaustive_retrieval=True,
            neighbor_window=max(settings.neighbor_window, 2),
            enable_reranker=True,
        )
    raise ValueError(f"Unknown retrieval mode: {mode}")


def _settings_from_args(settings: Settings, args: argparse.Namespace, default: str) -> Settings:
    if args.deep_retrieval and args.retrieval_mode not in {None, "audit"}:
        raise ValueError("--deep-retrieval cannot be combined with a non-audit retrieval mode")
    mode = "audit" if args.deep_retrieval else (args.retrieval_mode or default)
    return retrieval_mode_settings(settings, mode)


def corpus_folder_settings(settings: Settings, folder: Path) -> Settings:
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"PDF folder does not exist or is not a directory: {resolved}")
    return replace(settings, corpus_dirs=(resolved,))


def _prompt_yes_no(question: str, input_fn: Callable[[str], str], default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input_fn(question + suffix).strip().casefold()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_choice(
    label: str,
    choices: tuple[str, ...],
    default: str,
    input_fn: Callable[[str], str],
) -> str:
    allowed = {choice.casefold(): choice for choice in choices}
    while True:
        answer = input_fn(f"{label} ({'/'.join(choices)}) [{default}]: ").strip()
        if not answer:
            return default
        selected = allowed.get(answer.casefold())
        if selected:
            return selected
        print("Choose one of: " + ", ".join(choices))


def _prompt_number(
    label: str,
    input_fn: Callable[[str], str],
    default: float | int | None = None,
    integer: bool = False,
    required: bool = False,
):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input_fn(f"{label}{suffix}: ").strip()
        if not answer and default is not None:
            return int(default) if integer else float(default)
        if not answer and not required:
            return None
        try:
            value = int(answer) if integer else float(answer)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Enter a positive number.")


def _prompt_existing_path(
    label: str,
    input_fn: Callable[[str], str],
    expect_directory: bool,
) -> Path:
    while True:
        raw = input_fn(label + ": ").strip().strip('"')
        candidate = Path(raw).expanduser().resolve() if raw else None
        valid = candidate and (candidate.is_dir() if expect_directory else candidate.is_file())
        if valid:
            return candidate
        expected = "folder" if expect_directory else "file"
        print(f"That {expected} does not exist. Paste its complete path.")


def run_wizard(
    settings: Settings,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    output_fn("Road Safety RAG guided audit")
    output_fn(f"Existing vector database: {settings.persist_directory}")
    if _prompt_yes_no("Do you want to ingest/update IRC PDF code books?", input_fn):
        folder = _prompt_existing_path(
            "Paste the folder path containing the IRC PDF code books",
            input_fn,
            expect_directory=True,
        )
        folder_settings = corpus_folder_settings(settings, folder)
        pdf_count = len(IndexBuilder(folder_settings).discover_pdfs())
        if not pdf_count:
            output_fn(f"No PDF files were found under {folder}.")
            return 2
        output_fn(
            f"Found {pdf_count} PDF file(s). OCR re-ingestion into the existing database has started."
        )
        summary = IndexBuilder(folder_settings).build(use_ocr=True, force=True, prune=True)
        repair = IndexBuilder(folder_settings).repair_standard_metadata()
        output_fn(
            "Ingestion complete: "
            f"{summary['indexed_documents']} document(s), {summary['chunks']} chunk(s), "
            f"{summary['pruned_documents']} stale document(s) pruned, "
            f"{len(summary['warnings'])} warning(s); "
            f"metadata repaired for {repair['documents_changed']} document(s)."
        )

    metrics_file = _prompt_existing_path(
        "Paste the DL/CV Excel input file path", input_fn, expect_directory=False
    )
    if metrics_file.suffix.casefold() not in {".xlsx", ".xls", ".csv"}:
        output_fn("Input must be an .xlsx, .xls, or .csv file.")
        return 2

    allow_gps_lookup = _prompt_yes_no(
        "Allow sending up to 12 sampled GPS points to OpenStreetMap and Open-Meteo "
        "to infer road class, setting, posted-speed proxy, carriageway, and terrain?",
        input_fn,
        default=True,
    )

    output_default = metrics_file.with_name(f"{metrics_file.stem}_RAG_Audit.xlsx")
    html_default = metrics_file.with_name(f"{metrics_file.stem}_RAG_Map.html")
    output_raw = input_fn(f"Excel report path [{output_default}]: ").strip().strip('"')
    html_raw = input_fn(f"HTML map path [{html_default}]: ").strip().strip('"')
    output_path = Path(output_raw).expanduser().resolve() if output_raw else output_default
    html_path = Path(html_raw).expanduser().resolve() if html_raw else html_default

    audit_settings = retrieval_mode_settings(settings, "audit")
    output_fn(
        "Automatic workbook/GPS context resolution and deep evidence retrieval started. "
        "This may take several minutes."
    )
    rag = StandardsRAG(audit_settings)
    output = RoadSafetyAuditPipeline(
        metrics_file,
        rag,
        road_context=None,
        allow_network_road_lookup=allow_gps_lookup,
    ).run(output_path, html_output=html_path)
    output_fn(f"Excel audit written to: {output}")
    output_fn(f"HTML map written to: {html_path}")
    output_fn(
        "Review the Applicable Standards sheet. NEEDS_CONTEXT, AMBIGUOUS, and "
        "INVALID_EVIDENCE are abstentions, not missing display values."
    )
    return 0


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--road-class")
    parser.add_argument("--road-class-confidence", type=float)
    parser.add_argument("--setting", choices=["urban", "rural", "unknown"], default="unknown")
    parser.add_argument(
        "--terrain",
        choices=["plain", "rolling", "mountainous", "steep", "unknown"],
        default="unknown",
    )
    parser.add_argument(
        "--carriageway", choices=["divided", "undivided", "one_way", "unknown"], default="unknown"
    )
    parser.add_argument(
        "--lanes",
        type=int,
        help="Verified total lanes across the complete road (both carriageways when divided).",
    )
    parser.add_argument("--lanes-per-carriageway", type=int)
    parser.add_argument("--carriageway-count", type=int)
    parser.add_argument(
        "--design-speed",
        type=float,
        help="Verified highway design speed in km/h; do not enter an observed/posted speed here.",
    )
    parser.add_argument("--sign-class")
    parser.add_argument("--sign-shape")
    parser.add_argument(
        "--sign-mounting", choices=["shoulder", "overhead", "median", "unknown"], default="unknown"
    )
    parser.add_argument(
        "--barrier-type",
        choices=["w_beam", "thrie_beam", "concrete", "wire_rope", "unknown"],
        default="unknown",
    )
    parser.add_argument("--kerb-type")


def context_from_args(args: argparse.Namespace) -> RoadContext:
    return RoadContext(
        road_class=args.road_class,
        road_class_source="user/CLI" if args.road_class else None,
        road_class_confidence=args.road_class_confidence,
        setting=args.setting,
        terrain=args.terrain,
        carriageway=args.carriageway,
        lanes_total=args.lanes,
        lanes_per_carriageway=args.lanes_per_carriageway,
        carriageway_count=args.carriageway_count,
        lane_count_source=(
            "user/CLI verified base road configuration" if args.lanes is not None else None
        ),
        design_speed_kmph=args.design_speed,
        design_speed_source="user/CLI verified design speed"
        if args.design_speed is not None
        else None,
        sign_class=args.sign_class,
        sign_shape=args.sign_shape,
        sign_mounting=args.sign_mounting,
        barrier_type=args.barrier_type,
        kerb_type=args.kerb_type,
    )


def doctor(settings: Settings) -> int:
    print("Road Safety RAG doctor")
    print(f"Project: {settings.project_dir}")
    print(f"v3 index: {settings.persist_directory}")
    required = [
        "pydantic",
        "dotenv",
        "pandas",
        "openpyxl",
        "pypdf",
        "chromadb",
        "sentence_transformers",
        "langchain_chroma",
        "langchain_huggingface",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print(f"Dependencies: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    builder = IndexBuilder(settings)
    pdfs = builder.discover_pdfs()
    print(
        f"PDF corpus: {len(pdfs)} file(s) across {len(settings.corpus_dirs)} configured folder(s)"
    )
    conflicts = corpus_conflicts(pdfs)
    if conflicts:
        print("Possible edition/duplicate conflicts:")
        for standard_id, paths in sorted(conflicts.items()):
            print(f"  {standard_id}: {len(paths)} files")
            for path in paths:
                print(f"    - {path}")
    manifest = settings.persist_directory / "index_manifest.json"
    print(f"Index manifest: {'present' if manifest.exists() else 'not built'}")
    unknown_documents: list[tuple[str, dict[str, object]]] = []
    if manifest.exists():
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        documents = manifest_data.get("documents", {})
        failed_documents = manifest_data.get("failed_documents", {})
        chunks = sum(int(item.get("chunks", 0)) for item in documents.values())
        low_pages = sum(len(item.get("low_text_pages", [])) for item in documents.values())
        unassessed = sum("low_text_pages" not in item for item in documents.values())
        print(
            f"Index contents: {len(documents)} document(s), {chunks} chunk(s), "
            f"{low_pages} low-text page(s), {len(failed_documents)} failed document(s), "
            f"{unassessed} legacy/unassessed manifest record(s)"
        )
        print(
            "Index schema: "
            f"v{manifest_data.get('schema_version', 'unknown')}; "
            f"chunking={manifest_data.get('chunking_version', 'legacy')}"
        )
        unknown_documents = [
            (source, item) for source, item in documents.items() if not item.get("standard_id")
        ]
        if unknown_documents:
            print("Unknown/quarantined standard identities:")
            for source, item in unknown_documents:
                candidate = item.get("candidate_standard_id")
                score = item.get("candidate_score")
                suggestion = f"; candidate={candidate} ({score})" if candidate else ""
                print(f"  - {Path(source).name}{suggestion}")
        stale_sources = [source for source in documents if not Path(source).exists()]
        if stale_sources:
            print(
                f"Stale index sources: {len(stale_sources)} missing file(s); "
                "run index --prune with the original corpus folder(s)."
            )
    registry_path = (
        settings.standards_registry or settings.project_dir / "config" / "standards_registry.json"
    )
    registry = StandardsRegistry.load(registry_path)
    verified_policies = sum(policy.verified for policy in registry.policies.values())
    uncovered_metrics = [
        metric.key
        for metric in METRICS.values()
        if not any(
            (policy := registry.get(standard_id)) and policy.verified
            for standard_id in metric.preferred_standards
        )
    ]
    print(
        f"Standards registry: {registry_path} "
        f"({verified_policies}/{len(registry.policies)} policies reviewer-verified)"
    )
    print(
        "Audit source policy: "
        + (
            "STRICT (unverified standards cannot drive PASS/FAIL)"
            if settings.require_verified_standards
            else "PROVISIONAL SCREENING (latest indexed evidence may drive labelled decisions)"
        )
    )
    reranker_ready = False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(settings.reranker_model, local_files_only=True)
        print(f"Audit reranker '{settings.reranker_model}': cached and ready")
        reranker_ready = True
    except Exception as exc:
        print(
            f"Audit reranker '{settings.reranker_model}': NOT READY ({exc}). "
            "Temporarily allow model download before the first audit run."
        )
    ollama_ready = False
    try:
        client = OllamaClient(settings)
        models = client.models()
        print(f"Ollama: reachable; models: {', '.join(models) or 'none'}")
        client.check_ready()
        print(f"Configured model '{settings.ollama_model}': ready")
        ollama_ready = True
    except OllamaUnavailable as exc:
        print(f"Ollama: NOT READY ({exc})")
    blockers: list[str] = []
    if missing:
        blockers.append("missing dependencies")
    if not manifest.exists():
        blockers.append("index manifest is missing")
    elif (
        manifest_data.get("schema_version") != INDEX_SCHEMA_VERSION
        or manifest_data.get("chunking_version") != CHUNKING_VERSION
    ):
        blockers.append("index schema/chunking is not v3 table-aware")
    if settings.require_verified_standards and unknown_documents:
        blockers.append(
            f"{len(unknown_documents)} indexed document(s) have unknown standard identity"
        )
    if settings.require_verified_standards and verified_policies == 0:
        blockers.append("strict mode has no reviewer-verified standards")
    elif settings.require_verified_standards and uncovered_metrics:
        blockers.append("strict mode has no verified source for: " + ", ".join(uncovered_metrics))
    if not ollama_ready:
        blockers.append("Ollama/model is not ready")
    if not reranker_ready:
        blockers.append("audit-mode reranker is not cached")
    if blockers:
        print("Doctor result: NOT AUDIT READY (" + "; ".join(blockers) + ")")
        return 1
    if settings.require_verified_standards:
        print("Doctor result: AUDIT READY")
    else:
        print("Doctor result: READY FOR PROVISIONAL SCREENING (not a statutory audit)")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command in {None, "wizard"}:
        return run_wizard(settings)
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "index":
        index_settings = settings
        if args.folder:
            folders = tuple(path.expanduser().resolve() for path in args.folder)
            missing = [str(path) for path in folders if not path.is_dir()]
            if missing:
                print("Invalid PDF folder(s): " + ", ".join(missing), file=sys.stderr)
                return 2
            index_settings = replace(settings, corpus_dirs=folders)
        summary = IndexBuilder(index_settings).build(
            use_ocr=args.ocr,
            force=args.force,
            prune=args.prune,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "repair-metadata":
        summary = IndexBuilder(settings).repair_standard_metadata()
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "query":
        rag = StandardsRAG(_settings_from_args(settings, args, default="fast"))
        result = rag.extract_all(
            context_from_args(args), metric_keys=args.metric, use_cache=not args.no_cache
        )
        print(result.model_dump_json(indent=2))
        return 0
    if args.command == "retrieve":
        hits = HybridRetriever(_settings_from_args(settings, args, default="fast")).retrieve(
            get_metric(args.metric), context_from_args(args)
        )
        payload = [
            {
                "rank": rank,
                "evidence_id": hit.evidence_id,
                "standard_id": hit.standard_id,
                "edition_year": hit.edition_year,
                "source": hit.source,
                "page": hit.page,
                "section": hit.section,
                "score": hit.score,
                "text": " ".join(hit.text.split())[:700],
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "audit":
        rag = StandardsRAG(_settings_from_args(settings, args, default="audit"))
        output = RoadSafetyAuditPipeline(
            args.input,
            rag,
            road_context=context_from_args(args),
            allow_network_road_lookup=args.lookup_road,
        ).run(args.output, html_output=args.html_output)
        print(f"Audit written to {output}")
        if args.html_output:
            print(f"HTML report written to {args.html_output.resolve()}")
        return 0
    if args.command == "evaluate":
        result = evaluate_gold(settings, args.gold, with_llm=args.with_llm)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(f"Detailed evaluation written to {args.output.resolve()}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
