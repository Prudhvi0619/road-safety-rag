from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    corpus_dirs: tuple[Path, ...]
    persist_directory: Path
    collection_name: str = "road_safety_codes_v3"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    allow_model_download: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    enable_reranker: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_s: int = 180
    ollama_context: int = 16384
    chunk_size: int = 1400
    chunk_overlap: int = 220
    dense_k: int = 28
    lexical_k: int = 28
    final_k: int = 10
    exhaustive_retrieval: bool = False
    neighbor_window: int = 1
    standards_registry: Path | None = None
    require_verified_standards: bool = True

    @classmethod
    def from_env(cls, project_dir: str | Path | None = None) -> "Settings":
        base = (
            Path(project_dir).expanduser().resolve()
            if project_dir is not None
            else _discover_project_dir(Path(__file__).resolve())
        )
        try:
            from dotenv import load_dotenv

            load_dotenv(base / ".env", override=False)
        except ImportError:
            # The CLI doctor reports this dependency explicitly. Keeping this
            # fallback allows pure unit tests to run in minimal environments.
            pass
        corpus_value = os.getenv("ROAD_RAG_CORPUS_DIRS")
        if corpus_value:
            corpus_dirs = tuple(
                Path(item).expanduser().resolve() for item in corpus_value.split(os.pathsep) if item
            )
        else:
            parent = base.parent
            corpus_dirs = (parent / "code books", parent / "scanned_books")

        persist = (
            Path(os.getenv("ROAD_RAG_DB_DIR", str(base.parent / "chroma_db_v3")))
            .expanduser()
            .resolve()
        )

        registry_value = os.getenv("ROAD_RAG_STANDARDS_REGISTRY")
        registry_path = (
            Path(registry_value).expanduser().resolve()
            if registry_value
            else (base / "config" / "standards_registry.json").resolve()
        )

        return cls(
            project_dir=base,
            corpus_dirs=corpus_dirs,
            persist_directory=persist,
            collection_name=os.getenv("ROAD_RAG_COLLECTION", "road_safety_codes_v3"),
            embedding_model=os.getenv("ROAD_RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            allow_model_download=_env_bool("ROAD_RAG_ALLOW_MODEL_DOWNLOAD", False),
            reranker_model=os.getenv("ROAD_RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            enable_reranker=_env_bool("ROAD_RAG_ENABLE_RERANKER", False),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.getenv("ROAD_RAG_LLM_MODEL", "llama3.1"),
            ollama_timeout_s=int(os.getenv("ROAD_RAG_LLM_TIMEOUT", "180")),
            ollama_context=int(os.getenv("ROAD_RAG_NUM_CTX", "16384")),
            chunk_size=int(os.getenv("ROAD_RAG_CHUNK_SIZE", "1400")),
            chunk_overlap=int(os.getenv("ROAD_RAG_CHUNK_OVERLAP", "220")),
            dense_k=int(os.getenv("ROAD_RAG_DENSE_K", "28")),
            lexical_k=int(os.getenv("ROAD_RAG_LEXICAL_K", "28")),
            final_k=int(os.getenv("ROAD_RAG_FINAL_K", "10")),
            exhaustive_retrieval=_env_bool("ROAD_RAG_EXHAUSTIVE_RETRIEVAL", False),
            neighbor_window=int(os.getenv("ROAD_RAG_NEIGHBOR_WINDOW", "1")),
            standards_registry=registry_path,
            require_verified_standards=_env_bool("ROAD_RAG_REQUIRE_VERIFIED_STANDARDS", True),
        )


def _discover_project_dir(module_path: Path) -> Path:
    """Find the repository root for both flat and ``src/`` installations."""

    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return module_path.parents[1]
