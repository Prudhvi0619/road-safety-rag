import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ".example"}


def test_repository_has_no_private_runtime_artifacts():
    forbidden = {
        ".venv",
        "chroma_db",
        "chroma_db_v2",
        "chroma_db_v3",
        ".rag_cache",
        "local_data",
    }
    assert not any(path.name in forbidden for path in ROOT.rglob("*"))
    assert not list(ROOT.rglob("*.pdf"))


def test_repository_has_no_real_generated_reports():
    forbidden_root_files = {
        "audit.xlsx",
        "audit_manual.xlsx",
        "map.html",
        "map_manual.html",
        "y",
    }
    assert not any((ROOT / name).exists() for name in forbidden_root_files)


def test_repository_has_no_personal_absolute_paths():
    windows_absolute = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)")
    unix_home = re.compile(r"/(?:Users|home)/[^/\s]+/")
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if windows_absolute.search(text) or unix_home.search(text):
            findings.append(str(path.relative_to(ROOT)))
    assert findings == []
