from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StandardPolicy:
    standard_id: str
    active_edition_year: int | None
    approved_sources: tuple[str, ...]
    source_aliases: tuple[str, ...]
    approved_sha256: tuple[str, ...]
    official_source_url: str | None
    licence_basis: str | None
    amendments: tuple[str, ...]
    supersedes: str | None
    reviewed_by: str | None
    reviewed_on: str | None
    notes: str

    @property
    def verified(self) -> bool:
        return bool(
            self.active_edition_year
            and self.approved_sources
            and self.approved_sha256
            and self.official_source_url
            and self.licence_basis
            and self.reviewed_by
            and self.reviewed_on
        )


class StandardsRegistry:
    def __init__(self, policies: dict[str, StandardPolicy], path: Path):
        self.policies = policies
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "StandardsRegistry":
        if not path.exists():
            return cls({}, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        policies: dict[str, StandardPolicy] = {}
        for standard_id, raw in payload.get("standards", {}).items():
            policies[standard_id.upper()] = StandardPolicy(
                standard_id=standard_id,
                active_edition_year=raw.get("active_edition_year"),
                approved_sources=tuple(raw.get("approved_sources", [])),
                source_aliases=tuple(raw.get("source_aliases", [])),
                approved_sha256=tuple(
                    str(item).casefold() for item in raw.get("approved_sha256", [])
                ),
                official_source_url=raw.get("official_source_url"),
                licence_basis=raw.get("licence_basis"),
                amendments=tuple(raw.get("amendments", [])),
                supersedes=raw.get("supersedes"),
                reviewed_by=raw.get("reviewed_by"),
                reviewed_on=raw.get("reviewed_on"),
                notes=str(raw.get("notes", "")),
            )
        return cls(policies, path)

    def get(self, standard_id: str | None) -> StandardPolicy | None:
        return self.policies.get((standard_id or "").upper())

    def resolve_document(
        self, source: str, document_sha256: str | None
    ) -> tuple[StandardPolicy | None, str | None]:
        """Resolve identity only from explicit registry facts.

        A hash is strongest. Exact approved filenames and explicit aliases are
        useful for identity, but they do not make an incomplete policy
        reviewer-verified or audit-ready.
        """

        digest = (document_sha256 or "").casefold()
        source_folded = Path(source).name.casefold()
        if digest:
            matches = [
                policy for policy in self.policies.values() if digest in policy.approved_sha256
            ]
            if len(matches) == 1:
                return matches[0], "registry_sha256"
        approved_matches = [
            policy
            for policy in self.policies.values()
            if source_folded in {Path(item).name.casefold() for item in policy.approved_sources}
        ]
        if len(approved_matches) == 1:
            return approved_matches[0], "registry_approved_source"
        alias_matches = [
            policy
            for policy in self.policies.values()
            if source_folded in {Path(item).name.casefold() for item in policy.source_aliases}
        ]
        if len(alias_matches) == 1:
            return alias_matches[0], "registry_source_alias"
        return None, None

    def source_is_approved(
        self, policy: StandardPolicy, source: str, document_sha256: str | None
    ) -> bool:
        source_folded = source.casefold()
        source_matches = any(item.casefold() == source_folded for item in policy.approved_sources)
        hash_matches = bool(
            document_sha256 and document_sha256.casefold() in policy.approved_sha256
        )
        return source_matches and hash_matches
