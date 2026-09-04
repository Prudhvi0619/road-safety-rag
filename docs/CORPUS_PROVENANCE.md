# Standards corpus provenance

Standards PDFs, OCR exports, embeddings, and the vector database are intentionally excluded from this repository. Standards may be copyrighted and must be obtained through an authorised institutional or publisher channel.

Before a corpus is used for a demonstration or audit:

1. Record each document in a private manifest following `config/corpus_manifest.schema.json`.
2. Record the standard identifier, active edition, amendments, supersession chain, official source URL, licence basis, reviewer, and review date.
3. Add the reviewer-approved filename and full-document SHA-256 to `config/standards_registry.json`, together with the authoritative source URL, licence basis, amendment/supersession record, reviewer and date. Validate it against `config/standards_registry.schema.json`.
4. Rebuild the index and retain the ingestion manifest.
5. Re-run the reviewed gold evaluation and archive its versioned results.

A SHA-256 value demonstrates that the indexed file did not change. It does not establish that a document is official, licensed, current, or applicable.

The supplied `standards_registry.json` remains deliberately unverified until a qualified domain owner completes it. A filename from an unofficial mirror must not be accepted as authoritative evidence.
