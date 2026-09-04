# Evaluation protocol

This repository does not publish fabricated accuracy numbers. The current status is recorded in `results/status.json`.

An independent transportation-domain reviewer should prepare 50–100 cases using `gold_cases.schema.json`. Each `found` case must identify the exact expected passage using an evidence ID, content hash, or verbatim quote in addition to standard/edition/page metadata. This prevents an unrelated chunk from the correct standard from being counted as a retrieval hit. Cases also record applicability, threshold value/comparator, abstention outcome, optional observed measurements, and the expected audit decision.

Run retrieval-only evaluation:

```text
road-safety-rag evaluate path/to/reviewed_gold.json --output evaluation/results/retrieval.json
```

Run end-to-end extraction and decision evaluation:

```text
road-safety-rag evaluate path/to/reviewed_gold.json --with-llm --output evaluation/results/end_to_end.json
```

The evaluator reports Recall@5, Recall@10, mean reciprocal rank, citation, standard, edition, value, comparator and applicability accuracy, abstention precision/recall, end-to-end audit-decision accuracy, and mean/median/p95 latency per metric.

Only reviewed, versioned result JSON files should replace the pending status file.
