# Validation scope and interpretation

## What was tested

Two complementary controlled evaluations were run against the working deployment:

1. **Source-level retrieval:** seven distinct cases derived from four manually checked standard requirements and their applicability contexts. This set is used for Recall@5, Recall@10 and mean reciprocal rank.
2. **End-to-end decision logic:** 100 synthetic cases in the same logical input format as the DL/CV measurements, with deliberately selected compliant, non-compliant, boundary and missing-context values. Expected outcomes were known before the pipeline was run.

The structured-evidence upgrade moved the OCR-resistant W-beam requirement from missing in raw retrieval to rank 5. Aggregate retrieval performance changed from Recall@5/10 of 0.8571/0.8571 and MRR of 0.8571 to 1.0/1.0 and 0.8857.

The 100 controlled cases achieved 1.0 agreement for citation, standard, edition, value, comparator, applicability and end-to-end audit decision. These cases validate software behavior against known inputs; they do not measure errors made by the upstream camera, detector, depth estimator or GPS sensors.

## What the results do not establish

- The seven retrieval cases are too small to establish broad performance across all IRC clauses.
- The synthetic cases are not independently collected or expert-reviewed field observations.
- OSM context was independently checked on only one NH161 corridor.
- Standards provenance remains provisional until a qualified reviewer confirms source, edition, amendments, supersession and applicability.
- Safe/Low/Medium/High is a disclosed rule-based screening classification, not an auditor-approved engineering risk or crash-prediction model.
- Audit-mode latency is approximately 8–11 seconds per metric because it favors exhaustive retrieval and semantic reranking over interactive speed.

## Reporting rule

Always report the case count and controlled nature of the benchmark with each metric. Do not describe these results as “100% real-world accuracy,” “nationwide accuracy,” “certified IRC compliance,” or “auditor-approved risk prediction.”

The next meaningful validation step is an independently reviewed set of diverse field cases. Until that exists, the project should be presented as an evidence-bound research and decision-support system.
