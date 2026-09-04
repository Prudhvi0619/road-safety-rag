from road_safety_rag.evaluation import summarize_case_results


def test_evaluation_summary_reports_interview_metrics():
    records = [
        {
            "metric_key": "min_lane_width",
            "retrieval_hit_at_5": True,
            "retrieval_hit_at_10": True,
            "reciprocal_rank": 1.0,
            "total_latency_ms": 100.0,
            "citation_match": True,
            "standard_match": True,
            "edition_match": True,
            "value_match": True,
            "comparator_match": True,
            "applicability_match": True,
            "audit_decision_match": True,
            "overall_correct": True,
            "expected_abstention": False,
            "predicted_abstention": False,
        },
        {
            "metric_key": "min_lane_width",
            "retrieval_hit_at_5": False,
            "retrieval_hit_at_10": True,
            "reciprocal_rank": 0.1,
            "total_latency_ms": 200.0,
            "citation_match": False,
            "standard_match": True,
            "edition_match": False,
            "value_match": False,
            "comparator_match": True,
            "applicability_match": True,
            "audit_decision_match": False,
            "overall_correct": False,
            "expected_abstention": True,
            "predicted_abstention": True,
        },
    ]
    summary = summarize_case_results(records, with_llm=True)
    assert summary["recall_at_5"] == 0.5
    assert summary["recall_at_10"] == 1.0
    assert summary["mean_reciprocal_rank"] == 0.55
    assert summary["standard_accuracy"] == 1.0
    assert summary["citation_accuracy"] == 0.5
    assert summary["abstention_precision"] == 1.0
    assert summary["abstention_recall"] == 1.0
    assert summary["latency_ms_by_metric"]["min_lane_width"]["p95"] == 200.0
