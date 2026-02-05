"""Tests for stratified evaluation utilities."""

from trustlens.services.datasets_loader import DatasetRow, stratified_split
from trustlens.services.stratified_eval import compute_bucket_metrics
from trustlens.services.evaluation import EvalRow


def test_stratified_split_deterministic():
    rows = [DatasetRow(str(i), f"c{i}", 1 if i % 2 == 0 else 0) for i in range(100)]
    t1, v1, s1 = stratified_split(rows, seed=123)
    t2, v2, s2 = stratified_split(rows, seed=123)
    assert [r.claim_id for r in t1] == [r.claim_id for r in t2]
    assert [r.claim_id for r in v1] == [r.claim_id for r in v2]
    assert [r.claim_id for r in s1] == [r.claim_id for r in s2]


def test_bucket_metrics():
    eval_rows = [
        EvalRow(run_id="r1", dataset_name="d", claim_id="1", label=1, score=0.9, predicted_label="credible"),
        EvalRow(run_id="r2", dataset_name="d", claim_id="2", label=0, score=0.1, predicted_label="not_credible"),
    ]
    feature_map = {
        "r1": {"total_articles": 0, "unknown_source_ratio": 0.1, "max_domain_concentration": 0.2},
        "r2": {"total_articles": 5, "unknown_source_ratio": 0.6, "max_domain_concentration": 0.8},
    }
    buckets = compute_bucket_metrics(eval_rows, feature_map)
    assert "total_articles" in buckets
    assert "0" in buckets["total_articles"]
    assert ">10" in buckets["total_articles"] or "4-10" in buckets["total_articles"]
