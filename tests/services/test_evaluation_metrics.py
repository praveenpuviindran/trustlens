"""Unit tests for evaluation metrics."""

from trustlens.services.evaluation import EvalRow, compute_metrics


def test_compute_metrics_basic():
    rows = [
        EvalRow(run_id="r1", dataset_name="d", claim_id="1", label=1, score=0.9, predicted_label="credible"),
        EvalRow(run_id="r2", dataset_name="d", claim_id="2", label=0, score=0.2, predicted_label="not_credible"),
        EvalRow(run_id="r3", dataset_name="d", claim_id="3", label=1, score=0.4, predicted_label="uncertain"),
        EvalRow(run_id="r4", dataset_name="d", claim_id="4", label=0, score=0.6, predicted_label="credible"),
    ]

    metrics = compute_metrics(rows)
    assert metrics["n"] == 4
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert len(metrics["calibration_bins"]) == 5
