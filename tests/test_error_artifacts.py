"""Tests for error artifact generation."""

from pathlib import Path

from trustlens.services.stratified_eval import write_error_artifacts
from trustlens.services.evaluation import EvalRow


def test_error_artifacts_files(tmp_path: Path):
    eval_rows = [
        EvalRow(run_id="r1", dataset_name="d", claim_id="1", label=0, score=0.9, predicted_label="credible"),
        EvalRow(run_id="r2", dataset_name="d", claim_id="2", label=1, score=0.1, predicted_label="not_credible"),
    ]
    feature_map = {
        "r1": {"weighted_prior_mean": 0.2, "unknown_source_ratio": 0.9, "max_domain_concentration": 0.8},
        "r2": {"weighted_prior_mean": 0.8, "unknown_source_ratio": 0.1, "max_domain_concentration": 0.2},
    }
    claim_map = {"r1": "c1", "r2": "c2"}
    out = write_error_artifacts(eval_rows, feature_map, claim_map, tmp_path)
    assert (tmp_path / "false_positives.csv").exists()
    assert (tmp_path / "false_negatives.csv").exists()
    assert (tmp_path / "hard_cases.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert out["false_positives"] == 1
