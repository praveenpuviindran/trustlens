"""Offline benchmarking tests."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, TrainedModel
from trustlens.services.benchmarking import BenchmarkConfig, run_benchmark


def test_benchmark_no_fetch_offline(monkeypatch, tmp_path):
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    session.add(
        TrainedModel(
            model_id="lr_v1",
            feature_schema_version="v1",
            feature_names_json='["total_articles","unique_domains"]',
            weights_json='{"total_articles":1.0,"unique_domains":0.5,"intercept":0.0}',
            thresholds_json='{"t_lo":0.3,"t_hi":0.7}',
            metrics_json='{"accuracy":1.0}',
            dataset_name="test",
            dataset_hash="hash",
        )
    )
    session.commit()

    config = BenchmarkConfig(
        dataset_name="liar",
        dataset_split="test",
        max_examples=2,
        seed=42,
        model_ids=["baseline_v1", "lr_v1"],
        label_mapping_name="liar_mapping",
        max_records=1,
        no_fetch_evidence=True,
    )

    # Patch loader to avoid HF download
    from trustlens.services import datasets_loader
    from trustlens.services import benchmarking as bench_mod
    rows = [
        datasets_loader.DatasetRow(claim_id="1", claim_text="A", label=1),
        datasets_loader.DatasetRow(claim_id="2", claim_text="B", label=0),
    ]
    def fake_loader(*args, **kwargs):
        return rows, {"dataset_name":"liar","split":"test","max_examples":2,"seed":42,"dropped":0,"selected_indices":[0,1]}

    monkeypatch.setattr(bench_mod, "load_hf_dataset", fake_loader)
    report = run_benchmark(session, config, evidence_fetcher=None, output_dir=tmp_path)
    assert "metrics" in report
    assert "baseline_v1" in report["metrics"]
