"""Integration test for lr_v2 training and scoring with text features."""

import csv
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, EvidenceItem, Run
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.model_training import ModelTrainer


def test_train_and_score_lr_v2(tmp_path: Path):
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    runs = [
        Run(claim_text="NASA Launch", query_text="nasa launch", status="completed"),
        Run(claim_text="Mars Mission", query_text="mars mission", status="completed"),
        Run(claim_text="Aliens landed", query_text="aliens landed", status="completed"),
        Run(claim_text="Cure found", query_text="cure found", status="completed"),
    ]
    session.add_all(runs)
    session.flush()

    evidence_items = [
        EvidenceItem(
            run_id=runs[0].run_id,
            url="https://example.com/p1",
            domain="example.com",
            title="NASA Launch Successful",
            snippet="Official report confirms launch",
            source="test",
        ),
        EvidenceItem(
            run_id=runs[1].run_id,
            url="https://example.com/p2",
            domain="example.com",
            title="Mars Mission Announced",
            snippet="Space agency confirms mission",
            source="test",
        ),
        EvidenceItem(
            run_id=runs[2].run_id,
            url="https://example.com/n1",
            domain="example.com",
            title="Aliens landed hoax",
            snippet="Debunked by officials",
            source="test",
        ),
        EvidenceItem(
            run_id=runs[3].run_id,
            url="https://example.com/n2",
            domain="example.com",
            title="Cure found not true",
            snippet="False claim reported",
            source="test",
        ),
    ]
    session.add_all(evidence_items)
    session.commit()

    feature_service = FeatureEngineeringService(session)
    for run in runs:
        feature_service.compute_features(run.run_id)

    csv_path = tmp_path / "train_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "label"])
        writer.writeheader()
        writer.writerow({"run_id": runs[0].run_id, "label": 1})
        writer.writerow({"run_id": runs[1].run_id, "label": 1})
        writer.writerow({"run_id": runs[2].run_id, "label": 0})
        writer.writerow({"run_id": runs[3].run_id, "label": 0})

    trainer = ModelTrainer(session)
    trainer.train_and_register(
        csv_path,
        dataset_name="tiny_v2",
        model_id="lr_v2",
        split_ratio=0.75,
        feature_schema_version="v2",
    )

    pos_result = feature_service.compute_score_for_run(runs[0].run_id, model_version="lr_v2")
    neg_result = feature_service.compute_score_for_run(runs[2].run_id, model_version="lr_v2")

    assert pos_result.score > neg_result.score
    assert pos_result.label in {"credible", "uncertain"}
    assert neg_result.label in {"not_credible", "uncertain"}
