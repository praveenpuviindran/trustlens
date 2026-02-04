"""Integration test for trained model scoring."""

import csv
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, Feature, Run
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.model_training import ModelTrainer


def test_train_and_score_with_lr(tmp_path: Path):
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    run_pos = Run(claim_text="pos", query_text="q1", status="completed")
    run_neg = Run(claim_text="neg", query_text="q2", status="completed")
    session.add_all([run_pos, run_neg])
    session.flush()

    session.add_all(
        [
            Feature(run_id=run_pos.run_id, feature_group="volume", feature_name="total_articles", feature_value=5.0),
            Feature(run_id=run_neg.run_id, feature_group="volume", feature_name="total_articles", feature_value=0.0),
        ]
    )
    session.commit()

    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "label"])
        writer.writeheader()
        writer.writerow({"run_id": run_pos.run_id, "label": 1})
        writer.writerow({"run_id": run_neg.run_id, "label": 0})

    trainer = ModelTrainer(session)
    trainer.train_and_register(csv_path, dataset_name="tiny", model_id="lr_v1", split_ratio=0.5)

    service = FeatureEngineeringService(session)
    result = service.compute_score_for_run(run_pos.run_id, model_version="lr_v1")
    assert result.label in {"credible", "uncertain", "not_credible"}
