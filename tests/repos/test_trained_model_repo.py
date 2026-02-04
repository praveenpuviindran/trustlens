"""Tests for trained model repository."""

import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, TrainedModel
from trustlens.repos.trained_model_repo import TrainedModelRepository


def test_upsert_get_list_models():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    repo = TrainedModelRepository(session)
    model = TrainedModel(
        model_id="lr_v1",
        feature_schema_version="v1",
        feature_names_json=json.dumps(["a", "b"]),
        weights_json=json.dumps({"a": 0.1, "b": -0.2, "intercept": 0.0}),
        thresholds_json=json.dumps({"t_lo": 0.3, "t_hi": 0.7}),
        metrics_json=json.dumps({"accuracy": 1.0}),
        dataset_name="test",
        dataset_hash="abc",
    )
    repo.upsert(model)

    fetched = repo.get("lr_v1")
    assert fetched is not None
    assert fetched.model_id == "lr_v1"

    models = repo.list_models()
    assert len(models) == 1
