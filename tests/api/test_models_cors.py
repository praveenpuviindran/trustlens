"""API tests for /models and CORS."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.api.main import app
from trustlens.api.deps import get_db
from trustlens.db.schema import Base, TrainedModel


def _setup_db(tmp_path):
    db_path = tmp_path / "models_api.duckdb"
    engine = create_engine(f"duckdb:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def test_models_endpoint_and_cors(tmp_path):
    engine, Session = _setup_db(tmp_path)
    session = Session()
    session.add(
        TrainedModel(
            model_id="lr_v1",
            feature_schema_version="v1",
            feature_names_json='["total_articles"]',
            weights_json='{"total_articles": 1.0, "intercept": 0.0}',
            thresholds_json='{"t_lo": 0.2, "t_hi": 0.8}',
            metrics_json='{"accuracy": 1.0}',
            dataset_name="test",
            dataset_hash="hash",
        )
    )
    session.commit()
    session.close()

    def _db_override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db_override
    client = TestClient(app)

    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "baseline_v1" in data["models"]
    assert "lr_v1" in data["models"]

    # CORS preflight
    res = client.options(
        "/models",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.status_code in {200, 204}
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"

    app.dependency_overrides.clear()
