"""API tests for runs endpoints."""

import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.api.main import app
from trustlens.api.deps import get_db, get_evidence_fetcher, get_llm
from trustlens.db.schema import Base, TrainedModel
from trustlens.services.llm_client import StubLLMClient


def _setup_db(tmp_path):
    db_path = tmp_path / "runs_api.duckdb"
    engine = create_engine(f"duckdb:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def test_post_runs_baseline(tmp_path):
    engine, Session = _setup_db(tmp_path)

    def _db_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _fetch_override():
        def _fetch(query_text: str, max_records: int):
            return [
                {
                    "url": "https://example.com/a",
                    "domain": "example.com",
                    "title": "A",
                    "seendate": "20250101000000",
                    "raw": {},
                }
            ]
        return _fetch

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[get_evidence_fetcher] = _fetch_override
    app.dependency_overrides[get_llm] = lambda: StubLLMClient("EXPLAIN")

    client = TestClient(app)
    res = client.post(
        "/api/runs",
        json={
            "claim_text": "Test claim",
            "query_text": "Test query",
            "max_records": 1,
            "model_id": "baseline_v1",
            "include_explanation": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["evidence_count"] == 1
    assert data["explanation"]["summary"] == "EXPLAIN"

    app.dependency_overrides.clear()


def test_post_runs_trained_model(tmp_path):
    engine, Session = _setup_db(tmp_path)
    session = Session()
    session.add(
        TrainedModel(
            model_id="lr_v1",
            feature_schema_version="v1",
            feature_names_json=json.dumps(["total_articles"]),
            weights_json=json.dumps({"total_articles": 1.0, "intercept": 0.0}),
            thresholds_json=json.dumps({"t_lo": 0.2, "t_hi": 0.8}),
            metrics_json=json.dumps({"accuracy": 1.0}),
            dataset_name="test",
            dataset_hash="hash",
        )
    )
    session.commit()
    session.close()

    def _db_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _fetch_override():
        def _fetch(query_text: str, max_records: int):
            return [
                {
                    "url": "https://example.com/a",
                    "domain": "example.com",
                    "title": "A",
                    "seendate": "20250101000000",
                    "raw": {},
                },
                {
                    "url": "https://example.com/b",
                    "domain": "example.com",
                    "title": "B",
                    "seendate": "20250102000000",
                    "raw": {},
                },
            ]
        return _fetch

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[get_evidence_fetcher] = _fetch_override
    app.dependency_overrides[get_llm] = lambda: StubLLMClient("EXPLAIN")

    client = TestClient(app)
    res = client.post(
        "/api/runs",
        json={
            "claim_text": "Test claim",
            "query_text": "Test query",
            "max_records": 2,
            "model_id": "lr_v1",
            "include_explanation": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] in {"credible", "uncertain", "not_credible"}

    app.dependency_overrides.clear()
