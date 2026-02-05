"""Tests for max_records cap enforcement."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.api.main import app
from trustlens.api.deps import get_db, get_evidence_fetcher
from trustlens.config import settings as settings_mod
from trustlens.db.schema import Base


def test_max_records_cap(tmp_path, monkeypatch):
    db_path = tmp_path / "cap_test.duckdb"
    engine = create_engine(f"duckdb:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def _db_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    captured = {}

    def _fetch_override():
        def _fetch(query_text: str, max_records: int):
            captured["max_records"] = max_records
            return []
        return _fetch

    monkeypatch.setattr(settings_mod, "settings", settings_mod.settings)
    settings_mod.settings.max_records_cap = 2

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[get_evidence_fetcher] = _fetch_override

    client = TestClient(app)
    res = client.post(
        "/api/runs",
        json={
            "claim_text": "Test claim",
            "query_text": "Test query",
            "max_records": 10,
            "model_id": "baseline_v1",
            "include_explanation": False,
        },
    )
    assert res.status_code == 200
    assert captured.get("max_records") == 2

    app.dependency_overrides.clear()
