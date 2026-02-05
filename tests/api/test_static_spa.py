"""Tests for static SPA serving."""

from fastapi.testclient import TestClient

from trustlens.api.main import app


def test_spa_root_and_api_health():
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["service"] == "trustlens"

    res = client.get("/api/health")
    assert res.status_code == 200
    assert "status" in res.json()
