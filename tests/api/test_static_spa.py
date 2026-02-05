"""Tests for static SPA serving."""

from fastapi.testclient import TestClient

from trustlens.api.main import app


def test_spa_root_and_api_health():
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")

    res = client.get("/api/health")
    assert res.status_code == 200
    assert "status" in res.json()
