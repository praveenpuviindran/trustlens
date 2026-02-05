from fastapi.testclient import TestClient

from trustlens.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

    payload = r.json()
    assert payload["status"] == "healthy"
