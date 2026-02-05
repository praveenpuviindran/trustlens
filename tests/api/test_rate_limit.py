"""Tests for rate limiting middleware."""

from fastapi.testclient import TestClient

from trustlens.api.main import app
from trustlens.api.rate_limit import RateLimiter


def test_rate_limit_enforced():
    limiter = RateLimiter(limit_per_min=1, time_fn=lambda: 1000)
    app.state.rate_limiter = limiter
    client = TestClient(app)

    res1 = client.get("/api/health")
    res2 = client.get("/api/health")
    assert res1.status_code == 200
    assert res2.status_code == 429
