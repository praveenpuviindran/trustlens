"""CORS env var behavior."""

import os
from fastapi.testclient import TestClient

from trustlens.api import main as main_mod


def test_cors_env_allows_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com")
    # reload app module to pick up env
    import importlib
    importlib.reload(main_mod)
    client = TestClient(main_mod.app)
    res = client.options(
        "/api/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "https://example.com"

    # restore for other tests
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
