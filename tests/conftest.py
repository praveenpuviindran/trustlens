"""Global test fixtures."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trustlens.api.main import app  # noqa: E402
from trustlens.api.rate_limit import RateLimiter  # noqa: E402


@pytest.fixture(autouse=True)
def _relax_rate_limit():
    # Ensure tests are not impacted by rate limiting unless explicitly set.
    app.state.rate_limiter = RateLimiter(limit_per_min=10_000, time_fn=lambda: 0)
    yield


@pytest.fixture(autouse=True)
def _reset_cors(monkeypatch):
    # Ensure default CORS env for tests
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    yield
