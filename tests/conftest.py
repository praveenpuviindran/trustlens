"""Global test fixtures."""

import pytest

from trustlens.api.main import app
from trustlens.api.rate_limit import RateLimiter


@pytest.fixture(autouse=True)
def _relax_rate_limit():
    # Ensure tests are not impacted by rate limiting unless explicitly set.
    app.state.rate_limiter = RateLimiter(limit_per_min=10_000, time_fn=lambda: 0)
    yield
