"""Smoke test for deployed TrustLens API."""

from __future__ import annotations

import os
import sys
import httpx


def main() -> int:
    base = os.getenv("TRUSTLENS_BASE_URL", "http://localhost:8000")
    health = httpx.get(f"{base}/api/health", timeout=10)
    if health.status_code != 200:
        print("Health failed:", health.status_code, health.text)
        return 1

    payload = {
        "claim_text": "Test claim about public policy.",
        "query_text": "public policy test",
        "max_records": 3,
        "model_id": "baseline_v1",
        "include_explanation": True,
    }
    res = httpx.post(f"{base}/api/runs", json=payload, timeout=30)
    if res.status_code != 200:
        print("Run failed:", res.status_code, res.text)
        return 1
    data = res.json()
    print("run_id:", data.get("run_id"))
    print("label:", data.get("label"))
    print("score:", data.get("score"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
