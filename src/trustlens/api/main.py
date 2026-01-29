from __future__ import annotations

from fastapi import FastAPI

from trustlens.db.engine import build_engine, ping_db

app = FastAPI(title="TrustLens", version="0.1.0")


@app.get("/health")
def health() -> dict:
    engine = build_engine()
    db = ping_db(engine)
    return {
        "status": "ok" if db.ok else "degraded",
        "db": {"ok": db.ok, "detail": db.detail},
    }
