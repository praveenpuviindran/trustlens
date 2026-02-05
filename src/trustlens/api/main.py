from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from trustlens.db.engine import build_engine, ping_db
from trustlens.api.routes_runs import router as runs_router
from trustlens.api.routes_models import router as models_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(runs_router)
app.include_router(models_router)

@app.get("/health")
def health() -> dict:
    engine = build_engine()
    db = ping_db(engine)
    return {
        "status": "ok" if db.ok else "degraded",
        "db": {"ok": db.ok, "detail": db.detail},
    }
