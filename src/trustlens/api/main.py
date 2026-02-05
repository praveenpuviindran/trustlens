from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trustlens.db.engine import build_engine, ping_db
from trustlens.api.routes_runs import router as runs_router
from trustlens.api.routes_models import router as models_router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app = FastAPI()

# Local dev CORS (same-origin in production)
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

# API routes under /api
app.include_router(runs_router, prefix="/api")
app.include_router(models_router, prefix="/api")

# Static assets (Vite build output)
ASSETS_DIR = STATIC_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/api/health")
def health() -> dict:
    engine = build_engine()
    db = ping_db(engine)
    return {
        "status": "ok" if db.ok else "degraded",
        "db": {"ok": db.ok, "detail": db.detail},
    }


@app.get("/health")
def health_root() -> dict:
    return health()


@app.get("/")
def index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"detail": "UI not built"}


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith("api") or full_path.startswith("assets"):
        return {"detail": "Not Found"}
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"detail": "UI not built"}
