# src/trustlens/db/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from trustlens.config.settings import settings


@dataclass(frozen=True)
class DBPingResult:
    ok: bool
    detail: str


def build_engine(db_url: Optional[str] = None) -> Engine:
    """
    Build a SQLAlchemy Engine.

    Why allow db_url override?
    - tests need in-memory or temp DBs
    - CLI/API should default to settings.DB_URL
    """
    url = db_url or os.getenv("DATABASE_URL") or settings.db_url
    if url.startswith("postgresql"):
        return create_engine(url, future=True, pool_pre_ping=True)
    return create_engine(url, future=True)


def ping_db(engine: Engine) -> DBPingResult:
    """
    Lightweight DB connectivity check.
    Must NEVER return None (health endpoint depends on this).
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1")).scalar_one()
        return DBPingResult(ok=True, detail="ok")
    except Exception as e:
        return DBPingResult(ok=False, detail=f"{type(e).__name__}: {e}")
