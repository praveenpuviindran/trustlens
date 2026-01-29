from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from trustlens.config.settings import settings


@dataclass(frozen=True)
class DbPingResult:
    ok: bool
    detail: str


def build_engine(database_url: str | None = None) -> Engine:
    """
    Build a SQLAlchemy Engine.

    Inputs:
    - database_url: optional override (useful for tests). Defaults to settings.database_url.

    Failure modes:
    - invalid URL / missing driver => create_engine or connect will fail
    """
    url = database_url or settings.database_url
    return create_engine(url, pool_pre_ping=True)


def ping_db(engine: Engine) -> DbPingResult:
    """
    Minimal DB connectivity check.

    Returns:
    - ok=True if SELECT 1 succeeds
    - ok=False with error detail otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return DbPingResult(ok=True, detail="db_ping_ok")
    except Exception as e:  # noqa: BLE001
        return DbPingResult(ok=False, detail=f"db_ping_failed: {type(e).__name__}: {e}")
