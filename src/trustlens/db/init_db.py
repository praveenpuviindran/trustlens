from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.db.schema import SCHEMA_SQL


def init_db(engine: Engine) -> None:
    """
    Initialize database schema (idempotent).

    Why a function:
    - testable
    - clean separation from CLI
    - keeps SQL in one place

    Failure modes:
    - permissions / invalid SQL => raises exception
    """
    with engine.begin() as conn:
        # DuckDB + Postgres accept multiple statements via driver support in many cases,
        # but to be safe, we split on ';' and execute non-empty statements one by one.
        statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
