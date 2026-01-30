from __future__ import annotations

from sqlalchemy.engine import Engine

from trustlens.db.schema import Base


def init_db(engine: Engine) -> None:
    """
    Initialize DB schema (idempotent).

    Why ORM create_all:
    - single source of truth (schema.py)
    - easy evolution as we add Slice 3/4/5 tables
    - DB-agnostic (DuckDB now, Postgres later)

    Failure modes:
    - invalid DB URL / permissions -> connection errors upstream
    """
    Base.metadata.create_all(engine)
