from __future__ import annotations

from sqlalchemy.engine import Engine

from trustlens.db.schema import Base


def init_db(engine: Engine) -> None:
    """
    Initialize schema.

    DuckDB + SQLAlchemy can behave oddly with transactional DDL when dropping/creating
    tables and implicit indexes in a single managed transaction. The safest approach:
    - open a plain connection
    - drop_all
    - create_all
    - explicitly commit via the DBAPI connection when available
    """
    conn = engine.connect()
    try:
        # Drop & create schema
        Base.metadata.drop_all(bind=conn)
        Base.metadata.create_all(bind=conn)

        # Explicit commit for DBAPIs that need it (DuckDB benefits here)
        try:
            raw = conn.connection
            if hasattr(raw, "commit"):
                raw.commit()
        except Exception:
            # If the DBAPI doesn't expose commit this way, ignore.
            pass
    finally:
        conn.close()
