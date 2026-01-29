from __future__ import annotations

from sqlalchemy import text

from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db


def test_init_db_creates_tables(tmp_path) -> None:
    db_path = tmp_path / "test_trustlens.duckdb"
    engine = build_engine(f"duckdb:///{db_path}")

    init_db(engine)

    with engine.connect() as conn:
        # DuckDB exposes information_schema.tables
        rows = conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            )
        ).fetchall()

    table_names = {r[0] for r in rows}
    assert "runs" in table_names
    assert "evidence_items" in table_names
    assert "features" in table_names
    assert "model_versions" in table_names
