from __future__ import annotations

from trustlens.db.engine import build_engine, ping_db


def test_ping_db_inmemory_duckdb() -> None:
    engine = build_engine("duckdb:///:memory:")
    result = ping_db(engine)
    assert result.ok is True
