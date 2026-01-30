from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db
from trustlens.services import gdelt as gdelt_mod
from trustlens.services.pipeline_evidence import fetch_and_store_evidence


@pytest.fixture()
def temp_db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "test.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("TRUSTLENS_DB_URL", url)
    return url


def test_pipeline_inserts_run_and_evidence(monkeypatch: pytest.MonkeyPatch, temp_db_url: str):
    # Mock GDELT HTTP inside the pipeline by monkeypatching the underlying fetch
    payload = {
        "articles": [
            {"url": "https://ex.com/1", "domain": "ex.com", "title": "t1", "seendate": "20250101000000"},
            {"url": "https://ex.com/2", "domain": "ex.com", "title": "t2", "seendate": "20250102000000"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    def fake_fetch(query: str, max_records: int | None = None, client_override=None):
        return gdelt_mod.fetch_gdelt_articles(query=query, max_records=max_records, client=client)

    monkeypatch.setattr("trustlens.services.pipeline_evidence.fetch_gdelt_articles", fake_fetch)

    engine = build_engine()
    init_db(engine)

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        res = fetch_and_store_evidence(
            session=session,
            claim_text="some claim",
            query_text="some query",
            max_records=2,
        )
        session.commit()

    with engine.begin() as conn:
        runs = conn.execute(text("select count(*) from runs")).scalar_one()
        ev = conn.execute(text("select count(*) from evidence_items")).scalar_one()

    assert runs == 1
    assert ev == 2
    assert res.evidence_inserted == 2
