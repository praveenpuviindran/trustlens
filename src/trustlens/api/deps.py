"""API dependencies."""

from __future__ import annotations

from collections.abc import Generator
from typing import Callable

from sqlalchemy.orm import Session, sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.services.gdelt import fetch_gdelt_articles
from trustlens.config.settings import settings
import time
import os
from trustlens.services.llm_client import get_llm_client, LLMClient


def get_db() -> Generator[Session, None, None]:
    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_evidence_fetcher() -> Callable[[str, int], list[dict]]:
    def _fetch(query_text: str, max_records: int) -> list[dict]:
        attempts = 0
        backoff = 0.5
        while True:
            try:
                timeout_s = float(os.getenv("EVIDENCE_TIMEOUT_S", settings.evidence_timeout_s))
                articles = fetch_gdelt_articles(
                    query=query_text,
                    max_records=max_records,
                    timeout_s=timeout_s,
                )
                break
            except Exception:
                attempts += 1
                if attempts > 2:
                    return []
                time.sleep(backoff)
                backoff *= 2
        return [
            {
                "url": a.url,
                "domain": a.domain,
                "title": a.title,
                "snippet": a.snippet,
                "seendate": a.seendate,
                "raw": getattr(a, "raw", None),
            }
            for a in articles
        ]

    return _fetch


def get_llm() -> LLMClient:
    return get_llm_client()
