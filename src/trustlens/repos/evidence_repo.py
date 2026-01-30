# src/trustlens/repos/evidence_repo.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from trustlens.db.schema import EvidenceItem


def _parse_gdelt_seendate(value: Optional[str]) -> Optional[datetime]:
    """
    GDELT seendate format: 'YYYYMMDDhhmmss'
    Example: '20250129143000'
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return None


class EvidenceRepo:
    """
    DB access layer for EvidenceItem.
    Important: this repo operates on an existing SQLAlchemy Session.
    """

    def __init__(self, engine):
        # Kept for parity with your other repos; not strictly needed when Session is injected.
        self.engine = engine

    def upsert_from_gdelt(
        self,
        session: Session,
        run_id: str,
        url: str,
        domain: str | None,
        title: str | None,
        seendate: str | None,
        raw: dict | None = None,
        source: str = "gdelt_doc",
    ) -> EvidenceItem:
        """
        Upsert by URL (stable unique key).

        Mapping:
        - GDELT seendate -> EvidenceItem.published_at (datetime | None)
        - ingestion time -> EvidenceItem.retrieved_at (datetime, defaulted if not provided)
        - raw -> EvidenceItem.raw_json (string JSON)

        NOTE:
        - With your schema, EvidenceItem.evidence_id is auto-generated (UUID default),
          so we do NOT set it here.
        - Avoid ORDER BY created_at (DuckDB/SQLAlchemy index weirdness + not needed since url unique).
        """
        # One-or-none is safe because schema has UNIQUE(url)
        existing = session.execute(
            select(EvidenceItem).where(EvidenceItem.url == url)
        ).scalar_one_or_none()

        published_at = _parse_gdelt_seendate(seendate)
        raw_json = json.dumps(raw or {})

        now = datetime.utcnow()

        if existing is not None:
            # Update fields (idempotent)
            existing.run_id = run_id
            existing.domain = domain or ""  # domain is NOT NULL in schema
            existing.title = title
            existing.published_at = published_at
            existing.raw_json = raw_json
            existing.source = source

            # retrieved_at tracks "when we pulled it" (optional but useful)
            # If you want to preserve earliest retrieved_at, remove this line.
            existing.retrieved_at = now

            # created_at should NOT change on update; leave it alone.
            session.flush()
            return existing

        # INSERT new
        item = EvidenceItem(
            run_id=run_id,
            url=url,
            domain=domain or "",  # schema: NOT NULL
            title=title,
            source=source,
            raw_json=raw_json,
            published_at=published_at,
            retrieved_at=now,  # schema: NOT NULL
            created_at=now,    # schema: NOT NULL
        )
        session.add(item)
        session.flush()
        return item
