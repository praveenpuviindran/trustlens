from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.models.domain import EvidenceItemRow


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvidenceRepo:
    """Repository for evidence_items."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def add_evidence(
        self,
        run_id: str,
        url: str,
        domain: str,
        source: str,
        title: str | None = None,
        published_at: datetime | None = None,
        raw_json: str | None = None,
    ) -> str:
        evidence_id = str(uuid4())
        retrieved_at = utcnow()

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO evidence_items (
                      evidence_id, run_id, retrieved_at, published_at, url, domain, title, source, raw_json
                    ) VALUES (
                      :evidence_id, :run_id, :retrieved_at, :published_at, :url, :domain, :title, :source, :raw_json
                    )
                    """
                ),
                {
                    "evidence_id": evidence_id,
                    "run_id": run_id,
                    "retrieved_at": retrieved_at,
                    "published_at": published_at,
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "source": source,
                    "raw_json": raw_json,
                },
            )
        return evidence_id

    def list_by_run(self, run_id: str, limit: int = 200) -> list[EvidenceItemRow]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT evidence_id, run_id, retrieved_at, published_at, url, domain, title, source, raw_json
                    FROM evidence_items
                    WHERE run_id = :run_id
                    ORDER BY retrieved_at DESC
                    LIMIT :limit
                    """
                ),
                {"run_id": run_id, "limit": limit},
            ).fetchall()

        return [
            EvidenceItemRow(
                evidence_id=r[0],
                run_id=r[1],
                retrieved_at=r[2],
                published_at=r[3],
                url=r[4],
                domain=r[5],
                title=r[6],
                source=r[7],
                raw_json=r[8],
            )
            for r in rows
        ]
