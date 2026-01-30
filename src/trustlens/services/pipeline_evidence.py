from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from trustlens.db.schema import EvidenceItem, Run
from trustlens.services.gdelt import fetch_gdelt_articles
from trustlens.repos.evidence_repo import EvidenceRepo

@dataclass(frozen=True)
class FetchEvidenceResult:
    run_id: str
    evidence_inserted: int


def create_run(session: Session, claim_text: str, query_text: str | None = None) -> str:
    run_id = str(uuid4())
    run = Run(
        run_id=run_id,
        created_at=datetime.utcnow(),
        claim_text=claim_text,
        query_text=query_text,
        status="started",
        params_json=json.dumps({}),
        error_text=None,
    )
    session.add(run)
    session.flush()  # ensure run exists before evidence inserts
    return run_id


def fetch_and_store_evidence(
    session: Session,
    claim_text: str,
    query_text: str,
    max_records: int = 25,
) -> FetchEvidenceResult:
    engine = session.get_bind()
    if engine is None:
        raise RuntimeError("Session is not bound to an engine (session.get_bind() returned None)")

    evidence_repo = EvidenceRepo(engine=engine)

    # ✅ create ONE run using the helper (returns the run_id string)
    run_id = create_run(session=session, claim_text=claim_text, query_text=query_text)

    articles = fetch_gdelt_articles(query=query_text, max_records=max_records)

    inserted = 0
    for a in articles:
        evidence_repo.upsert_from_gdelt(
            session=session,
            run_id=run_id,
            url=a.url,
            domain=a.domain,
            title=a.title,
            seendate=a.seendate,
            raw=getattr(a, "raw", None),
        )
        inserted += 1
    return FetchEvidenceResult(run_id=run_id, evidence_inserted=inserted)
