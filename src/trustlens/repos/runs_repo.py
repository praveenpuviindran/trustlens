from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.models.domain import RunRow


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RunsRepo:
    """
    Repository for the `runs` table.

    Responsibility:
    - create runs
    - update run status
    - fetch runs

    Interview note:
    This pattern keeps SQL isolated so services stay testable and readable.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_run(self, claim_text: str, query_text: str | None = None, params_json: str | None = None) -> str:
        run_id = str(uuid4())
        created_at = utcnow()
        status = "started"

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO runs (run_id, created_at, claim_text, query_text, status, params_json, error_text)
                    VALUES (:run_id, :created_at, :claim_text, :query_text, :status, :params_json, NULL)
                    """
                ),
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "claim_text": claim_text,
                    "query_text": query_text,
                    "status": status,
                    "params_json": params_json,
                },
            )
        return run_id

    def mark_completed(self, run_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'completed', error_text = NULL WHERE run_id = :run_id"),
                {"run_id": run_id},
            )

    def mark_failed(self, run_id: str, error_text: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed', error_text = :error_text WHERE run_id = :run_id"),
                {"run_id": run_id, "error_text": error_text},
            )

    def get_run(self, run_id: str) -> RunRow | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT run_id, created_at, claim_text, query_text, status, params_json, error_text
                    FROM runs
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).fetchone()

        if row is None:
            return None

        return RunRow(
            run_id=row[0],
            created_at=row[1],
            claim_text=row[2],
            query_text=row[3],
            status=row[4],
            params_json=row[5],
            error_text=row[6],
        )
