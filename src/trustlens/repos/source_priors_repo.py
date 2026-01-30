from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.services.priors import PriorRecord


class SourcePriorsRepo:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert_many(self, priors: list[PriorRecord]) -> int:
        """
        Upsert priors by primary key (domain). Works for DuckDB (INSERT OR REPLACE)
        and Postgres (ON CONFLICT).

        Returns number of rows attempted.
        """
        if not priors:
            return 0

        dialect = self._engine.dialect.name

        rows = [
            {
                "domain": p.domain,
                "reliability_label": p.reliability_label,
                "newsguard_score": p.newsguard_score,
                "prior_score": p.prior_score,
                "updated_at": p.updated_at,
            }
            for p in priors
        ]

        if dialect == "duckdb":
            stmt = text(
                """
                INSERT OR REPLACE INTO source_priors
                (domain, reliability_label, newsguard_score, prior_score, updated_at)
                VALUES (:domain, :reliability_label, :newsguard_score, :prior_score, :updated_at)
                """
            )
        else:
            # Postgres-friendly
            stmt = text(
                """
                INSERT INTO source_priors
                (domain, reliability_label, newsguard_score, prior_score, updated_at)
                VALUES (:domain, :reliability_label, :newsguard_score, :prior_score, :updated_at)
                ON CONFLICT(domain) DO UPDATE SET
                    reliability_label = EXCLUDED.reliability_label,
                    newsguard_score = EXCLUDED.newsguard_score,
                    prior_score = EXCLUDED.prior_score,
                    updated_at = EXCLUDED.updated_at
                """
            )

        with self._engine.begin() as conn:
            conn.execute(stmt, rows)

        return len(rows)
