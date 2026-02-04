"""Score Repository."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from trustlens.db.schema import Score


class ScoreRepository:
    """Repository for scores table operations."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_score(
        self,
        run_id: str,
        model_version: str,
        score: float,
        label: str,
        explanation_dict: Optional[dict],
    ) -> Score:
        """
        Insert or replace a score row for a (run_id, model_version).

        DuckDB rowcount is unreliable for DELETE; we explicitly delete then insert.
        """
        self.session.execute(
            delete(Score).where(
                Score.run_id == run_id,
                Score.model_version == model_version,
            )
        )

        next_id = int(
            self.session.execute(
                select(func.coalesce(func.max(Score.score_id), 0))
            ).scalar_one()
        ) + 1

        explanation_json = json.dumps(explanation_dict or {})
        row = Score(
            score_id=next_id,
            run_id=run_id,
            model_version=model_version,
            score=float(score),
            label=label,
            explanation_json=explanation_json,
        )
        self.session.add(row)
        self.session.commit()
        return row

    def get_by_run(self, run_id: str) -> list[Score]:
        """Retrieve all scores for a run."""
        return (
            self.session.query(Score)
            .filter(Score.run_id == run_id)
            .order_by(Score.created_at)
            .all()
        )

    def count_by_run(self, run_id: str) -> int:
        """Count score rows for a run."""
        return int(
            self.session.execute(
                select(func.count())
                .select_from(Score)
                .where(Score.run_id == run_id)
            ).scalar_one()
        )
