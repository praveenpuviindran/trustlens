"""Explanation repository."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from trustlens.db.schema import Explanation


class ExplanationRepository:
    """Repository for explanations table operations."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_latest(
        self,
        run_id: str,
        model_id: str,
        mode: str,
        user_question: Optional[str],
        response_text: str,
        context_json: str,
        created_at: Optional[datetime] = None,
    ) -> Explanation:
        """
        Replace the latest explanation for a (run_id, model_id, mode, user_question).
        """
        self.session.execute(
            delete(Explanation).where(
                Explanation.run_id == run_id,
                Explanation.model_id == model_id,
                Explanation.mode == mode,
                Explanation.user_question == user_question,
            )
        )

        next_id = int(
            self.session.execute(
                select(func.coalesce(func.max(Explanation.explanation_id), 0))
            ).scalar_one()
        ) + 1

        row = Explanation(
            explanation_id=next_id,
            run_id=run_id,
            model_id=model_id,
            mode=mode,
            user_question=user_question,
            response_text=response_text,
            context_json=context_json,
            created_at=created_at or datetime.utcnow(),
        )
        self.session.add(row)
        self.session.commit()
        return row

    def list_by_run(self, run_id: str) -> List[Explanation]:
        """Retrieve explanations for a run."""
        return (
            self.session.query(Explanation)
            .filter(Explanation.run_id == run_id)
            .order_by(Explanation.created_at.desc())
            .all()
        )
