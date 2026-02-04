"""Evaluation results repository."""

from __future__ import annotations

from typing import Iterable, List

from sqlalchemy.orm import Session

from trustlens.db.schema import EvalResult


class EvalRepository:
    """Repository for eval_results operations."""

    def __init__(self, session: Session):
        self.session = session

    def insert_many(self, rows: Iterable[EvalResult]) -> int:
        """Insert evaluation result rows and commit."""
        items = list(rows)
        if not items:
            return 0
        self.session.add_all(items)
        self.session.commit()
        return len(items)

    def list_by_dataset(self, dataset_name: str) -> List[EvalResult]:
        """Retrieve results for a dataset."""
        return (
            self.session.query(EvalResult)
            .filter(EvalResult.dataset_name == dataset_name)
            .order_by(EvalResult.created_at)
            .all()
        )
