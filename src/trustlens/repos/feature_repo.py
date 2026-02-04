"""Feature Repository."""

from typing import List

from sqlalchemy import select, func, delete
from sqlalchemy.orm import Session

from trustlens.db.schema import Feature


class FeatureRepository:
    """Repository for features table operations."""

    def __init__(self, session: Session):
        self.session = session

    def insert_batch(self, features: List[Feature]) -> None:
        """Bulk insert features and commit."""
        if not features:
            return
        self.session.add_all(features)
        self.session.commit()

    def get_by_run(self, run_id: str) -> List[Feature]:
        """Retrieve all features for a given run."""
        return (
            self.session.query(Feature)
            .filter(Feature.run_id == run_id)
            .order_by(Feature.feature_group, Feature.feature_name)
            .all()
        )

    def delete_by_run(self, run_id: str) -> int:
        """
        Delete all features for a run.

        DuckDB does NOT reliably return rowcount for DELETE statements,
        so we must compute the count explicitly first.
        """
        # Count rows explicitly
        count = self.session.execute(
            select(func.count())
            .select_from(Feature)
            .where(Feature.run_id == run_id)
        ).scalar_one()

        # Perform delete
        self.session.execute(
            delete(Feature).where(Feature.run_id == run_id)
        )

        self.session.commit()
        return int(count)

    def count_by_run(self, run_id: str) -> int:
        """Count features for a run."""
        return (
            self.session.query(func.count(Feature.feature_id))
            .filter(Feature.run_id == run_id)
            .scalar()
            or 0
        )
