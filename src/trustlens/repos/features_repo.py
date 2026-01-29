from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.models.domain import FeatureRow


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FeaturesRepo:
    """Repository for features (key-value store)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert_feature(self, run_id: str, feature_group: str, feature_name: str, feature_value: float) -> str:
        """
        Insert a feature row. For MVP we insert new rows; later we can dedupe/update.
        """
        feature_id = str(uuid4())
        created_at = utcnow()

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO features (feature_id, run_id, created_at, feature_group, feature_name, feature_value)
                    VALUES (:feature_id, :run_id, :created_at, :feature_group, :feature_name, :feature_value)
                    """
                ),
                {
                    "feature_id": feature_id,
                    "run_id": run_id,
                    "created_at": created_at,
                    "feature_group": feature_group,
                    "feature_name": feature_name,
                    "feature_value": float(feature_value),
                },
            )
        return feature_id

    def list_by_run(self, run_id: str) -> list[FeatureRow]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT feature_id, run_id, created_at, feature_group, feature_name, feature_value
                    FROM features
                    WHERE run_id = :run_id
                    ORDER BY created_at ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()

        return [
            FeatureRow(
                feature_id=r[0],
                run_id=r[1],
                created_at=r[2],
                feature_group=r[3],
                feature_name=r[4],
                feature_value=float(r[5]),
            )
            for r in rows
        ]
