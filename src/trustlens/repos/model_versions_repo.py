from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from trustlens.models.domain import ModelVersionRow


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ModelVersionsRepo:
    """Repository for model_versions."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_model_version(
        self,
        name: str,
        model_type: str,
        calibration_method: str | None = None,
        artifact_uri: str | None = None,
        params_json: str | None = None,
        metrics_json: str | None = None,
    ) -> str:
        model_version_id = str(uuid4())
        created_at = utcnow()

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_versions (
                      model_version_id, created_at, name, model_type, calibration_method, artifact_uri, params_json, metrics_json
                    ) VALUES (
                      :model_version_id, :created_at, :name, :model_type, :calibration_method, :artifact_uri, :params_json, :metrics_json
                    )
                    """
                ),
                {
                    "model_version_id": model_version_id,
                    "created_at": created_at,
                    "name": name,
                    "model_type": model_type,
                    "calibration_method": calibration_method,
                    "artifact_uri": artifact_uri,
                    "params_json": params_json,
                    "metrics_json": metrics_json,
                },
            )
        return model_version_id

    def latest(self) -> ModelVersionRow | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT model_version_id, created_at, name, model_type, calibration_method, artifact_uri, params_json, metrics_json
                    FROM model_versions
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            ).fetchone()

        if row is None:
            return None

        return ModelVersionRow(
            model_version_id=row[0],
            created_at=row[1],
            name=row[2],
            model_type=row[3],
            calibration_method=row[4],
            artifact_uri=row[5],
            params_json=row[6],
            metrics_json=row[7],
        )
