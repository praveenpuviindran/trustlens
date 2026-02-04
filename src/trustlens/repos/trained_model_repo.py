"""Trained model repository."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import delete
from sqlalchemy.orm import Session

from trustlens.db.schema import TrainedModel


class TrainedModelRepository:
    """Repository for trained_models table operations."""

    def __init__(self, session: Session):
        self.session = session

    def upsert(self, model: TrainedModel) -> TrainedModel:
        self.session.execute(
            delete(TrainedModel).where(TrainedModel.model_id == model.model_id)
        )
        self.session.add(model)
        self.session.commit()
        return model

    def get(self, model_id: str) -> Optional[TrainedModel]:
        return (
            self.session.query(TrainedModel)
            .filter(TrainedModel.model_id == model_id)
            .first()
        )

    def list_models(self) -> List[TrainedModel]:
        return (
            self.session.query(TrainedModel)
            .order_by(TrainedModel.created_at.desc())
            .all()
        )
