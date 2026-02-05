"""Model list API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from trustlens.api.deps import get_db
from trustlens.api.schemas import ModelListResponse
from trustlens.repos.trained_model_repo import TrainedModelRepository

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(session: Session = Depends(get_db)):
    repo = TrainedModelRepository(session)
    models = [m.model_id for m in repo.list_models()]
    if "baseline_v1" not in models:
        models.insert(0, "baseline_v1")
    return ModelListResponse(models=models)
