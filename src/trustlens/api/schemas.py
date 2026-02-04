"""API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    claim_text: str
    query_text: Optional[str] = None
    max_records: int = Field(25, ge=1, le=250)
    model_id: str = "baseline_v1"
    include_explanation: bool = False


class FeatureOut(BaseModel):
    feature_group: str
    feature_name: str
    feature_value: float


class EvidenceOut(BaseModel):
    domain: str
    title: Optional[str]
    snippet: Optional[str] = None
    url: str
    published_at: Optional[datetime]
    retrieved_at: datetime


class ContributionOut(BaseModel):
    feature_name: str
    value: float
    contribution: float
    weight: Optional[float] = None


class ExplanationSummary(BaseModel):
    summary: str
    bullets: list[str]


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    score: float
    label: str
    top_contributions: list[ContributionOut]
    evidence_count: int
    features: list[FeatureOut]
    explanation: Optional[ExplanationSummary]


class RunMetaResponse(BaseModel):
    run_id: str
    status: str
    claim_text: str
    query_text: Optional[str]
    created_at: datetime
    params_json: Optional[str]
    error_text: Optional[str]
    score: Optional[float] = None
    label: Optional[str] = None


class ScoreResponse(BaseModel):
    run_id: str
    model_version: str
    score: float
    label: str
    created_at: Optional[datetime]
    contributions: Optional[dict[str, Any]]
    thresholds: Optional[dict[str, float]]


class ExplanationResponse(BaseModel):
    run_id: str
    model_id: str
    created_at: Optional[datetime]
    mode: str
    user_question: Optional[str]
    response_text: str
    context_json: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    run_id: str
    answer: str
