from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RunRow:
    run_id: str
    created_at: datetime
    claim_text: str
    query_text: str | None
    status: str
    params_json: str | None
    error_text: str | None


@dataclass(frozen=True)
class EvidenceItemRow:
    evidence_id: str
    run_id: str
    retrieved_at: datetime
    published_at: datetime | None
    url: str
    domain: str
    title: str | None
    source: str
    raw_json: str | None


@dataclass(frozen=True)
class FeatureRow:
    feature_id: str
    run_id: str
    created_at: datetime
    feature_group: str
    feature_name: str
    feature_value: float


@dataclass(frozen=True)
class ModelVersionRow:
    model_version_id: str
    created_at: datetime
    name: str
    model_type: str
    calibration_method: str | None
    artifact_uri: str | None
    params_json: str | None
    metrics_json: str | None
