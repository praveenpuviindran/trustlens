# src/trustlens/db/schema.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Run(Base):
    """
    One row per analysis execution.
    """
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False)  # started/completed/failed
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class EvidenceItem(Base):
    """
    Evidence rows linked to a run (articles, sources, etc).
    """
    __tablename__ = "evidence_items"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "gdelt_doc"
    raw_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (UniqueConstraint("url", name="uq_evidence_items_url"),)


class Feature(Base):
    """
    Feature store: one row per (run, feature_name).
    """
    __tablename__ = "features"

    feature_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    feature_group: Mapped[str] = mapped_column(String, nullable=False)
    feature_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)


class ModelVersion(Base):
    """
    Model registry.
    """
    __tablename__ = "model_versions"

    model_version_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    name: Mapped[str] = mapped_column(String, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    calibration_method: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class SourcePrior(Base):
    """
    Domain-level reliability prior loaded from HF dataset.

    reliability_label: {-1, 0, 1}
    newsguard_score: optional [0,100]
    prior_score: mapped to [0,1]
    """
    __tablename__ = "source_priors"

    domain: Mapped[str] = mapped_column(String, primary_key=True)
    reliability_label: Mapped[int] = mapped_column(Integer, nullable=False)
    newsguard_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    prior_score: Mapped[float] = mapped_column(Float, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )


class Score(Base):
    """
    Credibility score per run + model version.
    """
    __tablename__ = "scores"

    score_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=False,
    )
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)

    score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    explanation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (UniqueConstraint("run_id", "model_version", name="uq_scores_run_model"),)
