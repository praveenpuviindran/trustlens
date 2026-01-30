from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    """
    One row per analysis execution.
    """
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False)  # started/completed/failed
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class EvidenceItem(Base):
    """
    Evidence rows linked to a run. GDELT ingestion later.
    """
    __tablename__ = "evidence_items"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    retrieved_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "gdelt_doc2"
    raw_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Feature(Base):
    """
    Feature store: one row per (run, feature_name).
    """
    __tablename__ = "features"

    feature_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    feature_group: Mapped[str] = mapped_column(String, nullable=False)  # prior/corroboration/time
    feature_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)


class ModelVersion(Base):
    """
    Model registry.
    """
    __tablename__ = "model_versions"

    model_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    name: Mapped[str] = mapped_column(String, nullable=False)  # human-friendly identifier
    model_type: Mapped[str] = mapped_column(String, nullable=False)  # logreg/xgb/etc
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
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
