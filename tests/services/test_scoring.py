"""Unit tests for scoring."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, Feature, Run, Score
from trustlens.repos.score_repo import ScoreRepository
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.scoring import BaselineScorer, MODEL_VERSION


@pytest.fixture
def in_memory_db():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_run(in_memory_db):
    run = Run(claim_text="Test claim", query_text="test", status="completed")
    in_memory_db.add(run)
    in_memory_db.commit()
    return run


def _insert_features(session, run_id: str) -> None:
    features = [
        Feature(run_id=run_id, feature_group="source_quality", feature_name="weighted_prior_mean", feature_value=0.9),
        Feature(run_id=run_id, feature_group="corroboration", feature_name="domain_diversity", feature_value=1.4),
        Feature(run_id=run_id, feature_group="temporal", feature_name="recency_score", feature_value=0.8),
        Feature(run_id=run_id, feature_group="volume", feature_name="unique_domains", feature_value=5.0),
        Feature(run_id=run_id, feature_group="volume", feature_name="total_articles", feature_value=10.0),
        Feature(run_id=run_id, feature_group="corroboration", feature_name="max_domain_concentration", feature_value=0.2),
        Feature(run_id=run_id, feature_group="temporal", feature_name="missing_timestamp_ratio", feature_value=0.0),
        Feature(run_id=run_id, feature_group="source_quality", feature_name="unknown_source_ratio", feature_value=0.0),
    ]
    session.add_all(features)
    session.commit()


def test_baseline_scorer_math():
    scorer = BaselineScorer()
    features = {
        "weighted_prior_mean": 0.9,
        "domain_diversity": 1.5,
        "recency_score": 0.8,
        "unique_domains": 5.0,
        "total_articles": 10.0,
        "max_domain_concentration": 0.2,
        "missing_timestamp_ratio": 0.0,
        "unknown_source_ratio": 0.0,
    }
    raw_prob, _ = scorer._raw_score(features)
    calibrated = scorer._calibrate(raw_prob)

    assert 0.0 <= raw_prob <= 1.0
    assert 0.0 <= calibrated <= 1.0
    assert scorer._label(0.67) == "credible"
    assert scorer._label(0.66) == "uncertain"
    assert scorer._label(0.33) == "uncertain"
    assert scorer._label(0.32) == "not_credible"


def test_score_persistence(in_memory_db, sample_run):
    _insert_features(in_memory_db, sample_run.run_id)

    service = FeatureEngineeringService(in_memory_db)
    result = service.compute_score_for_run(sample_run.run_id)

    assert 0.0 <= result.score <= 1.0
    assert result.label in {"credible", "uncertain", "not_credible"}

    row = (
        in_memory_db.query(Score)
        .filter(Score.run_id == sample_run.run_id, Score.model_version == MODEL_VERSION)
        .first()
    )
    assert row is not None
    assert row.explanation_json is not None
    assert json.loads(row.explanation_json) != {}


def test_score_idempotent(in_memory_db, sample_run):
    _insert_features(in_memory_db, sample_run.run_id)

    service = FeatureEngineeringService(in_memory_db)
    result1 = service.compute_score_for_run(sample_run.run_id)
    result2 = service.compute_score_for_run(sample_run.run_id)

    repo = ScoreRepository(in_memory_db)
    assert repo.count_by_run(sample_run.run_id) == 1
    assert abs(result1.score - result2.score) < 1e-9
