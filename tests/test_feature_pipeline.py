"""
Integration tests for feature engineering pipeline.

Tests end-to-end flow: evidence → features → persistence.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, Run, EvidenceItem, SourcePrior, Feature
from trustlens.services.feature_engineering import FeatureEngineeringService


@pytest.fixture
def in_memory_db():
    """Create an in-memory DuckDB for testing."""
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def realistic_scenario(in_memory_db):
    """
    Create a realistic test scenario:
    - 1 run
    - 10 evidence items from 5 domains
    - 3 domains with priors (high, medium, low)
    - Mix of recent and old articles
    """
    # Create run
    run = Run(
        claim_text="COVID-19 vaccines are effective",
        query_text="COVID vaccine effectiveness",
        status="completed"
    )
    in_memory_db.add(run)
    in_memory_db.commit()
    
    # Create priors
    priors = [
        SourcePrior(domain="nytimes.com", reliability_label=1, prior_score=0.9),
        SourcePrior(domain="bbc.com", reliability_label=1, prior_score=0.85),
        SourcePrior(domain="fox.com", reliability_label=0, prior_score=0.6),
    ]
    in_memory_db.add_all(priors)
    in_memory_db.commit()
    
    # Create evidence
    now = datetime.utcnow()
    evidence = [
        # Recent articles from reliable sources
        EvidenceItem(
            run_id=run.run_id,
            url="https://nytimes.com/article1",
            domain="nytimes.com",
            title="Vaccine study shows high efficacy",
            source="gdelt",
            published_at=now - timedelta(hours=2)
        ),
        EvidenceItem(
            run_id=run.run_id,
            url="https://nytimes.com/article2",
            domain="nytimes.com",
            title="New data confirms vaccine benefits",
            source="gdelt",
            published_at=now - timedelta(hours=5)
        ),
        EvidenceItem(
            run_id=run.run_id,
            url="https://bbc.com/article1",
            domain="bbc.com",
            title="UK reports vaccine success",
            source="gdelt",
            published_at=now - timedelta(hours=3)
        ),
        # Article from medium-reliability source
        EvidenceItem(
            run_id=run.run_id,
            url="https://fox.com/article1",
            domain="fox.com",
            title="Vaccine debate continues",
            source="gdelt",
            published_at=now - timedelta(hours=1)
        ),
        # Articles from unknown sources
        EvidenceItem(
            run_id=run.run_id,
            url="https://localnews.com/article1",
            domain="localnews.com",
            title="Local clinic reports vaccine uptake",
            source="gdelt",
            published_at=now - timedelta(hours=6)
        ),
        EvidenceItem(
            run_id=run.run_id,
            url="https://scienceblog.com/article1",
            domain="scienceblog.com",
            title="Vaccine mechanism explained",
            source="gdelt",
            published_at=now - timedelta(days=2)
        ),
        # Old article
        EvidenceItem(
            run_id=run.run_id,
            url="https://nytimes.com/article_old",
            domain="nytimes.com",
            title="Early vaccine trials",
            source="gdelt",
            published_at=now - timedelta(days=30)
        ),
        # Article without timestamp
        EvidenceItem(
            run_id=run.run_id,
            url="https://nytimes.com/article_no_date",
            domain="nytimes.com",
            title="Vaccine overview",
            source="gdelt",
            published_at=None
        ),
        EvidenceItem(
            run_id=run.run_id,
            url="https://bbc.com/article2",
            domain="bbc.com",
            title="Global vaccination efforts",
            source="gdelt",
            published_at=now - timedelta(hours=12)
        ),
        EvidenceItem(
            run_id=run.run_id,
            url="https://localnews.com/article2",
            domain="localnews.com",
            title="Community response to vaccines",
            source="gdelt",
            published_at=now - timedelta(hours=8)
        ),
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    return run


def test_compute_features_realistic_scenario(in_memory_db, realistic_scenario):
    """Test feature extraction on a realistic scenario."""
    run = realistic_scenario
    service = FeatureEngineeringService(in_memory_db)
    
    # Extract features
    feature_count = service.compute_features(run.run_id)
    
    # Should have features from all groups
    assert feature_count >= 10  # At least 2-3 features per group
    
    # Retrieve and verify
    features = service.get_features(run.run_id)
    feature_dict = {
        (f.feature_group, f.feature_name): f.feature_value
        for f in features
    }
    
    # Volume features
    assert feature_dict[("volume", "total_articles")] == 10.0
    assert feature_dict[("volume", "unique_domains")] == 5.0
    
    # Source quality features
    # weighted_prior_mean should be between 0.6 and 0.9 (mix of known/unknown)
    weighted_mean = feature_dict[("source_quality", "weighted_prior_mean")]
    assert 0.6 <= weighted_mean <= 0.85
    
    # high_reliability_ratio: 4 articles from nytimes/bbc (> 0.7), out of 10
    high_rel_ratio = feature_dict[("source_quality", "high_reliability_ratio")]
    assert 0.3 <= high_rel_ratio <= 0.5  # ~40% (4/10)
    
    # unknown_source_ratio: 3 articles from unknown domains
    unknown_ratio = feature_dict[("source_quality", "unknown_source_ratio")]
    assert 0.25 <= unknown_ratio <= 0.35  # ~30% (3/10)
    
    # Temporal features
    # recency_score: mostly recent, one old, one missing
    recency = feature_dict[("temporal", "recency_score")]
    assert 0.5 <= recency <= 0.9  # High but not perfect
    
    # publication_span_hours: ~30 days span
    span = feature_dict[("temporal", "publication_span_hours")]
    assert span > 700.0  # At least 30 days in hours
    
    # missing_timestamp_ratio: 1/10
    missing = feature_dict[("temporal", "missing_timestamp_ratio")]
    assert abs(missing - 0.1) < 0.01
    
    # Corroboration features
    # domain_diversity: 5 unique domains, somewhat balanced
    diversity = feature_dict[("corroboration", "domain_diversity")]
    assert diversity > 1.5  # Reasonably diverse
    
    # max_domain_concentration: nytimes has 4/10 articles
    concentration = feature_dict[("corroboration", "max_domain_concentration")]
    assert abs(concentration - 0.4) < 0.01


def test_compute_features_idempotent(in_memory_db, realistic_scenario):
    """Test that compute_features is idempotent."""
    run = realistic_scenario
    service = FeatureEngineeringService(in_memory_db)
    
    # First extraction
    count1 = service.compute_features(run.run_id)
    features1 = service.get_features(run.run_id)
    
    # Second extraction (should delete and recreate)
    count2 = service.compute_features(run.run_id)
    features2 = service.get_features(run.run_id)
    
    # Same number of features
    assert count1 == count2
    assert len(features1) == len(features2)
    
    # Same feature values (order might differ)
    values1 = sorted([(f.feature_name, f.feature_value) for f in features1])
    values2 = sorted([(f.feature_name, f.feature_value) for f in features2])
    assert values1 == values2


def test_compute_features_nonexistent_run(in_memory_db):
    """Test that compute_features raises error for nonexistent run."""
    service = FeatureEngineeringService(in_memory_db)
    
    with pytest.raises(ValueError, match="Run 9999 does not exist"):
        service.compute_features(9999)


def test_compute_features_no_evidence(in_memory_db):
    """Test feature extraction when run has no evidence."""
    # Create empty run
    run = Run(
        claim_text="Test claim",
        query_text="test query",
        status="completed"
    )
    in_memory_db.add(run)
    in_memory_db.commit()
    
    service = FeatureEngineeringService(in_memory_db)
    feature_count = service.compute_features(run.run_id)
    
    # Should still create features (with default/neutral values)
    assert feature_count > 0
    
    features = service.get_features(run.run_id)
    
    # Volume features should be 0
    volume_features = [f for f in features if f.feature_group == "volume"]
    for f in volume_features:
        assert f.feature_value == 0.0


def test_feature_groups_completeness(in_memory_db, realistic_scenario):
    """Test that all expected feature groups are present."""
    run = realistic_scenario
    service = FeatureEngineeringService(in_memory_db)
    
    service.compute_features(run.run_id)
    features = service.get_features(run.run_id)
    
    groups = {f.feature_group for f in features}
    
    # All four groups must be present
    assert "volume" in groups
    assert "source_quality" in groups
    assert "temporal" in groups
    assert "corroboration" in groups


def test_feature_values_are_floats(in_memory_db, realistic_scenario):
    """Test that all feature values are valid floats."""
    run = realistic_scenario
    service = FeatureEngineeringService(in_memory_db)
    
    service.compute_features(run.run_id)
    features = service.get_features(run.run_id)
    
    for f in features:
        assert isinstance(f.feature_value, float)
        assert not (f.feature_value != f.feature_value)  # Not NaN


def test_multiple_runs_independence(in_memory_db):
    """Test that features for different runs are independent."""
    # Create two runs with different evidence profiles
    run1 = Run(claim_text="Claim 1", query_text="query1", status="completed")
    run2 = Run(claim_text="Claim 2", query_text="query2", status="completed")
    in_memory_db.add_all([run1, run2])
    in_memory_db.commit()
    
    # Run 1: 3 articles
    evidence_run1 = [
        EvidenceItem(
            run_id=run1.run_id,
            url=f"https://example.com/run1_{i}",
            domain="example.com",
            title=f"Run1 Article {i}",
            source="test"
        )
        for i in range(3)
    ]
    
    # Run 2: 7 articles
    evidence_run2 = [
        EvidenceItem(
            run_id=run2.run_id,
            url=f"https://example.com/run2_{i}",
            domain="example.com",
            title=f"Run2 Article {i}",
            source="test"
        )
        for i in range(7)
    ]
    
    in_memory_db.add_all(evidence_run1 + evidence_run2)
    in_memory_db.commit()
    
    # Extract features for both
    service = FeatureEngineeringService(in_memory_db)
    service.compute_features(run1.run_id)
    service.compute_features(run2.run_id)
    
    # Retrieve features
    features1 = service.get_features(run1.run_id)
    features2 = service.get_features(run2.run_id)
    
    # Get total_articles feature
    total1 = next(
        f.feature_value for f in features1
        if f.feature_name == "total_articles"
    )
    total2 = next(
        f.feature_value for f in features2
        if f.feature_name == "total_articles"
    )
    
    # Should reflect different evidence counts
    assert total1 == 3.0
    assert total2 == 7.0