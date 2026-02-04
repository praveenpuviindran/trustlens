"""
Unit tests for FeatureExtractor.

Tests SQL-based feature computation logic with fixture data.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, Run, EvidenceItem, SourcePrior, Feature
from trustlens.services.feature_extraction import FeatureExtractor


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
def sample_run(in_memory_db):
    """Create a sample run."""
    run = Run(
        claim_text="Sample claim for testing",
        query_text="sample query",
        status="completed"
    )
    in_memory_db.add(run)
    in_memory_db.commit()
    return run


def test_volume_features_basic(in_memory_db, sample_run):
    """Test volume features with basic evidence."""
    # Add evidence
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://example{i}.com/article",
            domain=f"example{i}.com",
            title=f"Article {i}",
            source="test"
        )
        for i in range(5)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    # Extract features
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._volume_features(sample_run.run_id)
    
    # Verify
    feature_dict = {f.feature_name: f.feature_value for f in features}
    assert feature_dict["total_articles"] == 5.0
    assert feature_dict["unique_domains"] == 5.0


def test_volume_features_duplicate_domains(in_memory_db, sample_run):
    """Test volume features when multiple articles share domains."""
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://nytimes.com/article{i}",
            domain="nytimes.com",
            title=f"Article {i}",
            source="test"
        )
        for i in range(3)
    ] + [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://bbc.com/article{i}",
            domain="bbc.com",
            title=f"Article {i}",
            source="test"
        )
        for i in range(2)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._volume_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    assert feature_dict["total_articles"] == 5.0
    assert feature_dict["unique_domains"] == 2.0


def test_source_quality_features_with_priors(in_memory_db, sample_run):
    """Test source quality features with known priors."""
    # Add priors
    priors = [
        SourcePrior(domain="nytimes.com", reliability_label=1, prior_score=0.9),
        SourcePrior(domain="bbc.com", reliability_label=1, prior_score=0.85),
        SourcePrior(domain="fox.com", reliability_label=0, prior_score=0.6),
    ]
    in_memory_db.add_all(priors)
    in_memory_db.commit()
    
    # Add evidence
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url="https://nytimes.com/article1",
            domain="nytimes.com",
            title="Article 1",
            source="test"
        ),
        EvidenceItem(
            run_id=sample_run.run_id,
            url="https://bbc.com/article2",
            domain="bbc.com",
            title="Article 2",
            source="test"
        ),
        EvidenceItem(
            run_id=sample_run.run_id,
            url="https://fox.com/article3",
            domain="fox.com",
            title="Article 3",
            source="test"
        ),
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    # Extract features
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._source_quality_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # Weighted mean = (0.9 + 0.85 + 0.6) / 3 = 0.783...
    assert abs(feature_dict["weighted_prior_mean"] - 0.783) < 0.01
    
    # High reliability ratio = 2/3 (nytimes and bbc)
    assert abs(feature_dict["high_reliability_ratio"] - 0.667) < 0.01
    
    # Unknown source ratio = 0/3
    assert feature_dict["unknown_source_ratio"] == 0.0


def test_source_quality_features_missing_priors(in_memory_db, sample_run):
    """Test that missing priors default to 0.5."""
    # Add evidence without priors
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url="https://unknown1.com/article",
            domain="unknown1.com",
            title="Article 1",
            source="test"
        ),
        EvidenceItem(
            run_id=sample_run.run_id,
            url="https://unknown2.com/article",
            domain="unknown2.com",
            title="Article 2",
            source="test"
        ),
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._source_quality_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # All default to 0.5
    assert feature_dict["weighted_prior_mean"] == 0.5
    assert feature_dict["high_reliability_ratio"] == 0.0
    assert feature_dict["unknown_source_ratio"] == 1.0


def test_temporal_features_recent_articles(in_memory_db, sample_run):
    """Test temporal features with recent articles."""
    now = datetime.utcnow()
    
    # Add recent evidence (within last day)
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://example.com/article{i}",
            domain="example.com",
            title=f"Article {i}",
            source="test",
            published_at=now - timedelta(hours=i)
        )
        for i in range(5)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._temporal_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # Recency score should be high (recent articles)
    assert feature_dict["recency_score"] > 0.7
    
    # Publication span = 4 hours (0 to 4 hours ago)
    assert abs(feature_dict["publication_span_hours"] - 4.0) < 0.1
    
    # No missing timestamps
    assert feature_dict["missing_timestamp_ratio"] == 0.0


def test_temporal_features_old_articles(in_memory_db, sample_run):
    """Test temporal features with old articles."""
    now = datetime.utcnow()
    
    # Add old evidence (30 days ago)
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://example.com/article{i}",
            domain="example.com",
            title=f"Article {i}",
            source="test",
            published_at=now - timedelta(days=30)
        )
        for i in range(3)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._temporal_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # Recency score should be very low (old articles)
    assert feature_dict["recency_score"] < 0.01


def test_temporal_features_missing_timestamps(in_memory_db, sample_run):
    """Test temporal features when timestamps are missing."""
    # Add evidence without published_at
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://example.com/article{i}",
            domain="example.com",
            title=f"Article {i}",
            source="test",
            published_at=None  # Missing
        )
        for i in range(3)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._temporal_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # Neutral recency (0.5) for missing timestamps
    assert feature_dict["recency_score"] == 0.5
    
    # No span (all missing)
    assert feature_dict["publication_span_hours"] == 0.0
    
    # All missing
    assert feature_dict["missing_timestamp_ratio"] == 1.0


def test_corroboration_features_diverse_sources(in_memory_db, sample_run):
    """Test corroboration with diverse sources."""
    # Add evidence from many domains (1 article each)
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://example{i}.com/article",
            domain=f"example{i}.com",
            title="Article",
            source="test"
        )
        for i in range(5)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._corroboration_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # High diversity (each domain appears once)
    # Entropy for uniform distribution of 5 = log2(5) ≈ 2.32
    assert abs(feature_dict["domain_diversity"] - 2.32) < 0.1
    
    # Low concentration (max is 1/5 = 0.2)
    assert feature_dict["max_domain_concentration"] == 0.2


def test_corroboration_features_concentrated_source(in_memory_db, sample_run):
    """Test corroboration when one source dominates."""
    # Add evidence: 8 from one domain, 2 from another
    evidence = [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://dominant.com/article{i}",
            domain="dominant.com",
            title=f"Article {i}",
            source="test"
        )
        for i in range(8)
    ] + [
        EvidenceItem(
            run_id=sample_run.run_id,
            url=f"https://minor.com/article{i}",
            domain="minor.com",
            title=f"Article {i}",
            source="test"
        )
        for i in range(2)
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._corroboration_features(sample_run.run_id)
    
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    # Lower diversity (unbalanced distribution)
    assert feature_dict["domain_diversity"] < 1.0
    
    # High concentration (8/10 = 0.8)
    assert feature_dict["max_domain_concentration"] == 0.8


def test_extract_for_run_all_groups(in_memory_db, sample_run):
    """Test that extract_for_run returns features from all groups."""
    # Add minimal evidence
    evidence = EvidenceItem(
        run_id=sample_run.run_id,
        url="https://example.com/article",
        domain="example.com",
        title="Article",
        source="test",
        published_at=datetime.utcnow()
    )
    in_memory_db.add(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor.extract_for_run(sample_run.run_id)
    
    # Verify all groups present
    groups = {f.feature_group for f in features}
    assert "volume" in groups
    assert "source_quality" in groups
    assert "temporal" in groups
    assert "corroboration" in groups
    
    # Verify minimum feature count (2 per group minimum)
    assert len(features) >= 8


def test_no_evidence(in_memory_db, sample_run):
    """Test feature extraction when no evidence exists."""
    extractor = FeatureExtractor(in_memory_db)
    features = extractor.extract_for_run(sample_run.run_id)
    
    # Should return features with default/neutral values
    assert len(features) > 0
    
    # Volume should be 0
    volume_features = [f for f in features if f.feature_group == "volume"]
    for f in volume_features:
        assert f.feature_value == 0.0