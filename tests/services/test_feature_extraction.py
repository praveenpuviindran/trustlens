"""Unit tests for FeatureExtractor."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trustlens.db.schema import Base, Run, EvidenceItem, SourcePrior
from trustlens.services.feature_extraction import FeatureExtractor


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
    in_memory_db.flush()  # This generates the ID
    return run


def test_volume_features_basic(in_memory_db, sample_run):
    evidence = [EvidenceItem(run_id=sample_run.run_id, url=f"https://ex{i}.com/a", 
                            domain=f"ex{i}.com", title=f"A{i}", source="test") 
                for i in range(5)]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._volume_features(sample_run.run_id)
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    assert feature_dict["total_articles"] == 5.0
    assert feature_dict["unique_domains"] == 5.0


def test_source_quality_with_priors(in_memory_db, sample_run):
    priors = [SourcePrior(domain="nyt.com", reliability_label=1, prior_score=0.9),
              SourcePrior(domain="bbc.com", reliability_label=1, prior_score=0.85)]
    in_memory_db.add_all(priors)
    in_memory_db.commit()
    
    evidence = [EvidenceItem(run_id=sample_run.run_id, url="https://nyt.com/a", 
                            domain="nyt.com", title="A", source="test"),
                EvidenceItem(run_id=sample_run.run_id, url="https://bbc.com/a", 
                            domain="bbc.com", title="B", source="test")]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._source_quality_features(sample_run.run_id)
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    assert abs(feature_dict["weighted_prior_mean"] - 0.875) < 0.01
    assert feature_dict["high_reliability_ratio"] == 1.0


def test_temporal_features_recent(in_memory_db, sample_run):
    now = datetime.utcnow()
    evidence = [EvidenceItem(run_id=sample_run.run_id, url=f"https://ex.com/{i}", 
                            domain="ex.com", title=f"A{i}", source="test",
                            published_at=now - timedelta(hours=i)) 
                for i in range(3)]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._temporal_features(sample_run.run_id)
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    assert feature_dict["recency_score"] > 0.7
    assert feature_dict["missing_timestamp_ratio"] == 0.0


def test_corroboration_diverse(in_memory_db, sample_run):
    evidence = [EvidenceItem(run_id=sample_run.run_id, url=f"https://ex{i}.com/a", 
                            domain=f"ex{i}.com", title="A", source="test") 
                for i in range(5)]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    
    extractor = FeatureExtractor(in_memory_db)
    features = extractor._corroboration_features(sample_run.run_id)
    feature_dict = {f.feature_name: f.feature_value for f in features}
    
    assert abs(feature_dict["domain_diversity"] - 2.32) < 0.1
    assert feature_dict["max_domain_concentration"] == 0.2