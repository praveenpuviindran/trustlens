"""Integration tests for feature engineering pipeline."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trustlens.db.schema import Base, Run, EvidenceItem, SourcePrior
from trustlens.services.feature_engineering import FeatureEngineeringService


@pytest.fixture
def in_memory_db():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def realistic_scenario(in_memory_db):
    run = Run(claim_text="COVID vaccines effective", query_text="covid vaccine", status="completed")
    in_memory_db.add(run)
    in_memory_db.flush()
    
    priors = [SourcePrior(domain="nyt.com", reliability_label=1, prior_score=0.9),
              SourcePrior(domain="bbc.com", reliability_label=1, prior_score=0.85)]
    in_memory_db.add_all(priors)
    in_memory_db.commit()
    
    now = datetime.utcnow()
    evidence = [
        EvidenceItem(run_id=run.run_id, url="https://nyt.com/a1", domain="nyt.com", 
                    title="Study", source="gdelt", published_at=now - timedelta(hours=2)),
        EvidenceItem(run_id=run.run_id, url="https://bbc.com/a1", domain="bbc.com", 
                    title="Report", source="gdelt", published_at=now - timedelta(hours=3)),
        EvidenceItem(run_id=run.run_id, url="https://local.com/a1", domain="local.com", 
                    title="News", source="gdelt", published_at=now - timedelta(hours=5)),
    ]
    in_memory_db.add_all(evidence)
    in_memory_db.commit()
    return run


def test_compute_features_realistic(in_memory_db, realistic_scenario):
    service = FeatureEngineeringService(in_memory_db)
    feature_count = service.compute_features(realistic_scenario.run_id)
    
    assert feature_count >= 10
    features = service.get_features(realistic_scenario.run_id)
    feature_dict = {(f.feature_group, f.feature_name): f.feature_value for f in features}
    
    assert feature_dict[("volume", "total_articles")] == 3.0
    assert feature_dict[("volume", "unique_domains")] == 3.0


def test_idempotent(in_memory_db, realistic_scenario):
    service = FeatureEngineeringService(in_memory_db)
    count1 = service.compute_features(realistic_scenario.run_id)
    count2 = service.compute_features(realistic_scenario.run_id)
    assert count1 == count2