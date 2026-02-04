"""Unit tests for FeatureRepository."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trustlens.db.schema import Base, Run, Feature
from trustlens.repos.feature_repo import FeatureRepository


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
    run = Run(claim_text="Test", query_text="test", status="completed")
    in_memory_db.add(run)
    in_memory_db.flush()  # Generate ID
    return run


def test_insert_batch(in_memory_db, sample_run):
    features = [Feature(run_id=sample_run.run_id, feature_group="test", 
                       feature_name=f"f{i}", feature_value=float(i)) 
                for i in range(3)]
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    retrieved = repo.get_by_run(sample_run.run_id)
    assert len(retrieved) == 3


def test_delete_by_run(in_memory_db, sample_run):
    features = [Feature(run_id=sample_run.run_id, feature_group="test", 
                       feature_name=f"f{i}", feature_value=float(i)) 
                for i in range(3)]
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    deleted = repo.delete_by_run(sample_run.run_id)
    assert deleted == 3
    assert len(repo.get_by_run(sample_run.run_id)) == 0