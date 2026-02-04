"""
Unit tests for FeatureRepository.

Tests CRUD operations for features table.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, Run, Feature
from trustlens.repos.feature_repo import FeatureRepository


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
        claim_text="Test claim",
        query_text="test query",
        status="completed"
    )
    in_memory_db.add(run)
    in_memory_db.commit()
    return run


def test_insert_batch_empty_list(in_memory_db):
    """Test that inserting empty list is a no-op."""
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch([])  # Should not raise


def test_insert_batch_single_feature(in_memory_db, sample_run):
    """Test inserting a single feature."""
    feature = Feature(
        run_id=sample_run.run_id,
        feature_group="test",
        feature_name="test_feature",
        feature_value=0.5
    )
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch([feature])
    
    # Verify persistence
    retrieved = repo.get_by_run(sample_run.run_id)
    assert len(retrieved) == 1
    assert retrieved[0].feature_name == "test_feature"
    assert retrieved[0].feature_value == 0.5


def test_insert_batch_multiple_features(in_memory_db, sample_run):
    """Test inserting multiple features."""
    features = [
        Feature(
            run_id=sample_run.run_id,
            feature_group="group1",
            feature_name=f"feature_{i}",
            feature_value=float(i)
        )
        for i in range(5)
    ]
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    # Verify all persisted
    retrieved = repo.get_by_run(sample_run.run_id)
    assert len(retrieved) == 5


def test_get_by_run_ordering(in_memory_db, sample_run):
    """Test that get_by_run returns features ordered by group and name."""
    features = [
        Feature(
            run_id=sample_run.run_id,
            feature_group="group_b",
            feature_name="feature_z",
            feature_value=1.0
        ),
        Feature(
            run_id=sample_run.run_id,
            feature_group="group_a",
            feature_name="feature_y",
            feature_value=2.0
        ),
        Feature(
            run_id=sample_run.run_id,
            feature_group="group_a",
            feature_name="feature_x",
            feature_value=3.0
        ),
    ]
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    retrieved = repo.get_by_run(sample_run.run_id)
    
    # Should be ordered: group_a/feature_x, group_a/feature_y, group_b/feature_z
    assert retrieved[0].feature_name == "feature_x"
    assert retrieved[1].feature_name == "feature_y"
    assert retrieved[2].feature_name == "feature_z"


def test_get_by_run_nonexistent(in_memory_db):
    """Test get_by_run for nonexistent run returns empty list."""
    repo = FeatureRepository(in_memory_db)
    retrieved = repo.get_by_run(9999)
    assert retrieved == []


def test_delete_by_run(in_memory_db, sample_run):
    """Test deleting features for a run."""
    features = [
        Feature(
            run_id=sample_run.run_id,
            feature_group="test",
            feature_name=f"feature_{i}",
            feature_value=float(i)
        )
        for i in range(3)
    ]
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    # Verify inserted
    assert len(repo.get_by_run(sample_run.run_id)) == 3
    
    # Delete
    deleted = repo.delete_by_run(sample_run.run_id)
    assert deleted == 3
    
    # Verify deleted
    assert len(repo.get_by_run(sample_run.run_id)) == 0


def test_delete_by_run_nonexistent(in_memory_db):
    """Test deleting features for nonexistent run returns 0."""
    repo = FeatureRepository(in_memory_db)
    deleted = repo.delete_by_run(9999)
    assert deleted == 0


def test_delete_by_run_idempotent(in_memory_db, sample_run):
    """Test that delete_by_run is idempotent."""
    feature = Feature(
        run_id=sample_run.run_id,
        feature_group="test",
        feature_name="test_feature",
        feature_value=1.0
    )
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch([feature])
    
    # First delete
    deleted1 = repo.delete_by_run(sample_run.run_id)
    assert deleted1 == 1
    
    # Second delete (idempotent)
    deleted2 = repo.delete_by_run(sample_run.run_id)
    assert deleted2 == 0


def test_count_by_run(in_memory_db, sample_run):
    """Test counting features for a run."""
    features = [
        Feature(
            run_id=sample_run.run_id,
            feature_group="test",
            feature_name=f"feature_{i}",
            feature_value=float(i)
        )
        for i in range(7)
    ]
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features)
    
    count = repo.count_by_run(sample_run.run_id)
    assert count == 7


def test_count_by_run_nonexistent(in_memory_db):
    """Test count for nonexistent run returns 0."""
    repo = FeatureRepository(in_memory_db)
    count = repo.count_by_run(9999)
    assert count == 0


def test_multiple_runs_isolation(in_memory_db):
    """Test that features for different runs are isolated."""
    # Create two runs
    run1 = Run(claim_text="Claim 1", query_text="query1", status="completed")
    run2 = Run(claim_text="Claim 2", query_text="query2", status="completed")
    in_memory_db.add_all([run1, run2])
    in_memory_db.commit()
    
    # Add features to each
    features_run1 = [
        Feature(
            run_id=run1.run_id,
            feature_group="test",
            feature_name=f"run1_feature_{i}",
            feature_value=float(i)
        )
        for i in range(3)
    ]
    features_run2 = [
        Feature(
            run_id=run2.run_id,
            feature_group="test",
            feature_name=f"run2_feature_{i}",
            feature_value=float(i)
        )
        for i in range(5)
    ]
    
    repo = FeatureRepository(in_memory_db)
    repo.insert_batch(features_run1)
    repo.insert_batch(features_run2)
    
    # Verify isolation
    run1_features = repo.get_by_run(run1.run_id)
    run2_features = repo.get_by_run(run2.run_id)
    
    assert len(run1_features) == 3
    assert len(run2_features) == 5
    assert all("run1" in f.feature_name for f in run1_features)
    assert all("run2" in f.feature_name for f in run2_features)