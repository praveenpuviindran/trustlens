"""Unit tests for model training utilities."""

import numpy as np

from trustlens.services.model_training import ModelTrainer
from trustlens.services.feature_vectorizer import FeatureVectorizer
from trustlens.db.schema import Base, Feature, Run
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_logistic_regression_deterministic():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    trainer = ModelTrainer(session=None)  # type: ignore[arg-type]

    w1, b1 = trainer.train_logistic_regression(X, y, lr=0.2, epochs=200)
    w2, b2 = trainer.train_logistic_regression(X, y, lr=0.2, epochs=200)

    assert np.allclose(w1, w2)
    assert abs(b1 - b2) < 1e-9


def test_threshold_tuning_ordering():
    probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    trainer = ModelTrainer(session=None)  # type: ignore[arg-type]
    t_lo, t_hi = trainer.tune_thresholds(probs, y)
    assert t_lo < t_hi


def test_feature_vectorizer_canonical_ordering():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    run = Run(claim_text="c", query_text="q", status="completed")
    session.add(run)
    session.flush()
    session.add_all(
        [
            Feature(run_id=run.run_id, feature_group="b", feature_name="z", feature_value=1.0),
            Feature(run_id=run.run_id, feature_group="a", feature_name="y", feature_value=1.0),
            Feature(run_id=run.run_id, feature_group="a", feature_name="x", feature_value=1.0),
        ]
    )
    session.commit()

    vectorizer = FeatureVectorizer(session)
    names = vectorizer.canonical_feature_names()
    assert names == ["x", "y", "z"]
