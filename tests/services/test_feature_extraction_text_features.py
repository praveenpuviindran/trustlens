"""Feature extraction tests for text-based features."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, EvidenceItem, Run
from trustlens.services.feature_extraction import FeatureExtractor


def test_text_similarity_entity_overlap_and_consistency():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    run = Run(claim_text="NASA Launch", query_text="nasa launch", status="completed")
    session.add(run)
    session.flush()

    session.add_all(
        [
            EvidenceItem(
                run_id=run.run_id,
                url="https://example.com/a",
                domain="example.com",
                title="NASA Launch Successful",
                snippet="Mission details announced",
                source="test",
            ),
            EvidenceItem(
                run_id=run.run_id,
                url="https://example.com/b",
                domain="example.com",
                title="Private Launch Failure",
                snippet="",
                source="test",
            ),
            EvidenceItem(
                run_id=run.run_id,
                url="https://example.com/c",
                domain="example.com",
                title="Launch schedule",
                snippet="This is not confirmed",
                source="test",
            ),
        ]
    )
    session.commit()

    extractor = FeatureExtractor(session)

    text_feats = extractor._text_similarity_features(run.run_id)
    text_map = {f.feature_name: f.feature_value for f in text_feats}
    assert abs(text_map["mean_jaccard"] - 0.2421) < 0.05
    assert abs(text_map["max_jaccard"] - 0.3333) < 0.05
    assert abs(text_map["topk_mean_jaccard"] - 0.2421) < 0.05

    entity_feats = extractor._entity_overlap_features(run.run_id)
    entity_map = {f.feature_name: f.feature_value for f in entity_feats}
    assert abs(entity_map["entity_overlap_mean"] - 0.6667) < 0.05
    assert abs(entity_map["entity_overlap_max"] - 1.0) < 0.01

    consistency_feats = extractor._consistency_features(run.run_id)
    consistency_map = {f.feature_name: f.feature_value for f in consistency_feats}
    assert abs(consistency_map["contradiction_signal_ratio"] - (1.0 / 3.0)) < 0.01
