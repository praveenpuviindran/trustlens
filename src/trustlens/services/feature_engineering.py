"""Feature Engineering Service."""

from sqlalchemy.orm import Session
from trustlens.services.feature_extraction import FeatureExtractor
from trustlens.repos.feature_repo import FeatureRepository
from trustlens.repos.score_repo import ScoreRepository
from trustlens.services.scoring import BaselineScorer, MODEL_VERSION, ScoreResult, TrainedModelScorer


class FeatureEngineeringService:
    """High-level service for feature engineering workflow."""
    
    def __init__(self, db_session: Session, feature_repo: FeatureRepository = None, feature_extractor: FeatureExtractor = None):
        self.db = db_session
        self.feature_repo = feature_repo or FeatureRepository(db_session)
        self.extractor = feature_extractor or FeatureExtractor(db_session)
    
    def compute_features(self, run_id: str) -> int:
        """Extract and persist features for a run."""
        from trustlens.db.schema import Run
        run = self.db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} does not exist")
        
        self.feature_repo.delete_by_run(run_id)
        features = self.extractor.extract_for_run(run_id)
        self.feature_repo.insert_batch(features)
        return len(features)
    
    def get_features(self, run_id: str):
        """Retrieve features for a run."""
        return self.feature_repo.get_by_run(run_id)

    def compute_score_for_run(self, run_id: str, model_version: str = MODEL_VERSION) -> ScoreResult:
        """
        Compute and persist a credibility score for a run.

        Requires that features have already been computed for the run.
        """
        from trustlens.db.schema import Run
        run = self.db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} does not exist")

        if self.feature_repo.count_by_run(run_id) == 0:
            raise ValueError(f"Run {run_id} has no computed features")

        if model_version == MODEL_VERSION:
            scorer = BaselineScorer()
            result = scorer.score_run(run_id, self.db)
        else:
            scorer = TrainedModelScorer(self.db)
            result = scorer.score_run(run_id, model_version)

        score_repo = ScoreRepository(self.db)
        score_repo.upsert_score(
            run_id=run_id,
            model_version=model_version,
            score=result.score,
            label=result.label,
            explanation_dict=result.explanation,
        )
        return result
