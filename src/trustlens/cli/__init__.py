"""Feature Engineering Service."""

from sqlalchemy.orm import Session
from trustlens.services.feature_extraction import FeatureExtractor
from trustlens.repos.feature_repo import FeatureRepository


class FeatureEngineeringService:
    """High-level service for feature engineering workflow."""
    
    def __init__(self, db_session: Session, feature_repo: FeatureRepository = None, feature_extractor: FeatureExtractor = None):
        self.db = db_session
        self.feature_repo = feature_repo or FeatureRepository(db_session)
        self.extractor = feature_extractor or FeatureExtractor(db_session)
    
    def compute_features(self, run_id: int) -> int:
        """Extract and persist features for a run."""
        from trustlens.db.schema import Run
        run = self.db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} does not exist")
        
        self.feature_repo.delete_by_run(run_id)
        features = self.extractor.extract_for_run(run_id)
        self.feature_repo.insert_batch(features)
        return len(features)
    
def get_features(self, run_id: int):
    """Retrieve features for a run."""
    return self.feature_repo.get_by_run(run_id)


def main() -> None:
    """CLI entrypoint for the trustlens console script."""
    from trustlens.cli.app import app

    app()
