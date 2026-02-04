"""Feature vectorization utilities."""

from __future__ import annotations

from typing import List

import numpy as np
from sqlalchemy.orm import Session

from trustlens.db.schema import Feature


BASE_FEATURE_GROUPS = [
    "volume",
    "source_quality",
    "temporal",
    "corroboration",
]

EXPANDED_FEATURE_GROUPS = BASE_FEATURE_GROUPS + [
    "text_similarity",
    "entity_overlap",
    "consistency",
]


class FeatureVectorizer:
    """Builds deterministic feature vectors from stored features."""

    def __init__(self, session: Session):
        self.session = session

    def canonical_feature_names(self, schema_version: str = "v1") -> List[str]:
        groups = BASE_FEATURE_GROUPS if schema_version == "v1" else EXPANDED_FEATURE_GROUPS
        rows = (
            self.session.query(Feature.feature_group, Feature.feature_name)
            .filter(Feature.feature_group.in_(groups))
            .distinct()
            .order_by(Feature.feature_group, Feature.feature_name)
            .all()
        )
        return [name for _group, name in rows]

    def vectorize_run(self, run_id: str, feature_names: List[str]) -> List[float]:
        rows = (
            self.session.query(Feature.feature_name, Feature.feature_value)
            .filter(Feature.run_id == run_id)
            .all()
        )
        feature_map = {name: float(value) for name, value in rows}
        return [float(feature_map.get(name, 0.0)) for name in feature_names]

    def build_matrix(self, run_ids: List[str], feature_names: List[str]) -> np.ndarray:
        vectors = [self.vectorize_run(run_id, feature_names) for run_id in run_ids]
        return np.asarray(vectors, dtype=float)
