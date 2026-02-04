"""Credibility scoring service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from sqlalchemy.orm import Session

from trustlens.db.schema import Feature


DEFAULTS: Dict[str, float] = {
    "weighted_prior_mean": 0.5,
    "domain_diversity": 0.0,
    "recency_score": 0.5,
    "unique_domains": 0.0,
    "max_domain_concentration": 0.0,
    "missing_timestamp_ratio": 0.0,
    "unknown_source_ratio": 0.0,
    "total_articles": 0.0,
}

MODEL_VERSION = "baseline_v1"

CALIBRATION_OFFSET = 0.05
CALIBRATION_SCALE = 0.90


@dataclass(frozen=True)
class ScoreResult:
    """Structured output for a credibility score."""

    run_id: str
    score: float
    label: str
    explanation: dict


class BaselineScorer:
    """
    Deterministic, interpretable baseline scorer using weighted features.
    """

    def _fetch_features(self, run_id: str, session: Session) -> Dict[str, float]:
        rows = (
            session.query(Feature.feature_name, Feature.feature_value)
            .filter(Feature.run_id == run_id)
            .all()
        )
        return {name: float(value) for name, value in rows}

    def _log1p(self, value: float) -> float:
        return math.log1p(max(0.0, value))

    def _raw_score(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        values = {**DEFAULTS, **features}

        scaled_total = self._log1p(values["total_articles"])
        scaled_unique = self._log1p(values["unique_domains"])

        contributions = {
            "weighted_prior_mean": 2.0 * values["weighted_prior_mean"],
            "domain_diversity": 1.0 * values["domain_diversity"],
            "recency_score": 0.75 * values["recency_score"],
            "unique_domains": 0.25 * scaled_unique,
            "max_domain_concentration": -1.5 * values["max_domain_concentration"],
            "missing_timestamp_ratio": -0.5 * values["missing_timestamp_ratio"],
            "unknown_source_ratio": -1.0 * values["unknown_source_ratio"],
            "total_articles": 0.1 * scaled_total,
        }

        weighted_sum = sum(contributions.values())
        raw_prob = 1.0 / (1.0 + math.exp(-weighted_sum))
        raw_prob = min(max(raw_prob, 0.0), 1.0)
        return raw_prob, contributions

    def _calibrate(self, raw_prob: float) -> float:
        calibrated = CALIBRATION_OFFSET + CALIBRATION_SCALE * raw_prob
        return min(max(calibrated, 0.0), 1.0)

    def _label(self, score: float) -> str:
        if score >= 0.67:
            return "credible"
        if score >= 0.33:
            return "uncertain"
        return "not_credible"

    def _explanation(
        self,
        features: Dict[str, float],
        contributions: Dict[str, float],
    ) -> dict:
        values = {**DEFAULTS, **features}
        values["total_articles"] = self._log1p(values["total_articles"])
        values["unique_domains"] = self._log1p(values["unique_domains"])

        def as_item(name: str, contrib: float) -> dict:
            return {
                "feature_name": name,
                "contribution": float(contrib),
                "value": float(values[name]),
            }

        positives: List[Tuple[str, float]] = []
        negatives: List[Tuple[str, float]] = []
        for name, contrib in contributions.items():
            if contrib >= 0:
                positives.append((name, contrib))
            else:
                negatives.append((name, contrib))

        positives = sorted(positives, key=lambda x: x[1], reverse=True)[:3]
        negatives = sorted(negatives, key=lambda x: x[1])[:3]

        return {
            "positive": [as_item(name, contrib) for name, contrib in positives],
            "negative": [as_item(name, contrib) for name, contrib in negatives],
        }

    def score_run(self, run_id: str, session: Session) -> ScoreResult:
        """
        Compute a calibrated credibility score and explanation for a run.
        """
        features = self._fetch_features(run_id, session)
        raw_prob, contributions = self._raw_score(features)
        calibrated = self._calibrate(raw_prob)
        label = self._label(calibrated)
        explanation = self._explanation(features, contributions)
        return ScoreResult(run_id=run_id, score=calibrated, label=label, explanation=explanation)
