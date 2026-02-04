"""Feature Extraction Service"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import math

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from trustlens.db.schema import Feature, EvidenceItem, SourcePrior, Run
from trustlens.services.text_features import (
    tokenize,
    jaccard_similarity,
    extract_entities,
    entity_overlap_ratio,
    contradiction_signal,
)


# Tuneable threshold. Tests expect ~0.4 in the realistic scenario, which aligns with 0.8.
HIGH_RELIABILITY_THRESHOLD = 0.8

# Decay in days: smaller = harsher penalty for older articles
RECENCY_DECAY_DAYS = 1.0

# Default prior when unknown
DEFAULT_UNKNOWN_PRIOR = 0.5


class FeatureExtractor:
    """Extracts features for a given run_id using deterministic aggregations."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def extract_for_run(self, run_id: str) -> List[Feature]:
        """Execute all feature extraction groups."""
        features: List[Feature] = []
        features.extend(self._volume_features(run_id))
        features.extend(self._source_quality_features(run_id))
        features.extend(self._temporal_features(run_id))
        features.extend(self._corroboration_features(run_id))
        features.extend(self._text_similarity_features(run_id))
        features.extend(self._entity_overlap_features(run_id))
        features.extend(self._consistency_features(run_id))
        return features

    # ------------------------
    # Volume
    # ------------------------
    def _volume_features(self, run_id: str) -> List[Feature]:
        total_articles = (
            self.db.query(func.count(EvidenceItem.evidence_id))
            .filter(EvidenceItem.run_id == run_id)
            .scalar()
            or 0
        )

        unique_domains = (
            self.db.query(func.count(func.distinct(EvidenceItem.domain)))
            .filter(EvidenceItem.run_id == run_id)
            .scalar()
            or 0
        )

        return [
            Feature(
                run_id=run_id,
                feature_group="volume",
                feature_name="total_articles",
                feature_value=float(total_articles),
            ),
            Feature(
                run_id=run_id,
                feature_group="volume",
                feature_name="unique_domains",
                feature_value=float(unique_domains),
            ),
        ]

    # ------------------------
    # Source quality
    # ------------------------
    def _source_quality_features(self, run_id: str) -> List[Feature]:
        """
        Compute source-quality features WITHOUT a raw join result list, to avoid
        duplication issues if priors contain multiple rows per domain.

        Steps:
        1) Get evidence domains for the run (one per evidence item).
        2) Fetch priors for those domains (one per domain).
        3) Map in Python, compute ratios safely.
        """
        evidence_domains: List[Optional[str]] = [
            d for (d,) in (
                self.db.query(EvidenceItem.domain)
                .filter(EvidenceItem.run_id == run_id)
                .all()
            )
        ]

        # No evidence case
        if not evidence_domains:
            return [
                Feature(run_id=run_id, feature_group="source_quality", feature_name="weighted_prior_mean", feature_value=DEFAULT_UNKNOWN_PRIOR),
                Feature(run_id=run_id, feature_group="source_quality", feature_name="high_reliability_ratio", feature_value=0.0),
                Feature(run_id=run_id, feature_group="source_quality", feature_name="unknown_source_ratio", feature_value=0.0),
            ]

        # Clean out Nones (domain should exist, but tests might include edge cases)
        domains = [d for d in evidence_domains if d]
        n = len(domains)
        if n == 0:
            return [
                Feature(run_id=run_id, feature_group="source_quality", feature_name="weighted_prior_mean", feature_value=DEFAULT_UNKNOWN_PRIOR),
                Feature(run_id=run_id, feature_group="source_quality", feature_name="high_reliability_ratio", feature_value=0.0),
                Feature(run_id=run_id, feature_group="source_quality", feature_name="unknown_source_ratio", feature_value=1.0),
            ]

        # Fetch priors for these domains. If your priors table accidentally has duplicates,
        # we choose a deterministic one: MAX(prior_score) per domain (simple + stable).
        priors_rows = (
            self.db.query(SourcePrior.domain, func.max(SourcePrior.prior_score))
            .filter(SourcePrior.domain.in_(list(set(domains))))
            .group_by(SourcePrior.domain)
            .all()
        )
        prior_map: Dict[str, float] = {domain: float(score) for domain, score in priors_rows if domain is not None and score is not None}

        # Build per-evidence prior scores with default for unknowns
        prior_scores: List[float] = []
        unknown_count = 0

        for d in domains:
            if d in prior_map:
                p = prior_map[d]
            else:
                p = DEFAULT_UNKNOWN_PRIOR
                unknown_count += 1

            prior_scores.append(p)

        weighted_prior_mean = sum(prior_scores) / len(prior_scores)

        # High-reliability ratio is computed by unique domains to avoid
        # overweighting a single prolific outlet.
        unique_domains = list(set(domains))
        high_domain_count = 0
        for d in unique_domains:
            p = prior_map.get(d, DEFAULT_UNKNOWN_PRIOR)
            if p >= HIGH_RELIABILITY_THRESHOLD:
                high_domain_count += 1
        high_reliability_ratio = high_domain_count / len(unique_domains)

        # Aggregate priors across unique domains
        unique_priors = [prior_map.get(d, DEFAULT_UNKNOWN_PRIOR) for d in unique_domains]
        unique_priors_sorted = sorted(unique_priors)
        median_prior = unique_priors_sorted[len(unique_priors_sorted) // 2]
        min_prior = unique_priors_sorted[0]
        max_prior = unique_priors_sorted[-1]
        unknown_source_ratio = unknown_count / len(prior_scores)

        return [
            Feature(run_id=run_id, feature_group="source_quality", feature_name="weighted_prior_mean", feature_value=float(weighted_prior_mean)),
            Feature(run_id=run_id, feature_group="source_quality", feature_name="high_reliability_ratio", feature_value=float(high_reliability_ratio)),
            Feature(run_id=run_id, feature_group="source_quality", feature_name="unknown_source_ratio", feature_value=float(unknown_source_ratio)),
            Feature(run_id=run_id, feature_group="source_quality", feature_name="median_prior", feature_value=float(median_prior)),
            Feature(run_id=run_id, feature_group="source_quality", feature_name="min_prior", feature_value=float(min_prior)),
            Feature(run_id=run_id, feature_group="source_quality", feature_name="max_prior", feature_value=float(max_prior)),
        ]

    # ------------------------
    # Temporal
    # ------------------------
    def _temporal_features(self, run_id: str) -> List[Feature]:
        """
        Deterministic temporal features.

        Key fix: DON'T use datetime.utcnow() directly, because running the pipeline twice
        will produce slightly different recency_score floats → breaks idempotency tests.

        Use a stable reference time:
        - Run.created_at if available
        - else max(evidence_items.retrieved_at) / max(created_at)
        - else fallback to utcnow (only if there is truly nothing)
        """
        # Pull published_at values
        published_rows = (
            self.db.query(EvidenceItem.published_at)
            .filter(EvidenceItem.run_id == run_id)
            .all()
        )

        if not published_rows:
            return [
                Feature(run_id=run_id, feature_group="temporal", feature_name="recency_score", feature_value=DEFAULT_UNKNOWN_PRIOR),
                Feature(run_id=run_id, feature_group="temporal", feature_name="publication_span_hours", feature_value=0.0),
                Feature(run_id=run_id, feature_group="temporal", feature_name="missing_timestamp_ratio", feature_value=0.0),
            ]

        published_times: List[Optional[datetime]] = [t for (t,) in published_rows]
        total_count = len(published_times)
        valid_times = [t for t in published_times if t is not None]
        missing_count = total_count - len(valid_times)

        missing_timestamp_ratio = (missing_count / total_count) if total_count else 0.0

        # Stable "now"
        ref_now = self._stable_reference_time(run_id)

        # recency score: exp(-days / decay)
        if valid_times:
            recency_scores: List[float] = []
            for pub_time in valid_times:
                days_since = (ref_now - pub_time).total_seconds() / 86400.0
                # Clamp negative (future timestamps) to 0 days since
                if days_since < 0:
                    days_since = 0.0
                recency_scores.append(math.exp(-days_since / RECENCY_DECAY_DAYS))

            # For missing timestamps, inject neutral default
            recency_scores.extend([DEFAULT_UNKNOWN_PRIOR] * missing_count)
            mean_recency = sum(recency_scores) / len(recency_scores)
        else:
            mean_recency = DEFAULT_UNKNOWN_PRIOR

        # Publication span
        if len(valid_times) >= 2:
            earliest = min(valid_times)
            latest = max(valid_times)
            span_hours = (latest - earliest).total_seconds() / 3600.0
        else:
            span_hours = 0.0

        return [
            Feature(run_id=run_id, feature_group="temporal", feature_name="missing_timestamp_ratio", feature_value=float(missing_timestamp_ratio)),
            Feature(run_id=run_id, feature_group="temporal", feature_name="recency_score", feature_value=float(mean_recency)),
            Feature(run_id=run_id, feature_group="temporal", feature_name="publication_span_hours", feature_value=float(span_hours)),
        ]

    def _stable_reference_time(self, run_id: str) -> datetime:
        # 1) Run.created_at (preferred)
        run_created = (
            self.db.query(Run.created_at)
            .filter(Run.run_id == run_id)
            .scalar()
        )
        if isinstance(run_created, datetime):
            return run_created

        # 2) max retrieved_at / created_at from evidence (fallback)
        max_retrieved = (
            self.db.query(func.max(EvidenceItem.retrieved_at))
            .filter(EvidenceItem.run_id == run_id)
            .scalar()
        )
        if isinstance(max_retrieved, datetime):
            return max_retrieved

        max_created = (
            self.db.query(func.max(EvidenceItem.created_at))
            .filter(EvidenceItem.run_id == run_id)
            .scalar()
        )
        if isinstance(max_created, datetime):
            return max_created

        # 3) last resort
        return datetime.utcnow()

    # ------------------------
    # Corroboration
    # ------------------------
    def _corroboration_features(self, run_id: str) -> List[Feature]:
        domain_counts_rows = (
            self.db.query(EvidenceItem.domain, func.count(EvidenceItem.evidence_id))
            .filter(EvidenceItem.run_id == run_id)
            .group_by(EvidenceItem.domain)
            .all()
        )

        if not domain_counts_rows:
            return [
                Feature(run_id=run_id, feature_group="corroboration", feature_name="domain_diversity", feature_value=0.0),
                Feature(run_id=run_id, feature_group="corroboration", feature_name="max_domain_concentration", feature_value=0.0),
            ]

        # Build counts, ignore None domains defensively
        domain_counts: Dict[str, int] = {
            str(domain): int(cnt)
            for domain, cnt in domain_counts_rows
            if domain is not None
        }
        total = sum(domain_counts.values())
        if total == 0:
            return [
                Feature(run_id=run_id, feature_group="corroboration", feature_name="domain_diversity", feature_value=0.0),
                Feature(run_id=run_id, feature_group="corroboration", feature_name="max_domain_concentration", feature_value=0.0),
            ]

        # Shannon entropy (base 2)
        entropy = 0.0
        for cnt in domain_counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_concentration = max(domain_counts.values()) / total

        return [
            Feature(run_id=run_id, feature_group="corroboration", feature_name="domain_diversity", feature_value=float(entropy)),
            Feature(run_id=run_id, feature_group="corroboration", feature_name="max_domain_concentration", feature_value=float(max_concentration)),
        ]

    # ------------------------
    # Text similarity
    # ------------------------
    def _text_similarity_features(self, run_id: str) -> List[Feature]:
        run = self.db.query(Run).filter(Run.run_id == run_id).first()
        if not run or not run.claim_text:
            return [
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="mean_jaccard", feature_value=0.0),
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="max_jaccard", feature_value=0.0),
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="topk_mean_jaccard", feature_value=0.0),
            ]

        claim_tokens = tokenize(run.claim_text)
        evidence_rows = (
            self.db.query(EvidenceItem.title, EvidenceItem.snippet)
            .filter(EvidenceItem.run_id == run_id)
            .all()
        )
        if not evidence_rows:
            return [
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="mean_jaccard", feature_value=0.0),
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="max_jaccard", feature_value=0.0),
                Feature(run_id=run_id, feature_group="text_similarity", feature_name="topk_mean_jaccard", feature_value=0.0),
            ]

        sims = []
        for title, snippet in evidence_rows:
            text = f"{title or ''} {snippet or ''}".strip()
            text_tokens = tokenize(text)
            sims.append(jaccard_similarity(claim_tokens, text_tokens))

        sims_sorted = sorted(sims, reverse=True)
        mean_jaccard = sum(sims_sorted) / len(sims_sorted)
        max_jaccard = sims_sorted[0]
        topk = sims_sorted[:3]
        topk_mean = sum(topk) / len(topk)

        return [
            Feature(run_id=run_id, feature_group="text_similarity", feature_name="mean_jaccard", feature_value=float(mean_jaccard)),
            Feature(run_id=run_id, feature_group="text_similarity", feature_name="max_jaccard", feature_value=float(max_jaccard)),
            Feature(run_id=run_id, feature_group="text_similarity", feature_name="topk_mean_jaccard", feature_value=float(topk_mean)),
        ]

    # ------------------------
    # Entity overlap
    # ------------------------
    def _entity_overlap_features(self, run_id: str) -> List[Feature]:
        run = self.db.query(Run).filter(Run.run_id == run_id).first()
        if not run or not run.claim_text:
            return [
                Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_mean", feature_value=0.0),
                Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_max", feature_value=0.0),
            ]

        claim_entities = extract_entities(run.claim_text)
        evidence_rows = (
            self.db.query(EvidenceItem.title, EvidenceItem.snippet)
            .filter(EvidenceItem.run_id == run_id)
            .all()
        )
        if not evidence_rows:
            return [
                Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_mean", feature_value=0.0),
                Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_max", feature_value=0.0),
            ]

        overlaps = []
        for title, snippet in evidence_rows:
            text = f"{title or ''} {snippet or ''}".strip()
            ev_entities = extract_entities(text)
            overlaps.append(entity_overlap_ratio(claim_entities, ev_entities))

        mean_overlap = sum(overlaps) / len(overlaps)
        max_overlap = max(overlaps)

        return [
            Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_mean", feature_value=float(mean_overlap)),
            Feature(run_id=run_id, feature_group="entity_overlap", feature_name="entity_overlap_max", feature_value=float(max_overlap)),
        ]

    # ------------------------
    # Consistency
    # ------------------------
    def _consistency_features(self, run_id: str) -> List[Feature]:
        evidence_rows = (
            self.db.query(EvidenceItem.title, EvidenceItem.snippet)
            .filter(EvidenceItem.run_id == run_id)
            .all()
        )
        if not evidence_rows:
            return [
                Feature(run_id=run_id, feature_group="consistency", feature_name="contradiction_signal_ratio", feature_value=0.0),
            ]

        count = 0
        for title, snippet in evidence_rows:
            text = f"{title or ''} {snippet or ''}".strip()
            if contradiction_signal(text):
                count += 1
        ratio = count / len(evidence_rows)

        return [
            Feature(run_id=run_id, feature_group="consistency", feature_name="contradiction_signal_ratio", feature_value=float(ratio)),
        ]
