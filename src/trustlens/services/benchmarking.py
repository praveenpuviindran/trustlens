"""Benchmarking and ablation utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

from trustlens.db.schema import Feature, Run
from trustlens.repos.trained_model_repo import TrainedModelRepository
from trustlens.services.datasets_loader import DatasetRow, load_hf_dataset
from trustlens.services.evaluation import EvalRow, compute_metrics
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.model_training import FEATURE_SCHEMA_VERSION
from trustlens.services.scoring import BaselineScorer, TrainedModelScorer
from trustlens.services.pipeline_evidence import create_run


FEATURE_GROUPS = ["volume", "source_quality", "temporal", "corroboration"]


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_name: str
    dataset_split: str
    max_examples: int
    seed: int
    model_ids: list[str]
    label_mapping_name: str
    max_records: int
    no_fetch_evidence: bool


def _dataset_hash(meta: dict) -> str:
    payload = json.dumps(meta, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    if not probs:
        return 0.0
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels)
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (probs_arr >= lo) & (probs_arr < hi) if i < bins - 1 else (probs_arr >= lo) & (probs_arr <= hi)
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(probs_arr[mask]))
        avg_obs = float(np.mean(labels_arr[mask]))
        ece += abs(avg_pred - avg_obs) * (np.sum(mask) / len(probs_arr))
    return float(ece)


def _plot_calibration(bins: list[dict], out_path: Path) -> None:
    xs = [b["avg_pred"] for b in bins]
    ys = [b["avg_obs"] for b in bins]
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title("Calibration")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _synthetic_features(claim_text: str) -> list[Feature]:
    seed = int(hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    def u(a, b): return float(rng.uniform(a, b))

    features = [
        Feature(feature_group="volume", feature_name="total_articles", feature_value=u(1, 20)),
        Feature(feature_group="volume", feature_name="unique_domains", feature_value=u(1, 10)),
        Feature(feature_group="source_quality", feature_name="weighted_prior_mean", feature_value=u(0, 1)),
        Feature(feature_group="source_quality", feature_name="high_reliability_ratio", feature_value=u(0, 1)),
        Feature(feature_group="source_quality", feature_name="unknown_source_ratio", feature_value=u(0, 1)),
        Feature(feature_group="temporal", feature_name="recency_score", feature_value=u(0, 1)),
        Feature(feature_group="temporal", feature_name="missing_timestamp_ratio", feature_value=u(0, 1)),
        Feature(feature_group="temporal", feature_name="publication_span_hours", feature_value=u(0, 1000)),
        Feature(feature_group="corroboration", feature_name="domain_diversity", feature_value=u(0, 3)),
        Feature(feature_group="corroboration", feature_name="max_domain_concentration", feature_value=u(0, 1)),
    ]
    return features


def run_benchmark(
    session: Session,
    config: BenchmarkConfig,
    evidence_fetcher: Optional[Callable[[str, int], list[dict]]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    if "lr_v1" in config.model_ids:
        repo = TrainedModelRepository(session)
        if not repo.get("lr_v1"):
            raise RuntimeError("lr_v1 not found. Run `trustlens train-model` first.")

    rows, meta = load_hf_dataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        max_examples=config.max_examples,
        seed=config.seed,
    )
    dataset_hash = _dataset_hash({**meta, "label_mapping_name": config.label_mapping_name})

    feature_service = FeatureEngineeringService(session)
    baseline_scorer = BaselineScorer()
    trained_scorer = TrainedModelScorer(session)

    outputs = []
    metrics_by_model: dict = {}
    ablations: dict = {}

    for model_id in config.model_ids:
        eval_rows: List[EvalRow] = []
        probs: List[float] = []
        labels: List[int] = []

        for row in rows:
            run_id = create_run(session, claim_text=row.claim_text, query_text=row.claim_text)

            if config.no_fetch_evidence:
                feats = _synthetic_features(row.claim_text)
                for f in feats:
                    f.run_id = run_id
                session.add_all(feats)
                session.commit()
            else:
                if evidence_fetcher is None:
                    raise RuntimeError("evidence_fetcher required for live benchmark")
                articles = evidence_fetcher(row.claim_text, config.max_records)
                from trustlens.repos.evidence_repo import EvidenceRepo
                evidence_repo = EvidenceRepo(session.get_bind())
                for a in articles:
                    evidence_repo.upsert_from_gdelt(
                        session=session,
                        run_id=run_id,
                        url=a["url"],
                        domain=a.get("domain"),
                        title=a.get("title"),
                        seendate=a.get("seendate"),
                        raw=a.get("raw"),
                    )
                session.commit()
                feature_service.compute_features(run_id)

            if model_id == "baseline_v1":
                result = baseline_scorer.score_run(run_id, session)
            else:
                result = trained_scorer.score_run(run_id, model_id)

            eval_rows.append(
                EvalRow(
                    run_id=run_id,
                    dataset_name=config.dataset_name,
                    claim_id=row.claim_id,
                    label=row.label,
                    score=result.score,
                    predicted_label=result.label,
                )
            )
            probs.append(result.score)
            labels.append(row.label)

            outputs.append(
                {
                    "run_id": run_id,
                    "claim": row.claim_text,
                    "label": row.label,
                    "predicted_prob": result.score,
                    "predicted_label": result.label,
                    "model_id": model_id,
                }
            )

        metrics = compute_metrics(eval_rows)
        metrics["ece"] = _ece(probs, labels)
        metrics_by_model[model_id] = metrics

    # Ablations: zero out each feature group
    for model_id in config.model_ids:
        ablations[model_id] = {}
        for group in FEATURE_GROUPS:
            eval_rows: List[EvalRow] = []
            probs: List[float] = []
            labels: List[int] = []
            for row in rows:
                run = (
                    session.query(Run)
                    .filter(Run.claim_text == row.claim_text)
                    .order_by(Run.created_at.desc())
                    .first()
                )
                if not run:
                    continue
                feats = (
                    session.query(Feature)
                    .filter(Feature.run_id == run.run_id)
                    .order_by(Feature.feature_group, Feature.feature_name)
                    .all()
                )
                feature_map = {f.feature_name: float(f.feature_value) for f in feats}
                group_names = {f.feature_name for f in feats if f.feature_group == group}
                for name in group_names:
                    feature_map[name] = 0.0

                if model_id == "baseline_v1":
                    raw, contribs = baseline_scorer._raw_score(feature_map)
                    score = baseline_scorer._calibrate(raw)
                    label = baseline_scorer._label(score)
                else:
                    # trained model scoring with group zeroed
                    model = TrainedModelRepository(session).get(model_id)
                    if not model:
                        continue
                    feature_names = json.loads(model.feature_names_json)
                    weights_map = json.loads(model.weights_json)
                    intercept = float(weights_map.get("intercept", 0.0))
                    values = [float(feature_map.get(n, 0.0)) for n in feature_names]
                    weights = [float(weights_map.get(n, 0.0)) for n in feature_names]
                    logit = intercept + sum(w * v for w, v in zip(weights, values))
                    score = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
                    thresholds = json.loads(model.thresholds_json)
                    t_lo = float(thresholds.get("t_lo", 0.33))
                    t_hi = float(thresholds.get("t_hi", 0.67))
                    if score >= t_hi:
                        label = "credible"
                    elif score <= t_lo:
                        label = "not_credible"
                    else:
                        label = "uncertain"

                eval_rows.append(
                    EvalRow(
                        run_id=run.run_id,
                        dataset_name=config.dataset_name,
                        claim_id=row.claim_id,
                        label=row.label,
                        score=float(score),
                        predicted_label=label,
                    )
                )
                probs.append(float(score))
                labels.append(row.label)

            metrics = compute_metrics(eval_rows)
            metrics["ece"] = _ece(probs, labels)
            ablations[model_id][group] = metrics

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "config": {
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "max_examples": config.max_examples,
            "seed": config.seed,
            "model_ids": config.model_ids,
            "label_mapping_name": config.label_mapping_name,
            "max_records": config.max_records,
            "no_fetch_evidence": config.no_fetch_evidence,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        "dataset_hash": dataset_hash,
        "metrics": metrics_by_model,
        "ablations": ablations,
        "timestamp": timestamp,
    }

    # Save outputs + plots
    out_dir = output_dir or (Path("reports") / "benchmarks" / config.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{timestamp}_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    preds_path = out_dir / f"{timestamp}_predictions.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "claim", "label", "predicted_prob", "predicted_label", "model_id"],
        )
        writer.writeheader()
        for row in outputs:
            writer.writerow(row)

    for model_id, metrics in metrics_by_model.items():
        bins = metrics.get("calibration_bins", [])
        if bins:
            _plot_calibration(bins, out_dir / f"{timestamp}_{model_id}.png")

    return report
