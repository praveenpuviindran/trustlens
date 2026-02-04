"""Evaluation harness and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from trustlens.db.schema import EvalResult
from trustlens.repos.evidence_repo import EvidenceRepo
from trustlens.repos.eval_repo import EvalRepository
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.pipeline_evidence import create_run
from trustlens.services.scoring import MODEL_VERSION, ScoreResult


@dataclass(frozen=True)
class EvalRow:
    """Row of evaluation results used for metric computation."""

    run_id: str
    dataset_name: str
    claim_id: str
    label: int
    score: float
    predicted_label: str


def _predicted_to_binary(predicted_label: str) -> int:
    """
    Map predicted_label to binary.
    Rule: credible=1, not_credible=0, uncertain=0.
    """
    if predicted_label == "credible":
        return 1
    return 0


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom != 0 else 0.0


def _compute_auroc(labels: List[int], scores: List[float]) -> Optional[float]:
    """
    Compute AUROC via trapezoidal integration of ROC curve.
    Returns None if not enough class variety.
    """
    if not labels:
        return None
    if len(set(labels)) < 2:
        return None

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    last_score = None

    for score, label in pairs:
        if last_score is None or score != last_score:
            tpr.append(tp / pos)
            fpr.append(fp / neg)
            last_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    tpr.append(tp / pos)
    fpr.append(fp / neg)

    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(auc)


def _calibration_bins(scores: List[float], labels: List[int], bins: int = 5) -> List[dict]:
    results = []
    if bins <= 0:
        return results

    for i in range(bins):
        low = i / bins
        high = (i + 1) / bins
        if i == bins - 1:
            idx = [j for j, s in enumerate(scores) if low <= s <= high]
        else:
            idx = [j for j, s in enumerate(scores) if low <= s < high]

        if not idx:
            results.append(
                {"bin": i, "low": low, "high": high, "count": 0, "avg_pred": 0.0, "avg_obs": 0.0}
            )
            continue

        avg_pred = sum(scores[j] for j in idx) / len(idx)
        avg_obs = sum(labels[j] for j in idx) / len(idx)
        results.append(
            {
                "bin": i,
                "low": low,
                "high": high,
                "count": len(idx),
                "avg_pred": float(avg_pred),
                "avg_obs": float(avg_obs),
            }
        )
    return results


def compute_metrics(rows: Iterable[EvalRow]) -> Dict[str, object]:
    """
    Compute evaluation metrics for a dataset.
    """
    rows_list = list(rows)
    labels = [r.label for r in rows_list]
    scores = [r.score for r in rows_list]
    preds = [_predicted_to_binary(r.predicted_label) for r in rows_list]

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(labels)) if labels else 0.0

    brier = 0.0
    if labels:
        brier = sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)

    auroc = _compute_auroc(labels, scores)

    return {
        "n": len(labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "brier": float(brier),
        "auroc": auroc,
        "calibration_bins": _calibration_bins(scores, labels, bins=5),
    }


def _load_dataset(dataset_path: str | Path) -> List[dict]:
    path = Path(dataset_path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_evaluation(
    session: Session,
    dataset_path: str | Path,
    dataset_name: str,
    max_records: int,
    evidence_fetcher: Callable[[str, int], list],
    now_fn: Callable[[], datetime] = datetime.utcnow,
    model_version: str = MODEL_VERSION,
) -> Dict[str, object]:
    """
    Run evaluation end-to-end for all rows in a dataset.

    evidence_fetcher: (query_text, max_records) -> list of articles
    now_fn: injectable clock for deterministic tests
    """
    engine = session.get_bind()
    if engine is None:
        raise RuntimeError("Session is not bound to an engine (session.get_bind() returned None)")

    rows = _load_dataset(dataset_path)
    evidence_repo = EvidenceRepo(engine=engine)
    feature_service = FeatureEngineeringService(session)
    eval_repo = EvalRepository(session)

    eval_rows: List[EvalRow] = []
    eval_models: List[EvalResult] = []

    for row in rows:
        claim_id = str(row["claim_id"])
        claim_text = str(row["claim_text"])
        query_text = str(row["query_text"])
        label = int(row["label"])

        run_id = create_run(session=session, claim_text=claim_text, query_text=query_text)
        articles = evidence_fetcher(query_text, max_records)

        for a in articles:
            evidence_repo.upsert_from_gdelt(
                session=session,
                run_id=run_id,
                url=a["url"],
                domain=a.get("domain"),
                title=a.get("title"),
                snippet=a.get("snippet"),
                seendate=a.get("seendate"),
                raw=a.get("raw"),
            )

        session.commit()

        feature_service.compute_features(run_id)
        score_result: ScoreResult = feature_service.compute_score_for_run(run_id, model_version=model_version)

        eval_rows.append(
            EvalRow(
                run_id=run_id,
                dataset_name=dataset_name,
                claim_id=claim_id,
                label=label,
                score=score_result.score,
                predicted_label=score_result.label,
            )
        )
        eval_models.append(
            EvalResult(
                run_id=run_id,
                dataset_name=dataset_name,
                claim_id=claim_id,
                label=label,
                score=score_result.score,
                predicted_label=score_result.label,
                created_at=now_fn(),
            )
        )

    eval_repo.insert_many(eval_models)
    metrics = compute_metrics(eval_rows)

    return {
        "dataset_name": dataset_name,
        "n": metrics["n"],
        "metrics": {k: v for k, v in metrics.items() if k != "calibration_bins"},
        "calibration_bins": metrics["calibration_bins"],
        "model_version": model_version,
        "timestamp": now_fn().isoformat(),
        "rows": eval_rows,
    }
