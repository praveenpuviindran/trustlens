"""Stratified evaluation utilities and error artifact generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from trustlens.services.evaluation import EvalRow, compute_metrics


@dataclass(frozen=True)
class ErrorCase:
    run_id: str
    claim_text: str
    label: int
    score: float
    predicted_label: str
    key_features: dict


def _pred_to_binary(predicted_label: str) -> int:
    return 1 if predicted_label == "credible" else 0


def _bucket_total_articles(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 10:
        return "4-10"
    return ">10"


def _bucket_unknown_ratio(value: float) -> str:
    if value <= 0.2:
        return "0-0.2"
    if value <= 0.5:
        return "0.2-0.5"
    return ">0.5"


def _bucket_domain_concentration(value: float) -> str:
    if value <= 0.4:
        return "0-0.4"
    if value <= 0.7:
        return "0.4-0.7"
    return ">0.7"


def compute_bucket_metrics(
    eval_rows: List[EvalRow],
    feature_map: Dict[str, Dict[str, float]],
) -> dict:
    buckets = {
        "total_articles": {},
        "unknown_source_ratio": {},
        "max_domain_concentration": {},
    }

    for row in eval_rows:
        feats = feature_map.get(row.run_id, {})
        total_articles = float(feats.get("total_articles", 0.0))
        unknown_ratio = float(feats.get("unknown_source_ratio", 0.0))
        max_conc = float(feats.get("max_domain_concentration", 0.0))

        buckets["total_articles"].setdefault(_bucket_total_articles(total_articles), []).append(row)
        buckets["unknown_source_ratio"].setdefault(_bucket_unknown_ratio(unknown_ratio), []).append(row)
        buckets["max_domain_concentration"].setdefault(_bucket_domain_concentration(max_conc), []).append(row)

    out = {}
    for name, groups in buckets.items():
        out[name] = {}
        for bucket, rows in groups.items():
            out[name][bucket] = compute_metrics(rows)
    return out


def _summarize_features(feats: Dict[str, float]) -> dict:
    keys = ["weighted_prior_mean", "unknown_source_ratio", "max_domain_concentration", "recency_score", "domain_diversity"]
    return {k: float(feats.get(k, 0.0)) for k in keys}


def write_error_artifacts(
    eval_rows: List[EvalRow],
    feature_map: Dict[str, Dict[str, float]],
    claim_map: Dict[str, str],
    out_dir: Path,
    top_n: int = 20,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    false_pos: List[ErrorCase] = []
    false_neg: List[ErrorCase] = []

    for row in eval_rows:
        pred_bin = _pred_to_binary(row.predicted_label)
        feats = feature_map.get(row.run_id, {})
        case = ErrorCase(
            run_id=row.run_id,
            claim_text=claim_map.get(row.run_id, ""),
            label=row.label,
            score=row.score,
            predicted_label=row.predicted_label,
            key_features=_summarize_features(feats),
        )
        if pred_bin == 1 and row.label == 0:
            false_pos.append(case)
        elif pred_bin == 0 and row.label == 1:
            false_neg.append(case)

    false_pos_sorted = sorted(false_pos, key=lambda c: c.score, reverse=True)[:top_n]
    false_neg_sorted = sorted(false_neg, key=lambda c: c.score)[:top_n]

    def _write_csv(path: Path, cases: List[ErrorCase]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["run_id", "claim_text", "label", "predicted_label", "score", "key_features"],
            )
            writer.writeheader()
            for c in cases:
                writer.writerow(
                    {
                        "run_id": c.run_id,
                        "claim_text": c.claim_text,
                        "label": c.label,
                        "predicted_label": c.predicted_label,
                        "score": c.score,
                        "key_features": json.dumps(c.key_features, sort_keys=True),
                    }
                )

    _write_csv(out_dir / "false_positives.csv", false_pos_sorted)
    _write_csv(out_dir / "false_negatives.csv", false_neg_sorted)

    hard_cases = [c for c in false_pos + false_neg if (c.score >= 0.8 or c.score <= 0.2)]
    _write_csv(out_dir / "hard_cases.csv", hard_cases)

    md_path = out_dir / "summary.md"
    md_path.write_text(
        "\n".join(
            [
                "# Error Analysis Summary",
                "",
                f"False positives: {len(false_pos)}",
                f"False negatives: {len(false_neg)}",
                f"Hard cases: {len(hard_cases)}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "false_positives": len(false_pos),
        "false_negatives": len(false_neg),
        "hard_cases": len(hard_cases),
        "out_dir": str(out_dir),
    }
