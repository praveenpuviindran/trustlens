"""Offline model training and threshold tuning."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sqlalchemy.orm import Session

from trustlens.db.schema import TrainedModel
from trustlens.repos.trained_model_repo import TrainedModelRepository
from trustlens.services.feature_vectorizer import FeatureVectorizer


FEATURE_SCHEMA_VERSION = "v1"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _label_to_int(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in {"1", "credible"}:
        return 1
    if value in {"0", "not_credible"}:
        return 0
    if value in {"uncertain"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class DatasetRow:
    run_id: str
    label: int


class ModelTrainer:
    """Offline logistic regression trainer."""

    def __init__(self, session: Session | None):
        self.session = session
        self.vectorizer = FeatureVectorizer(session) if session is not None else None
        self.repo = TrainedModelRepository(session) if session is not None else None

    def load_dataset(self, csv_path: str | Path) -> List[DatasetRow]:
        path = Path(csv_path)
        rows: List[DatasetRow] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = str(row["run_id"]).strip()
                label_raw = str(row["label"])
                label = _label_to_int(label_raw)
                if label is None:
                    # Uncertain or invalid labels are excluded.
                    continue
                rows.append(DatasetRow(run_id=run_id, label=label))
        return rows

    def split_train_val(
        self,
        rows: List[DatasetRow],
        split_ratio: float,
        seed: int = 42,
    ) -> Tuple[List[DatasetRow], List[DatasetRow]]:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        n_train = max(1, int(len(rows) * split_ratio))
        n_train = min(n_train, len(rows) - 1) if len(rows) > 1 else len(rows)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        train = [rows[i] for i in train_idx]
        val = [rows[i] for i in val_idx]
        return train, val

    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        epochs: int = 500,
    ) -> Tuple[np.ndarray, float]:
        n_samples, n_features = X.shape
        weights = np.zeros(n_features, dtype=float)
        intercept = 0.0

        for _ in range(epochs):
            logits = X @ weights + intercept
            probs = _sigmoid(logits)
            error = probs - y
            grad_w = (X.T @ error) / n_samples
            grad_b = float(np.sum(error) / n_samples)
            weights -= lr * grad_w
            intercept -= lr * grad_b

        return weights, intercept

    def predict_proba(self, X: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
        return _sigmoid(X @ weights + intercept)

    def tune_thresholds(self, probs_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        if len(probs_val) == 0:
            return 0.33, 0.67

        candidates = sorted(set(float(p) for p in probs_val))
        best_f1 = -1.0
        best_pair = (0.33, 0.67)

        for i, t_lo in enumerate(candidates):
            for t_hi in candidates[i + 1 :]:
                preds = np.where(probs_val >= t_hi, 1, np.where(probs_val <= t_lo, 0, 0))
                tp = int(np.sum((preds == 1) & (y_val == 1)))
                fp = int(np.sum((preds == 1) & (y_val == 0)))
                fn = int(np.sum((preds == 0) & (y_val == 1)))
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_pair = (float(t_lo), float(t_hi))

        t_lo, t_hi = best_pair
        if t_lo >= t_hi:
            t_lo, t_hi = 0.33, 0.67
        return t_lo, t_hi

    def evaluate(self, probs: np.ndarray, y: np.ndarray) -> dict:
        preds = (probs >= 0.5).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(y) if len(y) else 0.0
        brier = float(np.mean((probs - y) ** 2)) if len(y) else 0.0

        auroc = None
        if len(set(y.tolist())) >= 2:
            order = np.argsort(-probs)
            y_sorted = y[order]
            tpr = [0.0]
            fpr = [0.0]
            pos = int(np.sum(y))
            neg = len(y) - pos
            tp_c = 0
            fp_c = 0
            for label in y_sorted:
                if label == 1:
                    tp_c += 1
                else:
                    fp_c += 1
                tpr.append(tp_c / pos if pos else 0.0)
                fpr.append(fp_c / neg if neg else 0.0)
            auc = 0.0
            for i in range(1, len(tpr)):
                auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
            auroc = float(auc)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "brier": float(brier),
            "auroc": auroc,
        }

    def train_and_register(
        self,
        dataset_path: str | Path,
        dataset_name: str,
        model_id: str,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> TrainedModel:
        if self.session is None or self.vectorizer is None or self.repo is None:
            raise RuntimeError("ModelTrainer requires a valid database session for training.")

        rows = self.load_dataset(dataset_path)
        train_rows, val_rows = self.split_train_val(rows, split_ratio, seed=seed)

        feature_names = self.vectorizer.canonical_feature_names()
        train_ids = [r.run_id for r in train_rows]
        val_ids = [r.run_id for r in val_rows]
        X_train = self.vectorizer.build_matrix(train_ids, feature_names)
        y_train = np.asarray([r.label for r in train_rows], dtype=float)
        X_val = self.vectorizer.build_matrix(val_ids, feature_names)
        y_val = np.asarray([r.label for r in val_rows], dtype=float)

        weights, intercept = self.train_logistic_regression(X_train, y_train)
        probs_val = self.predict_proba(X_val, weights, intercept)
        t_lo, t_hi = self.tune_thresholds(probs_val, y_val)
        metrics = self.evaluate(probs_val, y_val)

        data_bytes = Path(dataset_path).read_bytes()
        dataset_hash = hashlib.sha256(data_bytes).hexdigest()

        feature_names_json = json.dumps(feature_names, sort_keys=True)
        weights_map = {name: float(w) for name, w in zip(feature_names, weights)}
        weights_map["intercept"] = float(intercept)
        weights_json = json.dumps(weights_map, sort_keys=True)
        thresholds_json = json.dumps({"t_lo": t_lo, "t_hi": t_hi}, sort_keys=True)
        metrics_json = json.dumps(metrics, sort_keys=True)

        model = TrainedModel(
            model_id=model_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names_json=feature_names_json,
            weights_json=weights_json,
            thresholds_json=thresholds_json,
            metrics_json=metrics_json,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
        )
        return self.repo.upsert(model)
