"""Hugging Face dataset loader and label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from datasets import load_dataset
import numpy as np


LIAR_LABEL_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}

FEVER_LABEL_MAP = {
    "SUPPORTS": 1,
    "REFUTES": 0,
    "NOT ENOUGH INFO": None,
}


@dataclass(frozen=True)
class DatasetRow:
    claim_id: str
    claim_text: str
    label: int


def map_label(dataset_name: str, raw_label: str) -> Optional[int]:
    if dataset_name == "liar":
        return LIAR_LABEL_MAP.get(raw_label.strip().lower())
    if dataset_name == "fever":
        return FEVER_LABEL_MAP.get(raw_label.strip().upper())
    return None


def load_hf_dataset(
    dataset_name: str,
    split: str,
    max_examples: int,
    seed: int,
) -> tuple[list[DatasetRow], dict]:
    if dataset_name not in {"liar", "fever"}:
        raise ValueError("dataset_name must be 'liar' or 'fever'")

    ds = load_dataset(dataset_name, split=split)
    n = len(ds)
    indices = list(range(n))
    if max_examples and max_examples < n:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=max_examples, replace=False).tolist())

    rows: List[DatasetRow] = []
    dropped = 0
    for idx in indices:
        item = ds[int(idx)]
        if dataset_name == "liar":
            claim_text = str(item.get("statement", ""))
            raw_label = str(item.get("label", ""))
        else:
            claim_text = str(item.get("claim", ""))
            raw_label = str(item.get("label", ""))

        label = map_label(dataset_name, raw_label)
        if label is None:
            dropped += 1
            continue
        rows.append(DatasetRow(claim_id=str(idx), claim_text=claim_text, label=label))

    meta = {
        "dataset_name": dataset_name,
        "split": split,
        "max_examples": max_examples,
        "seed": seed,
        "dropped": dropped,
        "selected_indices": indices,
    }
    return rows, meta
