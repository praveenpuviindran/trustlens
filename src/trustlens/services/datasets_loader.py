"""Hugging Face dataset loader and label mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
from pathlib import Path


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

# Dataset sources (parquet-based on HF to avoid script loaders)
DATASET_SOURCES = {
    # UKPLab/liar: binary labels, labels=0 (true), labels=1 (false)
    "liar": {
        "repo": "UKPLab/liar",
        "claim_col": "text",
        "label_col": "labels",
    },
    # lucadiliello/fever: labels converted to integers (SUPPORTS=0, NEI=1, REFUTES=2)
    "fever": {
        "repo": "lucadiliello/fever",
        "claim_col": "claim",
        "label_col": "label",
    },
}

HF_CACHE_DIR = Path("data") / "hf_cache"
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


def _find_cached_dataset_dir(repo_id: str) -> Path | None:
    slug = repo_id.replace("/", "___")
    if not DEFAULT_HF_CACHE.exists():
        return None
    candidates = list(DEFAULT_HF_CACHE.glob(f"{slug}/*/*/*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class DatasetRow:
    claim_id: str
    claim_text: str
    label: int


def map_label(dataset_name: str, raw_label: str) -> Optional[int]:
    if dataset_name == "liar":
        raw = raw_label.strip().lower()
        if raw.isdigit():
            # UKPLab/liar: 0=true, 1=false
            if raw == "0":
                return 1
            if raw == "1":
                return 0
        return LIAR_LABEL_MAP.get(raw)
    if dataset_name == "fever":
        raw = raw_label.strip()
        if raw.isdigit():
            # lucadiliello/fever: 0=SUPPORTS, 1=NEI, 2=REFUTES
            if raw == "0":
                return 1
            if raw == "2":
                return 0
            return None
        return FEVER_LABEL_MAP.get(raw.upper())
    return None


def load_hf_dataset(
    dataset_name: str,
    split: str,
    max_examples: int,
    seed: int,
) -> tuple[list[DatasetRow], dict]:
    if dataset_name not in {"liar", "fever"}:
        raise ValueError("dataset_name must be 'liar' or 'fever'")

    source = DATASET_SOURCES.get(dataset_name)
    if source is None:
        raise ValueError("dataset_name must be 'liar' or 'fever'")

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(source["repo"], split=split, cache_dir=str(HF_CACHE_DIR))
    except Exception:
        cached_dir = _find_cached_dataset_dir(source["repo"])
        if cached_dir is None:
            raise
        try:
            cached = load_from_disk(str(cached_dir))
            if hasattr(cached, "keys") and split in cached:  # DatasetDict
                ds = cached[split]
            else:
                ds = cached
        except Exception:
            # Fallback: read arrow shard directly (common in HF cache)
            arrow_candidate = cached_dir / f"{dataset_name}-{split}.arrow"
            if not arrow_candidate.exists():
                matches = list(cached_dir.glob(f"*{split}*.arrow"))
                if matches:
                    arrow_candidate = matches[0]
            if arrow_candidate.exists():
                ds = Dataset.from_file(str(arrow_candidate))
            else:
                raise
    n = len(ds)
    indices = list(range(n))
    if max_examples and max_examples < n:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=max_examples, replace=False).tolist())

    rows: List[DatasetRow] = []
    dropped = 0
    for idx in indices:
        item = ds[int(idx)]
        claim_text = str(item.get(source["claim_col"], ""))
        raw_label = str(item.get(source["label_col"], ""))

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
