from __future__ import annotations

from typing import Iterable

from datasets import load_dataset


def load_reliability_rows() -> Iterable[dict]:
    """
    Load rows from HF dataset sergioburdisso/news_media_reliability.

    Dataset columns: domain, reliability_label, newsguard_score. :contentReference[oaicite:2]{index=2}
    """
    ds = load_dataset("sergioburdisso/news_media_reliability")
    # HF returns a DatasetDict with 'train'
    return ds["train"]
