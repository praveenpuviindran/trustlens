"""Text feature helpers."""

from __future__ import annotations

import re
from typing import Iterable, List, Set


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")

CONTRADICTION_KEYWORDS = {
    "not",
    "no",
    "false",
    "hoax",
    "debunk",
    "debunked",
    "deny",
    "denies",
    "refute",
    "refuted",
    "fake",
    "misleading",
}


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def jaccard_similarity(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set and not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def extract_entities(text: str) -> Set[str]:
    return {m.group(0) for m in _ENTITY_RE.finditer(text or "")}


def entity_overlap_ratio(claim_entities: Set[str], evidence_entities: Set[str]) -> float:
    if not claim_entities:
        return 0.0
    return len(claim_entities & evidence_entities) / len(claim_entities)


def contradiction_signal(text: str) -> bool:
    tokens = set(tokenize(text))
    return any(k in tokens for k in CONTRADICTION_KEYWORDS)
