from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class PriorRecord:
    domain: str
    reliability_label: int
    newsguard_score: Optional[float]
    prior_score: float
    updated_at: datetime


def normalize_domain(value: str) -> str:
    """
    Normalize either a raw domain ('nytimes.com') or URL ('https://www.nytimes.com/foo').

    Rules:
    - lowercase
    - strip scheme, path, query, fragment
    - strip port
    - strip leading 'www.'
    """
    v = value.strip().lower()
    if not v:
        return ""

    # If it looks like a URL, parse it; else treat as host
    if "://" in v:
        parsed = urlparse(v)
        host = parsed.netloc
    else:
        host = v.split("/")[0]  # defensive in case someone passes 'domain/path'

    # strip credentials if any (rare, but safe)
    if "@" in host:
        host = host.split("@", 1)[1]

    # strip port
    if ":" in host:
        host = host.split(":", 1)[0]

    if host.startswith("www."):
        host = host[4:]

    return host


def label_to_prior_score(label: int) -> float:
    """
    Map dataset label {-1, 0, 1} to a conservative prior in [0,1].
    """
    mapping = {-1: 0.15, 0: 0.50, 1: 0.85}
    if label not in mapping:
        raise ValueError(f"Unexpected reliability_label={label}. Expected one of -1,0,1.")
    return mapping[label]


def build_prior_records(
    rows: Iterable[dict],
    *,
    now: Optional[datetime] = None,
) -> list[PriorRecord]:
    """
    Convert dataset rows into normalized PriorRecord objects.

    Keeps this pure + testable: caller decides where rows come from (HF vs local).
    """
    ts = now or datetime.now(timezone.utc)
    out: list[PriorRecord] = []
    for r in rows:
        domain_raw = str(r.get("domain", "")).strip()
        if not domain_raw:
            continue

        domain = normalize_domain(domain_raw)
        if not domain:
            continue

        label = int(r["reliability_label"])
        prior = label_to_prior_score(label)

        ng = r.get("newsguard_score", None)
        newsguard = None if ng is None else float(ng)

        out.append(
            PriorRecord(
                domain=domain,
                reliability_label=label,
                newsguard_score=newsguard,
                prior_score=prior,
                updated_at=ts,
            )
        )
    return out
