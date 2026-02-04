from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from trustlens.config.settings import settings


from dataclasses import dataclass

@dataclass(frozen=True)
class GdeltArticle:
    url: str
    domain: str | None
    title: str | None
    snippet: str | None
    seendate: str | None  # GDELT: "YYYYMMDDhhmmss"



def _parse_gdelt_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    GDELT commonly returns times like '20250129143000' (YYYYMMDDhhmmss).
    We'll parse that if present.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_gdelt_doc_url(
    query: str,
    max_records: int,
    sort: str = "datedesc",
    mode: str = "artlist",
    fmt: str = "json",
) -> str:
    params = {
        "query": query,
        "mode": mode,
        "format": fmt,
        "maxrecords": max_records,
        "sort": sort,
    }
    return f"{settings.gdelt_doc_base_url}?{urlencode(params)}"


def fetch_gdelt_articles(
    query: str,
    max_records: int | None = None,
    client: httpx.Client | None = None,
) -> list[GdeltArticle]:
    """
    Pull articles from the GDELT DOC API.

    Design:
    - Sync client (simple for CLI + tests)
    - Dependency injection via `client` makes it testable without real HTTP
    """
    max_records = max_records or settings.gdelt_max_records_default
    url = build_gdelt_doc_url(query=query, max_records=max_records)

    close_client = False
    if client is None:
        client = httpx.Client(timeout=settings.gdelt_timeout_s)
        close_client = True

    try:
        r = client.get(url)
        r.raise_for_status()
        payload = r.json()
    finally:
        if close_client:
            client.close()

    articles = payload.get("articles", []) or []

    out: list[GdeltArticle] = []
    for a in articles:
        article_url = a.get("url")
        domain = a.get("domain")
        if not article_url or not domain:
            continue

        out.append(
            GdeltArticle(
                url=article_url,
                domain=domain,
                title=a.get("title"),
                snippet=(
                    a.get("snippet")
                    or a.get("summary")
                    or a.get("description")
                ),
                seendate=a.get("seendate") or a.get("datetime"),
            )
        )

    return out
