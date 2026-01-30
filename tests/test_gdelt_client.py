from __future__ import annotations

import httpx

from trustlens.services.gdelt import fetch_gdelt_articles


def test_fetch_gdelt_articles_parses_json_without_network():
    # Fake minimal GDELT JSON payload
    payload = {
        "articles": [
            {
                "url": "https://example.com/a",
                "domain": "example.com",
                "title": "Example A",
                "seendate": "20250129143000",
            },
            {
                "url": "https://example.com/b",
                "domain": "example.com",
                "title": "Example B",
                "seendate": "20250129153000",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    arts = fetch_gdelt_articles(query="anything", max_records=2, client=client)
    assert len(arts) == 2
    assert arts[0].domain == "example.com"
    assert arts[0].url == "https://example.com/a"
