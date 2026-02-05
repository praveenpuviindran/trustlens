"""Tests for evidence fetch retry logic."""

from trustlens.api import deps as deps_mod


def test_evidence_fetch_retry(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temp error")
        return []

    monkeypatch.setattr(deps_mod, "fetch_gdelt_articles", fake_fetch)
    monkeypatch.setattr(deps_mod.time, "sleep", lambda s: None)

    fetcher = deps_mod.get_evidence_fetcher()
    res = fetcher("query", 2)
    assert res == []
    assert calls["n"] == 3
