import pytest

from trustlens.services.priors import (
    build_prior_records,
    label_to_prior_score,
    normalize_domain,
)


def test_normalize_domain_basic():
    assert normalize_domain("NYTimes.com") == "nytimes.com"
    assert normalize_domain("www.bbc.co.uk") == "bbc.co.uk"


def test_normalize_domain_url():
    assert normalize_domain("https://www.nytimes.com/2025/01/01/foo") == "nytimes.com"
    assert normalize_domain("http://example.com:8080/path?q=1") == "example.com"
    assert normalize_domain("https://user:pass@www.reuters.com/world") == "reuters.com"


def test_label_to_prior_score_mapping():
    assert label_to_prior_score(-1) == 0.15
    assert label_to_prior_score(0) == 0.50
    assert label_to_prior_score(1) == 0.85
    with pytest.raises(ValueError):
        label_to_prior_score(2)


def test_build_prior_records_pure():
    rows = [
        {"domain": "https://www.reuters.com/world", "reliability_label": 1, "newsguard_score": 95},
        {"domain": "example.com", "reliability_label": 0, "newsguard_score": None},
        {"domain": "", "reliability_label": 1, "newsguard_score": None},
    ]
    priors = build_prior_records(rows)
    assert len(priors) == 2
    assert priors[0].domain == "reuters.com"
    assert priors[0].prior_score == 0.85
    assert priors[1].domain == "example.com"
    assert priors[1].prior_score == 0.50
