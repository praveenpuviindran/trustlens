"""Unit tests for text feature helpers."""

from trustlens.services.text_features import (
    tokenize,
    jaccard_similarity,
    extract_entities,
    entity_overlap_ratio,
    contradiction_signal,
)


def test_tokenize_and_jaccard():
    tokens = tokenize("Hello, World 123!")
    assert tokens == ["hello", "world", "123"]
    assert jaccard_similarity(tokens, ["hello", "planet"]) == 0.25


def test_entity_overlap():
    claim = "NASA Launch"
    evidence = "NASA Launch Successful"
    claim_entities = extract_entities(claim)
    evidence_entities = extract_entities(evidence)
    assert claim_entities == {"NASA", "Launch"}
    assert entity_overlap_ratio(claim_entities, evidence_entities) == 1.0


def test_contradiction_signal():
    assert contradiction_signal("This is a hoax") is True
    assert contradiction_signal("Verified report") is False
