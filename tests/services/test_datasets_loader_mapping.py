"""Tests for dataset label mapping."""

from trustlens.services.datasets_loader import map_label


def test_liar_mapping():
    assert map_label("liar", "true") == 1
    assert map_label("liar", "pants-fire") == 0
    assert map_label("liar", "0") == 1
    assert map_label("liar", "1") == 0
    assert map_label("liar", "unknown") is None


def test_fever_mapping():
    assert map_label("fever", "SUPPORTS") == 1
    assert map_label("fever", "REFUTES") == 0
    assert map_label("fever", "0") == 1
    assert map_label("fever", "2") == 0
    assert map_label("fever", "1") is None
    assert map_label("fever", "NOT ENOUGH INFO") is None
