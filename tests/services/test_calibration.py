"""Tests for calibration utilities."""

import numpy as np

from trustlens.services.model_training import ModelTrainer


def test_platt_scaling_deterministic():
    trainer = ModelTrainer(session=None)  # type: ignore[arg-type]
    probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

    a1, b1 = trainer.fit_platt_scaling(probs, y)
    a2, b2 = trainer.fit_platt_scaling(probs, y)
    assert abs(a1 - a2) < 1e-9
    assert abs(b1 - b2) < 1e-9

    calibrated = trainer.apply_platt_scaling(probs, a1, b1)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
