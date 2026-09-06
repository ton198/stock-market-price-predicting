"""Train-fitted transforms for model signals consumed by the RL stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalNormalizer:
    """Statistics learned from the RL-training signal period only."""

    mean_: float
    scale_: float


def fit_signal_normalizer(raw_probabilities: np.ndarray) -> SignalNormalizer:
    """Fit a stable z-score transform on training probabilities."""

    values = np.asarray(raw_probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("raw_probabilities must be a non-empty finite 1-D array")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("raw_probabilities must lie in [0, 1]")
    scale = float(values.std())
    return SignalNormalizer(mean_=float(values.mean()), scale_=scale if scale > 0.0 else 1.0)


def normalize_signal(
    raw_probabilities: np.ndarray,
    normalizer: SignalNormalizer,
) -> np.ndarray:
    """Apply a previously train-fitted transform without refitting on test data."""

    values = np.asarray(raw_probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("raw_probabilities must be a finite 1-D array")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("raw_probabilities must lie in [0, 1]")
    if not np.isfinite(normalizer.mean_) or not np.isfinite(normalizer.scale_) or normalizer.scale_ <= 0:
        raise ValueError("normalizer must contain finite positive scale")
    z = (values - normalizer.mean_) / normalizer.scale_
    return (0.5 + 0.5 * np.tanh(z)).astype(np.float32)
