"""Leakage-aware data and evaluation utilities for the NASDAQ project."""

from .data import (
    BASE_FEATURE_COLS,
    DEFAULT_FEATURE_COLS,
    ENGINEERED_FEATURE_COLS,
    MarketFrame,
    PreparedDataset,
    SequenceWindows,
    Standardizer,
    SplitSlices,
    build_sequence_windows,
    chronological_slices,
    fit_standardizer,
    fit_target_threshold,
    load_market_csv,
    make_forward_target,
    prepare_dataset,
)
from .signals import SignalNormalizer, fit_signal_normalizer, normalize_signal
from .evaluator import align_test_predictions, evaluate_probability_stream, load_prediction_csv

__all__ = [
    "BASE_FEATURE_COLS",
    "DEFAULT_FEATURE_COLS",
    "ENGINEERED_FEATURE_COLS",
    "MarketFrame",
    "PreparedDataset",
    "SequenceWindows",
    "Standardizer",
    "SplitSlices",
    "build_sequence_windows",
    "chronological_slices",
    "fit_standardizer",
    "fit_target_threshold",
    "load_market_csv",
    "make_forward_target",
    "prepare_dataset",
    "SignalNormalizer",
    "fit_signal_normalizer",
    "normalize_signal",
    "align_test_predictions",
    "evaluate_probability_stream",
    "load_prediction_csv",
]
