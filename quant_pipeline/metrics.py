"""Dependency-light metrics for reproducible model and strategy reports."""

from __future__ import annotations

import numpy as np


def _validate_binary_targets(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int64)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("targets must be a one-dimensional binary array")
    return values


def classification_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return thresholded accuracy/F1, Brier score and rank AUC."""

    y = _validate_binary_targets(targets)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("probabilities must be finite and match targets")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    predicted = (p >= 0.5).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((predicted == y).mean()),
        "f1": float(f1),
        "brier": float(np.mean((p - y) ** 2)),
        "positive_rate": float(y.mean()),
        "auc_roc": binary_auc(y, p),
    }


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from average ranks, including tied scores."""

    y = _validate_binary_targets(targets)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("scores must be finite and match targets")
    positives = y == 1
    negatives = y == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if not n_positive or not n_negative:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < s.size:
        end = start + 1
        while end < s.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_positive = ranks[positives].sum()
    u = rank_sum_positive - n_positive * (n_positive + 1) / 2.0
    return float(u / (n_positive * n_negative))


def strategy_metrics(equity: np.ndarray) -> dict[str, float]:
    """Compute return, annualized Sharpe/Sortino, drawdown and win rate."""

    values = np.asarray(equity, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("equity must be a finite one-dimensional series with two points")
    if (values <= 0).any():
        raise ValueError("equity must stay positive for log-free return metrics")
    daily_returns = values[1:] / values[:-1] - 1.0
    std = float(daily_returns.std(ddof=1)) if daily_returns.size > 1 else 0.0
    sharpe = float(daily_returns.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    downside_returns = daily_returns[daily_returns < 0.0]
    downside_std = (
        float(downside_returns.std(ddof=1)) if downside_returns.size > 1 else 0.0
    )
    sortino = (
        float(daily_returns.mean() / downside_std * np.sqrt(252.0))
        if downside_std > 0
        else 0.0
    )
    running_peak = np.maximum.accumulate(values)
    drawdowns = (running_peak - values) / running_peak
    return {
        "total_return": float(values[-1] / values[0] - 1.0),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdowns.max()),
        "win_rate": float((daily_returns > 0).mean()),
    }
