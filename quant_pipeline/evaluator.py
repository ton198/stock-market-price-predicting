"""Align and evaluate saved model predictions against canonical split anchors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .backtest import buy_and_hold_positions, probability_positions, simulate_positions
from .data import PreparedDataset
from .metrics import classification_metrics, strategy_metrics


def align_split_predictions(
    dataset: PreparedDataset,
    split: str,
    prediction_dates: Sequence[str],
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align a prediction stream to one exact canonical split.

    Exact date equality is intentional.  Silently taking an intersection can
    hide missing rows or reorder predictions, which would make a backtest
    impossible to audit.
    """

    if split not in ("train", "validation", "test"):
        raise ValueError("split must be one of: train, validation, test")
    split_slice = getattr(dataset.splits, split)
    expected_dates = tuple(dataset.dates[split_slice])
    provided_dates = tuple(str(value) for value in prediction_dates)
    values = np.asarray(probabilities, dtype=np.float64)
    if provided_dates != expected_dates:
        raise ValueError(
            f"prediction dates must exactly match canonical {split} anchors "
            f"({len(expected_dates)} expected, {len(provided_dates)} provided)"
        )
    if values.ndim != 1 or values.shape[0] != len(expected_dates):
        raise ValueError(f"probabilities must have one value for every {split} anchor")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return (
        np.asarray(dataset.targets[split_slice], dtype=np.int64),
        np.asarray(dataset.close[split_slice], dtype=np.float64),
        values,
    )


def align_test_predictions(
    dataset: PreparedDataset,
    prediction_dates: Sequence[str],
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the public exact-test alignment contract."""

    return align_split_predictions(dataset, "test", prediction_dates, probabilities)


def evaluate_probability_stream_for_split(
    dataset: PreparedDataset,
    split: str,
    prediction_dates: Sequence[str],
    probabilities: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> dict[str, Any]:
    """Evaluate one exact canonical split and a fair buy/hold baseline."""

    targets, prices, values = align_split_predictions(
        dataset, split, prediction_dates, probabilities
    )
    dates = tuple(str(value) for value in prediction_dates)
    model_equity = simulate_positions(
        prices,
        probability_positions(values),
        initial_equity=initial_equity,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    buy_hold_equity = simulate_positions(
        prices,
        buy_and_hold_positions(prices.size),
        initial_equity=initial_equity,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    return {
        "split": split,
        "split_start": dates[0],
        "split_end": dates[-1],
        "n_samples": int(values.size),
        "classification": classification_metrics(targets, values),
        "strategy": strategy_metrics(model_equity),
        "buy_and_hold": strategy_metrics(buy_hold_equity),
        "backtest": {
            "initial_equity": float(initial_equity),
            "transaction_cost": float(transaction_cost),
            "slippage": float(slippage),
            "position_range": [-1.0, 1.0],
            "position_rule": "clip(2 * probability - 1, -1, 1)",
        },
    }


def evaluate_probability_stream(
    dataset: PreparedDataset,
    prediction_dates: Sequence[str],
    probabilities: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> dict[str, Any]:
    """Evaluate the canonical test set with the original public report schema."""

    split_report = evaluate_probability_stream_for_split(
        dataset,
        "test",
        prediction_dates,
        probabilities,
        initial_equity=initial_equity,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    return {
        "test_start": split_report["split_start"],
        "test_end": split_report["split_end"],
        "n_samples": split_report["n_samples"],
        "classification": split_report["classification"],
        "strategy": split_report["strategy"],
        "buy_and_hold": split_report["buy_and_hold"],
        "backtest": split_report["backtest"],
    }


def load_prediction_csv(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read a two-column ``Date,probability`` prediction artifact."""

    prediction_path = Path(path)
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if "Date" not in fields or "probability" not in fields:
            raise ValueError(f"{prediction_path} must contain Date and probability columns")
        dates: list[str] = []
        values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            dates.append(row["Date"])
            try:
                value = float(row["probability"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_number}: probability is not numeric") from exc
            values.append(value)
    return tuple(dates), np.asarray(values, dtype=np.float64)
