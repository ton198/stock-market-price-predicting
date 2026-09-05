"""Canonical, leakage-aware preparation for the project data.

The original notebooks independently engineered labels and splits.  This
module is the single source of truth used by the reproducible evaluation
scripts.  It deliberately depends only on the Python standard library and
NumPy so that the data contract can be tested without a deep-learning stack.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np


BASE_FEATURE_COLS: tuple[str, ...] = (
    "Return",
    "Return_lag1",
    "Return_lag2",
    "Return_lag5",
    "Price_SMA20",
    "Price_SMA50",
    "RSI_14",
    "MACD_Hist",
    "BB_pos",
    "BB_width",
    "Volume_ratio",
    "Volume_change",
    "VIX",
    "TNX",
    "FedRate",
    "CPI_MoM",
)

ENGINEERED_FEATURE_COLS: tuple[str, ...] = (
    "Regime",
    "VIX_percentile",
    "Momentum_20d",
    "Momentum_60d",
    "Momentum_252d",
    "Volatility_20d",
    "ATR_14",
    "Yield_slope",
    "Dist_52w_high",
    "Dist_52w_low",
    "VIX_change5",
    "Gap",
    "FedRate_chg20",
    "FedRate_chg60",
)

# Keep this order identical to the Hybrid LSTM notebook's feature contract.
DEFAULT_FEATURE_COLS: tuple[str, ...] = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS


@dataclass(frozen=True)
class SplitSlices:
    """Non-overlapping chronological anchor ranges."""

    train: slice
    validation: slice
    test: slice

    def as_dict(self) -> dict[str, slice]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class Standardizer:
    """A small StandardScaler equivalent with explicit fit statistics."""

    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                "values must be a 2-D array with the same number of columns "
                "used to fit the standardizer"
            )
        return ((array - self.mean_) / self.scale_).astype(np.float32)


@dataclass(frozen=True)
class MarketFrame:
    dates: tuple[str, ...]
    close: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    forward_returns: np.ndarray


@dataclass(frozen=True)
class SequenceWindows:
    features: np.ndarray
    targets: np.ndarray
    anchor_indices: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared arrays and the train-fitted statistics used to create them."""

    dates: tuple[str, ...]
    close: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    forward_returns: np.ndarray
    targets: np.ndarray
    splits: SplitSlices
    threshold: float
    scaler: Standardizer
    horizon: int
    raw_rows: int = 0
    dropped_rows: int = 0


def chronological_slices(
    n_rows: int,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> SplitSlices:
    """Return contiguous, exhaustive slices in chronological order.

    The ratios apply to usable prediction anchors, not to raw rows that lack a
    future label.  ``test_ratio`` is explicit to make accidental row omission
    impossible (the old notebook used ratios summing to 0.9).
    """

    if n_rows < 3:
        raise ValueError("at least three rows are required for train/val/test")
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(r <= 0.0 for r in ratios):
        raise ValueError("all split ratios must be positive")
    if not np.isclose(sum(ratios), 1.0, atol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    # Assign rounding residue to test so every usable row belongs to a split.
    train_end = max(1, min(train_end, n_rows - 2))
    val_end = max(train_end + 1, min(val_end, n_rows - 1))
    return SplitSlices(slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows))


def _slice_indices(values: np.ndarray, row_slice: slice) -> np.ndarray:
    indices = np.arange(values.shape[0])[row_slice]
    if indices.size == 0:
        raise ValueError("fit slice selects no rows")
    return indices


def fit_target_threshold(forward_returns: np.ndarray, train_slice: slice) -> float:
    """Fit the binary-label threshold using *training rows only*.

    This is intentionally separate from :func:`make_forward_target`: callers
    can inspect and persist the threshold, making it impossible to silently
    recompute it from validation or test outcomes.
    """

    values = np.asarray(forward_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("forward_returns must be one-dimensional")
    selected = values[_slice_indices(values, train_slice)]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        raise ValueError("training rows contain no finite forward returns")
    return float(np.median(finite))


def make_forward_target(forward_returns: np.ndarray, threshold: float) -> np.ndarray:
    """Convert forward returns to 0/1 labels; unavailable labels become -1."""

    values = np.asarray(forward_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("forward_returns must be one-dimensional")
    targets = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    targets[finite] = (values[finite] > float(threshold)).astype(np.int64)
    return targets


def fit_standardizer(values: np.ndarray, fit_slice: slice) -> Standardizer:
    """Fit column-wise mean/std on one chronological slice only."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must be a 2-D array")
    rows = array[_slice_indices(array, fit_slice)]
    if not np.isfinite(rows).all():
        raise ValueError("fit rows contain non-finite feature values")
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    # A constant feature should remain zero after transformation, not divide by
    # zero or introduce NaNs into a sequence model.
    scale = np.where(scale > 0.0, scale, 1.0)
    return Standardizer(mean_=mean.astype(np.float32), scale_=scale.astype(np.float32))


def build_sequence_windows(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    anchor_slice: slice,
    window_size: int,
    stride: int = 1,
) -> SequenceWindows:
    """Build windows ending at each anchor in ``anchor_slice``.

    The window includes the anchor's observed features and never uses a row
    after the anchor.  Validation/test windows may therefore use historical
    context immediately preceding their split boundary, which is available at
    prediction time and avoids throwing away the first ``window_size - 1``
    observations of each split.
    """

    x = np.asarray(features)
    y = np.asarray(targets)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be 2-D and targets 1-D with equal rows")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")

    start = 0 if anchor_slice.start is None else anchor_slice.start
    stop = x.shape[0] if anchor_slice.stop is None else min(anchor_slice.stop, x.shape[0])
    anchors = np.arange(max(start, window_size - 1), stop, dtype=np.int64)[::stride]
    if anchors.size == 0:
        return SequenceWindows(
            features=np.empty((0, window_size, x.shape[1]), dtype=x.dtype),
            targets=np.empty((0,), dtype=y.dtype),
            anchor_indices=anchors,
        )

    windows = np.stack([x[i - window_size + 1 : i + 1] for i in anchors])
    return SequenceWindows(features=windows, targets=y[anchors], anchor_indices=anchors)


def _read_float(row: dict[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column!r} is not numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"row {row_number}: {column!r} is not finite")
    return value


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a right-aligned rolling mean with NaN warm-up rows."""

    values = np.asarray(values, dtype=np.float64)
    if window < 1 or values.ndim != 1:
        raise ValueError("rolling mean requires a 1-D array and positive window")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size < window:
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Return a sample rolling standard deviation (pandas ``std`` semantics)."""

    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or values.size < window:
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
    total = cumulative[window:] - cumulative[:-window]
    total_sq = cumulative_sq[window:] - cumulative_sq[:-window]
    variance = (total_sq - total * total / window) / (window - 1)
    result[window - 1 :] = np.sqrt(np.maximum(variance, 0.0))
    return result


def _rolling_extreme(values: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 1:
        raise ValueError("rolling window must be positive")
    for index in range(window - 1, values.size):
        current = values[index - window + 1 : index + 1]
        result[index] = np.max(current) if maximum else np.min(current)
    return result


def _rolling_percentile_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Match pandas ``rolling(...).rank(pct=True)`` for the current value."""

    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 1:
        raise ValueError("rolling window must be positive")
    for index in range(window - 1, values.size):
        current_window = values[index - window + 1 : index + 1]
        current = values[index]
        less = np.count_nonzero(current_window < current)
        equal = np.count_nonzero(current_window == current)
        average_rank = (less + 1 + less + equal) / 2.0
        result[index] = average_rank / window
    return result


def _engineer_features(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the engineered columns used by the Hybrid LSTM notebook."""

    close = raw["Close"]
    high = raw["High"]
    low = raw["Low"]
    open_price = raw["Open"]
    returns = raw["Return"]
    vix = raw["VIX"]
    tnx = raw["TNX"]
    fed_rate = raw["FedRate"]

    rolling_close_60 = _rolling_mean(close, 60)
    rolling_vol_20 = _rolling_std(returns, 20)
    rolling_range_14 = _rolling_mean(high - low, 14)
    rolling_vix_rank = _rolling_percentile_rank(vix, 252)
    rolling_high_252 = _rolling_extreme(close, 252, maximum=True)
    rolling_low_252 = _rolling_extreme(close, 252, maximum=False)

    regime = np.full(close.shape, np.nan, dtype=np.float64)
    valid_regime = np.isfinite(rolling_close_60)
    regime[valid_regime] = (close[valid_regime] > rolling_close_60[valid_regime]).astype(np.float64)

    def shifted_ratio(period: int) -> np.ndarray:
        result = np.full(close.shape, np.nan, dtype=np.float64)
        if close.size > period:
            result[period:] = close[period:] / close[:-period] - 1.0
        return result

    def difference(period: int, values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape, np.nan, dtype=np.float64)
        if values.size > period:
            result[period:] = values[period:] - values[:-period]
        return result

    gap = np.full(close.shape, np.nan, dtype=np.float64)
    if close.size > 1:
        gap[1:] = open_price[1:] / close[:-1] - 1.0

    return {
        "Regime": regime,
        "VIX_percentile": rolling_vix_rank,
        "Momentum_20d": shifted_ratio(20),
        "Momentum_60d": shifted_ratio(60),
        "Momentum_252d": shifted_ratio(252),
        "Volatility_20d": rolling_vol_20,
        "ATR_14": rolling_range_14 / close,
        "Yield_slope": tnx - fed_rate,
        "Dist_52w_high": close / rolling_high_252 - 1.0,
        "Dist_52w_low": close / rolling_low_252 - 1.0,
        "VIX_change5": difference(5, vix),
        "Gap": gap,
        "FedRate_chg20": difference(20, fed_rate),
        "FedRate_chg60": difference(60, fed_rate),
    }


def load_market_csv(
    path: str | Path,
    *,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    horizon: int = 20,
) -> MarketFrame:
    """Load the multivariate CSV and compute a close-to-close label.

    Engineered columns are computed from observations available at or before
    each row.  Their warm-up NaNs are intentionally retained here and removed
    by :func:`prepare_dataset`, so the raw frame remains auditable.
    """

    csv_path = Path(path)
    requested_features = tuple(feature_cols)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        engineered = set(ENGINEERED_FEATURE_COLS)
        engineered_dependencies = {"High", "Low", "Open", "VIX", "TNX", "FedRate", "Return"}
        raw_requested = {name for name in requested_features if name not in engineered}
        required = {"Date", "Close"} | raw_requested
        if engineered.intersection(requested_features):
            required |= engineered_dependencies
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

        dates: list[str] = []
        raw_rows: dict[str, list[float]] = {name: [] for name in required if name != "Date"}
        previous_date: date | None = None
        for row_number, row in enumerate(reader, start=2):
            raw_date = row.get("Date", "")
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(f"row {row_number}: invalid ISO date {raw_date!r}") from exc
            if previous_date is not None and parsed_date <= previous_date:
                raise ValueError("dates must be strictly increasing")
            previous_date = parsed_date
            dates.append(raw_date)
            for column in raw_rows:
                raw_rows[column].append(_read_float(row, column, row_number))

    raw = {column: np.asarray(values, dtype=np.float64) for column, values in raw_rows.items()}
    close_array = raw["Close"]
    if close_array.size <= horizon:
        raise ValueError("dataset must contain more rows than the prediction horizon")
    engineered_values = _engineer_features(raw) if engineered.intersection(requested_features) else {}
    feature_values: list[np.ndarray] = []
    for column in requested_features:
        if column in raw:
            feature_values.append(raw[column])
        elif column in engineered_values:
            feature_values.append(engineered_values[column])
        else:
            raise ValueError(f"unsupported feature column: {column!r}")

    forward_returns = np.full(close_array.shape, np.nan, dtype=np.float64)
    forward_returns[:-horizon] = close_array[horizon:] / close_array[:-horizon] - 1.0
    return MarketFrame(
        dates=tuple(dates),
        close=close_array,
        features=np.column_stack(feature_values),
        feature_names=requested_features,
        forward_returns=forward_returns,
    )


def prepare_dataset(
    path: str | Path,
    *,
    horizon: int = 20,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
) -> PreparedDataset:
    """Create the canonical dataset using train-fitted target/scaler state."""

    frame = load_market_csv(path, feature_cols=feature_cols, horizon=horizon)
    valid_rows = np.isfinite(frame.features).all(axis=1) & np.isfinite(frame.forward_returns)
    if valid_rows.sum() < 3:
        raise ValueError("dataset has fewer than three rows with finite features and labels")
    usable_frame_features = frame.features[valid_rows]
    usable_forward_returns = frame.forward_returns[valid_rows]
    usable_dates = tuple(date_value for date_value, is_valid in zip(frame.dates, valid_rows) if is_valid)
    usable_close = frame.close[valid_rows]
    usable = usable_frame_features.shape[0]
    splits = chronological_slices(
        usable,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    threshold = fit_target_threshold(usable_forward_returns, splits.train)
    targets = make_forward_target(usable_forward_returns, threshold)
    scaler = fit_standardizer(usable_frame_features, splits.train)
    scaled_features = scaler.transform(usable_frame_features)
    return PreparedDataset(
        dates=usable_dates,
        close=usable_close,
        features=scaled_features,
        feature_names=frame.feature_names,
        forward_returns=usable_forward_returns,
        targets=targets,
        splits=splits,
        threshold=threshold,
        scaler=scaler,
        horizon=horizon,
        raw_rows=frame.close.shape[0],
        dropped_rows=frame.close.shape[0] - usable,
    )
