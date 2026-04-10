from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

_BASELINE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINE_DIR.parent.parent

DATASET_PATH = _PROJECT_ROOT / "datasets_aligned" / "NASDAQCOM.csv"
TRAIN_RATIO = 0.8


def load_data(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_daily_percentage_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["NASDAQCOM"].pct_change() * 10
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    split_idx = int(len(df) * train_ratio)

    train = df["return"].iloc[:split_idx]
    test = df["return"].iloc[split_idx:]
    test_dates = df["date"].iloc[split_idx:]

    return train, test, test_dates

def walk_forward_arima(train, test, order=(5, 0, 1)):
    history = list(train)
    predictions = []

    for actual in test:
        model = ARIMA(history, order=order)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=1)
        pred = float(forecast[0])

        predictions.append(pred)
        history.append(actual)

    return predictions


def calculate_directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    correct = 0
    for pred, target in zip(preds, targets):
        if (pred > 0 and target > 0) or (pred < 0 and target < 0):
            correct += 1
    return correct / len(preds)


def calculate_mean_squared_error(preds: np.ndarray, targets: np.ndarray) -> float:
    return np.mean(np.square(preds - targets))


def direction_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    pred_dir = (preds > 0).astype(int)
    true_dir = (targets > 0).astype(int)

    tp = np.sum((pred_dir == 1) & (true_dir == 1))
    fp = np.sum((pred_dir == 1) & (true_dir == 0))
    fn = np.sum((pred_dir == 0) & (true_dir == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def plot_predictions(preds: np.ndarray, targets: np.ndarray, test_dates: np.ndarray) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(test_dates, targets, label="Actual", linewidth=1.0)
    axes[0].plot(test_dates, preds, label="Predicted", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("Daily return (*10 scaled)")
    axes[0].set_title("ARIMA — test set: actual vs predicted daily return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residual = preds - targets
    axes[1].plot(test_dates, residual, linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset at {DATASET_PATH}")

    print("Loading data...")
    df = load_data(DATASET_PATH)

    print("Computing returns...")
    df = get_daily_percentage_change(df)

    print("Splitting data...")
    train, test, test_dates = split_data(df)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("Training ARIMA...")
    model = ARIMA(train, order=(1, 0, 1)).fit()

    print("Forecasting...")
    predictions = model.forecast(steps=len(test))
    predictions = np.asarray(predictions)
    # predictions = walk_forward_arima(train, test, order=(5, 0, 1))
    targets = np.asarray(test)
    test_dates = np.asarray(test_dates)

    mse = calculate_mean_squared_error(predictions, targets)
    mae = mean_absolute_error(targets, predictions)
    da = calculate_directional_accuracy(predictions, targets)
    f1 = direction_f1(predictions, targets)

    print(f"Test MSE: {mse:.8f}")
    print(f"Test MAE: {mae:.8f}")
    print(f"Test directional accuracy: {da:.8f}")
    print(f"Test direction F1: {f1:.8f}")

    plot_predictions(predictions, targets, test_dates)


if __name__ == "__main__":
    main()