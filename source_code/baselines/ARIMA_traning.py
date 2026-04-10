from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

_DATASETS_ALIGNED_DIR = Path(__file__).resolve().parent.parent.parent / "datasets_aligned"
TRAIN_RATIO = 0.8


def load_data():
    path = _DATASETS_ALIGNED_DIR / "NASDAQCOM.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["NASDAQCOM"].pct_change() * 10  # match LSTM
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = int(len(df) * TRAIN_RATIO)
    train = df["return"].iloc[:split_idx]
    test = df["return"].iloc[split_idx:]
    return train, test


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


def direction_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)

    tp = np.sum((true_dir == 1) & (pred_dir == 1))
    fp = np.sum((true_dir == 0) & (pred_dir == 1))
    fn = np.sum((true_dir == 1) & (pred_dir == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Computing returns...")
    df = get_return(df)

    print("Splitting data...")
    train, test = split_data(df)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("Training ARIMA (this may take time)...")
    model = ARIMA(train, order=(5, 0, 1)).fit()

    predictions = model.forecast(steps=len(test))
    predictions = np.asarray(predictions)

    print("Evaluating...")
    mae = mean_absolute_error(test, predictions)
    f1 = direction_f1(test, predictions)

    print("\n===== RESULTS =====")
    print(f"MAE: {mae:.6f}")
    print(f"Direction F1: {f1:.6f}")