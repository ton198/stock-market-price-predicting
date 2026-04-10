from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

_DATASETS_ALIGNED_DIR = Path(__file__).resolve().parent.parent.parent / "datasets_aligned"

TRAIN_RATIO = 0.8

ARIMA_ORDER = (5, 0, 1)
ROLLING_WINDOW_SIZE = 1000
MAX_TEST_POINTS = 500   

def load_data() -> pd.DataFrame:
    path = _DATASETS_ALIGNED_DIR / "NASDAQCOM.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["NASDAQCOM"].pct_change() * 10
    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> tuple[pd.Series, pd.Series, pd.Series]:
    split_idx = int(len(df) * train_ratio)
    train = df["return"].iloc[:split_idx].reset_index(drop=True)
    test = df["return"].iloc[split_idx:].reset_index(drop=True)
    test_dates = df["date"].iloc[split_idx:].reset_index(drop=True)

    return train, test, test_dates


def restrict_test(test, dates):
    if len(test) > MAX_TEST_POINTS:
        test = test.iloc[-MAX_TEST_POINTS:].reset_index(drop=True)
        dates = dates.iloc[-MAX_TEST_POINTS:].reset_index(drop=True)
    return test, dates


def walk_forward(train, test):
    history = list(train)
    predictions = []

    for i, actual in enumerate(test):
        hist = history[-ROLLING_WINDOW_SIZE:]

        try:
            model = ARIMA(hist, order=ARIMA_ORDER)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            pred = float(np.asarray(forecast)[0])
        except:
            pred = float(np.mean(hist))

        predictions.append(pred)
        history.append(float(actual))

        if (i + 1) % 50 == 0 or (i + 1) == len(test):
            print(f"Processed {i + 1}/{len(test)} test points")

    return np.array(predictions)


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(true_dir == pred_dir))


def direction_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)

    tp = np.sum((true_dir == 1) & (pred_dir == 1))
    fp = np.sum((true_dir == 0) & (pred_dir == 1))
    fn = np.sum((true_dir == 1) & (pred_dir == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Computing returns...")
    df = get_return(df)

    print("Splitting data...")
    train, test, test_dates = split_data(df)

    print(f"Original train size: {len(train)}")
    print(f"Original test size: {len(test)}")

    test, test_dates = restrict_test(test, test_dates)

    print(f"Using rolling window size: {ROLLING_WINDOW_SIZE}")
    print(f"Using test size: {len(test)}")
    print(f"ARIMA order: {ARIMA_ORDER}")

    print("Training ARIMA with walk-forward rolling window...")
    predictions = walk_forward(train, test)

    y_true = np.asarray(test, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)

    print("Evaluating...")
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    da = direction_accuracy(y_true, y_pred)
    f1 = direction_f1(y_true, y_pred)

    print("\n===== RESULTS =====")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"Direction Accuracy: {da:.8f}")
    print(f"Direction F1: {f1:.8f}")

    results_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_true,
        "predicted": y_pred,
    })

    output_dir = Path(__file__).resolve().parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "arima_v2_walk_forward_predictions.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")