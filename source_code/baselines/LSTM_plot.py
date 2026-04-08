"""Plot LSTM test-set predictions vs actual daily returns (requires trained `models/Vanilla_LSTM.pth`)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
_BASELINE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINE_DIR.parent.parent
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

from LSTM_training import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    VAL_RATIO,
    WINDOW_SIZE,
    MyLSTMSequential,
    build_windows,
    get_daily_percentage_change,
)



MODEL_SAVE_PATH = _PROJECT_ROOT / "models" / "Vanilla_LSTM.pth"
DATASET_PATH = _PROJECT_ROOT / "datasets_aligned" / "NASDAQCOM.csv"


def load_test_split(dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    df = pd.read_csv(dataset_path)
    original_data = torch.tensor(df["NASDAQCOM"].values, dtype=torch.float32)
    test_start = int(len(original_data) * VAL_RATIO - WINDOW_SIZE - 1)

    data = get_daily_percentage_change(original_data)
    test_x, test_y = build_windows(data[test_start:], WINDOW_SIZE)


    dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # Align with build_windows: first y is data[test_start + WINDOW_SIZE]
    test_dates = np.array(dates[test_start + WINDOW_SIZE :])
    return test_x, test_y, test_dates

def load_model(model_save_path: Path) -> tuple[MyLSTMSequential, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyLSTMSequential(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_size=1,
    ).to(device)
    try:
        state = torch.load(model_save_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def plot_predictions(preds: np.ndarray, targets: np.ndarray, test_dates: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(test_dates, targets, label="Actual", linewidth=1.0)
    axes[0].plot(test_dates, preds, label="Predicted", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("Daily return (fraction)")
    axes[0].set_title("LSTM — test set: actual vs predicted daily return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    residual = preds - targets
    axes[1].plot(test_dates, residual, color="C2", linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Residual (pred − actual)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def main() -> None:
    if not MODEL_SAVE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model at {MODEL_SAVE_PATH}. Run LSTM_training.py from the project root first."
        )
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset at {DATASET_PATH}")

    test_x, test_y, test_dates = load_test_split(DATASET_PATH)

    model, device = load_model(MODEL_SAVE_PATH)
    pin = device.type == "cuda"

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin,
    )
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=pin)
            targets = targets.to(device, non_blocking=pin)
            outputs = model(inputs)
            all_preds.append(outputs.squeeze(-1).cpu())
            all_targets.append(targets.cpu())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)


    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    print(f"Test MSE (daily fractional return): {mse:.8f}")
    print(f"Test MAE (daily fractional return): {mae:.8f}")

    plot_predictions(preds, targets, test_dates)


if __name__ == "__main__":
    main()
