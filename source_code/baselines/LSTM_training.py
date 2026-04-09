import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
torch.backends.cudnn.enabled = False

BATCH_SIZE = 128
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
WINDOW_SIZE = 30
EARLY_STOP_PATIENCE = 10
MODEL_SAVE_PATH = "models/Vanilla_LSTM.pth"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.9


class MyLSTMSequential(nn.Module):

    lstm : nn.LSTM
    linear : nn.Linear
    lstm_cells : list[nn.LSTMCell]
    hidden_size : int
    num_layers : int

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.lstm_cells = [nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            pass
        else:
            raise ValueError(f"Input shape must be 2 or 3, got {x.shape} and it should be (batch_size, sequence_length, input_size)")
        

        
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x


def build_windows(data: torch.Tensor, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sliding-window (X, y) pairs. y is the value immediately after the window (one-step-ahead)."""
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return (
        torch.tensor(np.array(x), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )

def get_daily_percentage_change(data: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(len(data))
    for i in range(1, len(data)):
        result[i] = ((data[i] - data[i - 1]) / data[i - 1]) * 10
    result[0] = 0
    return result

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


if __name__ == "__main__":


    device = get_device()

    df = pd.read_csv("datasets_aligned/NASDAQCOM.csv")
    data = torch.tensor(df["NASDAQCOM"].values, dtype=torch.float32)
    data = get_daily_percentage_change(data)
    print(data)
    print(data.shape)
    train_end = int(len(data) * TRAIN_RATIO)
    val_start = train_end - WINDOW_SIZE - 1
    val_end = int(len(data) * VAL_RATIO)
    test_start = val_end - WINDOW_SIZE - 1
    
    
    train_x, train_y = build_windows(data[:train_end], WINDOW_SIZE)
    val_x, val_y = build_windows(data[val_start:val_end], WINDOW_SIZE)
    test_x, test_y = build_windows(data[test_start:], WINDOW_SIZE)


    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MyLSTMSequential(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    no_improve_count = 0
    epoch = 0
    while no_improve_count < EARLY_STOP_PATIENCE:
        # training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1), targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
        epoch += 1

    print(f"training completed in {epoch} epochs")
    print(f"best validation loss: {best_val_loss}")
    print(f"best model saved to {MODEL_SAVE_PATH}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(best_model_state, MODEL_SAVE_PATH)
