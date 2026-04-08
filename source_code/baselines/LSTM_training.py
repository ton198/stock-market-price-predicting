import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 0.2
WINDOW_SIZE = 30


class MyLSTMSequential(nn.Module):

    lstm : nn.LSTM
    linear : nn.Linear

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

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



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    df = pd.read_csv("datasets_aligned/NASDAQCOM.csv")
    original_data = torch.tensor(df["NASDAQCOM"].values, dtype=torch.float32)
    print(original_data)
    print(original_data.shape)
    
    training_x_samples : list[torch.Tensor] = []
    training_y_samples : list[torch.Tensor] = []
    validation_x_samples : list[torch.Tensor] = []
    validation_y_samples : list[torch.Tensor] = []
    
    for i in range(len(original_data) - WINDOW_SIZE - 1):
        if i % 10 == 9:
            validation_x_samples.append(original_data[i:i + WINDOW_SIZE])
            validation_y_samples.append(original_data[i + WINDOW_SIZE + 1])
        else:
            training_x_samples.append(original_data[i:i + WINDOW_SIZE])
            training_y_samples.append(original_data[i + WINDOW_SIZE + 1])
    training_x_batch = torch.stack(training_x_samples)
    training_y_batch = torch.stack(training_y_samples)
    validation_x_batch = torch.stack(validation_x_samples)
    validation_y_batch = torch.stack(validation_y_samples)


    pin = device.type == "cuda"
    train_data = DataLoader(
        TensorDataset(training_x_batch, training_y_batch),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin,
    )
    validation_data = DataLoader(
        TensorDataset(validation_x_batch, validation_y_batch),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin,
    )

    model = MyLSTMSequential(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    best_model_state_dict = None
    
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        for inputs, targets in train_data:
            inputs = inputs.to(device, non_blocking=pin)
            targets = targets.to(device, non_blocking=pin)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_data:
                inputs = inputs.to(device, non_blocking=pin)
                targets = targets.to(device, non_blocking=pin)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1), targets)
                validation_loss += loss.item()
        validation_loss /= len(validation_data)
        print(f"Epoch {epoch}, Validation Loss: {validation_loss}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    os.makedirs("models", exist_ok=True)
    torch.save(best_model_state_dict, "models/Vanilla_LSTM.pth")
