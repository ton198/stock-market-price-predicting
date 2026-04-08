import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 0.2
WINDOW_SIZE = 30


class MyLSTMSequential(nn.Module):

    layers : nn.ModuleList

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout),
            nn.Linear(hidden_size, output_size)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



if __name__ == "__main__":
    df = pd.read_csv("datasets_aligned/NASDAQCOM.csv")
    original_data = torch.tensor(df["NASDAQCOM"].values, dtype=torch.float32)
    print(original_data)
    print(original_data.shape)
    
    x_data : list[torch.Tensor] = []
    y_data : list[torch.Tensor] = []
    
    for i in range(len(original_data) - WINDOW_SIZE - 1):
        torch_x = original_data[i:i + WINDOW_SIZE]
        torch_y = original_data[i + WINDOW_SIZE + 1]

        x_data.append(torch_x)
        y_data.append(torch_y)

    batched_data = DataLoader(TensorDataset(torch.stack(x_data), torch.stack(y_data)), batch_size=BATCH_SIZE, shuffle=True)

    
    model = MyLSTMSequential(input_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    for epoch in range(EPOCHS):
        for inputs, targets in batched_data:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
