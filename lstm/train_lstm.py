from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
SEED = 42


class LapTimeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len) -> convert to (N, seq_len, 1) for LSTM
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # last timestep: (B, H)
        return self.fc(last)           # (B, 1)


def train_val_split(X, y, val_ratio=0.2):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = int(n * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X = np.load(DATA_DIR / "X_lstm.npy")
    y = np.load(DATA_DIR / "y_lstm.npy")

    X_train, y_train, X_val, y_val = train_val_split(X, y)

    # Baseline: predict next lap time = last lap time in the window
    baseline_pred = X_val[:, -1, 0]  # last LapTimeNorm (feature 0)
    baseline_mse = np.mean((baseline_pred - y_val) ** 2)
    print(f"Baseline (last-lap) Val MSE: {baseline_mse:.4f}")

    train_ds = LapTimeDataset(X_train, y_train)
    val_ds = LapTimeDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = LSTMRegressor(hidden_size=64, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_DIR / "lstm_laptime_regressor.pt")
    print("Saved model to models/lstm_laptime_regressor.pt")


if __name__ == "__main__":
    main()