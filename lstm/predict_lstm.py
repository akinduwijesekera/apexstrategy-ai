from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = Path("models/lstm_laptime_regressor.pt")


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, input_size=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # (B, T, H)
        last = out[:, -1, :]       # (B, H)
        return self.fc(last)       # (B, 1)


def load_model(device=None):
    """Load the trained LSTM model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(hidden_size=64, num_layers=1, input_size=5).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


@torch.no_grad()
def predict_next_laptime_norm(window_10x5: np.ndarray, model=None, device=None) -> float:
    """
    Predict next LapTimeNorm given a single input window of shape (10, 5).
    Returns a float prediction (seconds, normalised).
    """
    if model is None or device is None:
        model, device = load_model(device=device)

    x = torch.tensor(window_10x5, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 10, 5)
    pred = model(x).cpu().numpy().reshape(-1)[0]
    return float(pred)


def main():
    # Quick test: load one sample from processed dataset and predict
    X = np.load("data/processed/X_lstm.npy")
    y = np.load("data/processed/y_lstm.npy")

    model, device = load_model()
    sample = X[0]  # (10, 5)
    pred = predict_next_laptime_norm(sample, model=model, device=device)

    print("Pred:", pred)
    print("True:", float(y[0]))


if __name__ == "__main__":
    main()