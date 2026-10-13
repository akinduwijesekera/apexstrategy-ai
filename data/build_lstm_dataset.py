import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_SIZE = 10

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}


def compound_onehot(compound: str) -> np.ndarray:
    v = np.zeros(3, dtype=np.float32)
    idx = COMPOUND_MAP.get(str(compound).upper(), None)
    if idx is not None:
        v[idx] = 1.0
    return v


def build_sequences_from_stint(df_stint: pd.DataFrame, window_size: int):
    """
    Build (X, y) for one driver stint.
    X: (num_samples, window, 5)
    y: (num_samples,)
    """
    # Ensure sorted by lap
    df_stint = df_stint.sort_values("LapNumber").reset_index(drop=True)

    # --- Features ---
    lap_norm = df_stint["LapTimeNorm"].to_numpy(dtype=np.float32)

    # TyreLife can have NaNs; fill with forward fill then 0
    tyre_life = df_stint["TyreLife"].fillna(method="ffill").fillna(0).to_numpy(dtype=np.float32)
    # Normalise tyre life (cap to avoid weird spikes)
    tyre_life = np.clip(tyre_life, 0, 60)
    tyre_life_norm = tyre_life / 60.0  # 0..1

    # Compound one-hot (usually constant within a stint but we compute per row anyway)
    comp_oh = np.stack([compound_onehot(c) for c in df_stint["Compound"].values], axis=0)  # (T, 3)

    # Combine into feature matrix per timestep: (T, 5)
    feats = np.column_stack([lap_norm, tyre_life_norm, comp_oh]).astype(np.float32)

    X, y = [], []
    T = len(df_stint)
    for i in range(T - window_size):
        X.append(feats[i:i + window_size])
        y.append(lap_norm[i + window_size])  # predict next LapTimeNorm

    return X, y


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_X, all_y = [], []

    csv_files = sorted(RAW_DIR.glob("2023_*_race_laps.csv"))
    if not csv_files:
        raise FileNotFoundError("No 2023_*_race_laps.csv files found in data/raw/")

    for file in csv_files:
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        # --- Normalise within race (remove track baseline) ---
        race_median = df["LapTimeSeconds"].median()
        df["LapTimeNorm"] = df["LapTimeSeconds"] - race_median

        # Group by driver and stint (degradation resets after pit stop)
        for (_, _), df_stint in df.groupby(["Driver", "Stint"]):
            if len(df_stint) > WINDOW_SIZE:
                X_stint, y_stint = build_sequences_from_stint(df_stint, WINDOW_SIZE)
                all_X.extend(X_stint)
                all_y.extend(y_stint)

    X = np.array(all_X, dtype=np.float32)  # (N, window, 5)
    y = np.array(all_y, dtype=np.float32)  # (N,)

    print("Final dataset shape:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y stats: mean", float(y.mean()), "std", float(y.std()), "min", float(y.min()), "max", float(y.max()))

    np.save(PROCESSED_DIR / "X_lstm.npy", X)
    np.save(PROCESSED_DIR / "y_lstm.npy", y)
    print("Saved processed LSTM dataset.")


if __name__ == "__main__":
    main()