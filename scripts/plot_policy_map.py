import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import SimConfig
from env.race_env import COMPOUND_TO_ONEHOT


MODEL_PATH = "models/dqn_qnet_best.pt"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)


class QNet(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def build_state(lap, tyre_life, total_laps, compound="MEDIUM", last_pred=0.0, slope=0.0):
    lap_frac = lap / total_laps
    laps_remaining_frac = (total_laps - lap) / total_laps
    tyre_life_norm = min(tyre_life, 60) / 60.0
    comp_oh = COMPOUND_TO_ONEHOT[compound]

    return np.array(
        [lap_frac, laps_remaining_frac, tyre_life_norm, *comp_oh, last_pred, slope],
        dtype=np.float32
    )


def action_name(a):
    return {
        0: "STAY",
        1: "PIT_SOFT",
        2: "PIT_MEDIUM",
        3: "PIT_HARD"
    }[a]


def main():
    config = SimConfig()
    total_laps = config.total_laps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNet(obs_dim=8, n_actions=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Grid for visualisation
    lap_values = np.arange(1, total_laps + 1)
    tyre_life_values = np.arange(1, 41)  # enough range for plotting

    action_grid = np.zeros((len(tyre_life_values), len(lap_values)), dtype=np.int32)
    qdiff_grid = np.zeros((len(tyre_life_values), len(lap_values)), dtype=np.float32)

    for i, tyre_life in enumerate(tyre_life_values):
        for j, lap in enumerate(lap_values):
            state = build_state(
                lap=lap,
                tyre_life=tyre_life,
                total_laps=total_laps,
                compound="MEDIUM",
                last_pred=0.0,
                slope=0.0,
            )

            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                qvals = model(s).cpu().numpy()[0]

            best_action = int(np.argmax(qvals))
            action_grid[i, j] = best_action

            # Optional interpretability metric:
            # best pit action value minus stay-out value
            best_pit_q = np.max(qvals[1:])
            qdiff_grid[i, j] = best_pit_q - qvals[0]

    # -----------------------------
    # Plot 1: Action heatmap
    # -----------------------------
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        action_grid,
        origin="lower",
        aspect="auto",
        extent=[lap_values.min(), lap_values.max(), tyre_life_values.min(), tyre_life_values.max()],
    )
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["STAY", "PIT SOFT", "PIT MEDIUM", "PIT HARD"])

    plt.xlabel("Lap")
    plt.ylabel("Tyre life (laps)")
    plt.title("DQN policy map (compound fixed to MEDIUM)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "policy_map_actions.png"), dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: Pit preference strength
    # -----------------------------
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        qdiff_grid,
        origin="lower",
        aspect="auto",
        extent=[lap_values.min(), lap_values.max(), tyre_life_values.min(), tyre_life_values.max()],
    )
    plt.colorbar(im, label="max(Q(pit)) - Q(stay)")
    plt.xlabel("Lap")
    plt.ylabel("Tyre life (laps)")
    plt.title("Pit preference strength (positive means pit is preferred)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "policy_map_qdiff.png"), dpi=300)
    plt.close()

    print(f"Saved policy visualisation figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()