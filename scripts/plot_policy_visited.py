import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

MODEL_PATH = "models/dqn_qnet_best.pt"
TRACE_PATH = "data/processed/dqn_eval_trace.csv"
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


ACTION_LABELS = {
    0: "STAY",
    1: "PIT SOFT",
    2: "PIT MEDIUM",
    3: "PIT HARD",
}


def main():
    df = pd.read_csv(TRACE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNet(obs_dim=8, n_actions=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    state_cols = [
        "state_lap_frac",
        "state_laps_remaining_frac",
        "state_tyre_life_norm",
        "state_is_soft",
        "state_is_medium",
        "state_is_hard",
        "state_last_pred",
        "state_slope",
    ]

    states = df[state_cols].to_numpy(dtype=np.float32)

    qdiffs = []
    greedy_actions = []

    with torch.no_grad():
        for s in states:
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            qvals = model(s_t).cpu().numpy()[0]

            greedy_action = int(np.argmax(qvals))
            best_pit_q = np.max(qvals[1:])
            qdiff = best_pit_q - qvals[0]

            greedy_actions.append(greedy_action)
            qdiffs.append(qdiff)

    df["greedy_action"] = greedy_actions
    df["qdiff_pit_vs_stay"] = qdiffs

    pit_df = df[df["pit"] == 1].copy()

    # Use symmetric-ish limits around zero so the pit-vs-stay boundary is easier to see
    qmax = max(abs(df["qdiff_pit_vs_stay"].min()), abs(df["qdiff_pit_vs_stay"].max()))
    qlim = max(5.0, qmax)

    # ----------------------------------
    # Plot 1: visited states coloured by greedy action
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    # faint trajectory line
    ax.plot(df["lap"], df["tyre_life"], alpha=0.25, linewidth=1)

    # discrete colour mapping
    cmap_actions = plt.get_cmap("viridis", 4)
    norm_actions = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    sc = ax.scatter(
        df["lap"],
        df["tyre_life"],
        c=df["greedy_action"],
        cmap=cmap_actions,
        norm=norm_actions,
        s=120,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["STAY", "PIT SOFT", "PIT MEDIUM", "PIT HARD"])
    cbar.set_label("Greedy action")

    if not pit_df.empty:
        ax.scatter(
            pit_df["lap"],
            pit_df["tyre_life"],
            facecolors="none",
            edgecolors="red",
            s=380,
            linewidths=2.5,
            label="Actual pit events",
            zorder=4,
        )
        ax.legend()

    ax.set_xlabel("Lap")
    ax.set_ylabel("Tyre life (laps)")
    ax.set_title("Visited DQN states coloured by greedy action")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "policy_visited_actions.png"), dpi=300)
    plt.close(fig)

    # ----------------------------------
    # Plot 2: visited states coloured by qdiff
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(df["lap"], df["tyre_life"], alpha=0.25, linewidth=1)

    sc = ax.scatter(
        df["lap"],
        df["tyre_life"],
        c=df["qdiff_pit_vs_stay"],
        cmap="coolwarm",
        vmin=-qlim,
        vmax=qlim,
        s=120,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("max(Q(pit)) - Q(stay)")

    if not pit_df.empty:
        ax.scatter(
            pit_df["lap"],
            pit_df["tyre_life"],
            facecolors="none",
            edgecolors="red",
            s=380,
            linewidths=2.5,
            label="Actual pit events",
            zorder=4,
        )
        ax.legend()

    ax.set_xlabel("Lap")
    ax.set_ylabel("Tyre life (laps)")
    ax.set_title("Visited DQN states coloured by pit preference strength")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "policy_visited_qdiff.png"), dpi=300)
    plt.close(fig)

        # ----------------------------------
    # Plot 3: compound-separated qdiff
    # ----------------------------------
    compounds = ["SOFT", "MEDIUM", "HARD"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    scatter_handle = None

    for ax, compound in zip(axes, compounds):
        sub = df[df["compound"] == compound].copy()

        if not sub.empty:
            ax.plot(sub["lap"], sub["tyre_life"], alpha=0.25, linewidth=1)

            scatter_handle = ax.scatter(
                sub["lap"],
                sub["tyre_life"],
                c=sub["qdiff_pit_vs_stay"],
                cmap="coolwarm",
                vmin=-qlim,
                vmax=qlim,
                s=120,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )

            pit_sub = sub[sub["pit"] == 1]
            if not pit_sub.empty:
                ax.scatter(
                    pit_sub["lap"],
                    pit_sub["tyre_life"],
                    facecolors="none",
                    edgecolors="red",
                    s=380,
                    linewidths=2.5,
                    zorder=4,
                )

        ax.set_title(compound)
        ax.set_xlabel("Lap")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Tyre life (laps)")
    fig.suptitle("Visited DQN states: pit preference by compound", y=1.02)

    # Leave room on the right for a dedicated colorbar
    fig.subplots_adjust(right=0.88, wspace=0.28)

    if scatter_handle is not None:
        cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.68])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter_handle, cax=cbar_ax)
        cbar.set_label("max(Q(pit)) - Q(stay)")

    fig.savefig(
        os.path.join(OUT_DIR, "policy_visited_by_compound.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved policy visualisation figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
