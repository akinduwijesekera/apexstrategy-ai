import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "data/processed/sensitivity_summary.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_total_times(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["pit_loss_seconds"], df["best_2stop_time"], marker="o", label="Best 2-stop (BF)")
    plt.plot(df["pit_loss_seconds"], df["dqn_time"], marker="o", label="DQN")

    plt.xlabel("Pit loss (s)")
    plt.ylabel("Total race time (s)")
    plt.title("Sensitivity analysis: effect of pit loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sensitivity_pit_loss_times.png"), dpi=300)
    plt.close()


def plot_gap_to_best(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["pit_loss_seconds"].astype(str), df["dqn_gap_to_best_2stop"])

    plt.xlabel("Pit loss (s)")
    plt.ylabel("DQN gap to best 2-stop (s)")
    plt.title("Sensitivity analysis: DQN robustness vs pit loss")
    plt.grid(axis="y", alpha=0.3)

    for i, row in df.iterrows():
        plt.text(i, row["dqn_gap_to_best_2stop"], f"{row['dqn_gap_to_best_2stop']:.3f}s", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sensitivity_pit_loss_gap.png"), dpi=300)
    plt.close()


def main():
    df = pd.read_csv(IN_PATH)

    plot_total_times(df)
    plot_gap_to_best(df)

    print(f"Saved sensitivity plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()