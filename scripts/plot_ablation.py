import os
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "data/processed/ablation_summary.csv"
OUT_DIR = "reports/figures"
OUT_FNAME = "ablation_comparison.png"

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(IN_PATH)

    # cleaner labels
    label_map = {
        "full_system": "Full system",
        "no_reward_shaping": "No reward shaping",
        "no_min_stint": "No min stint",
    }
    df["label"] = df["variant"].map(label_map)

    df = df.sort_values("best_time", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df["best_time"])

    plt.ylabel("Best greedy total time (s)")
    plt.title("Ablation study: effect of key design choices")

    # ---- Zoom the y-axis ----
    ymin = df["best_time"].min() - 2
    ymax = df["best_time"].max() + 2
    plt.ylim(ymin, ymax)

    plt.grid(axis="y", alpha=0.3)

    for i, row in df.iterrows():
        plt.text(
            i,
            row["best_time"] + 0.2,
            f"{row['best_time']:.3f}s",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_FNAME)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
