import os
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "data/processed/baselines_summary.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_total_time(df):
    df = df.sort_values("total_time_s", ascending=True).reset_index(drop=True)

    # Short cleaner labels
    label_map = {
        "No pit": "No pit",
        "Best 1-stop (BF)": "Best 1-stop",
        "Best 2-stop (BF, gap>=16)": "Best 2-stop",
        "DQN (best checkpoint)": "DQN"
    }

    df["label"] = df["strategy"].map(label_map)

    colors = ["steelblue"] * len(df)
    for i, row in df.iterrows():
        if row["label"] == "DQN":
            colors[i] = "darkorange"

    plt.figure(figsize=(8,5))
    plt.bar(df["label"], df["total_time_s"], color=colors)

    plt.ylabel("Total race time (s)")
    plt.title("Total race time by strategy")
    plt.grid(axis="y", alpha=0.3)

    # Just show time value
    for i, row in df.iterrows():
        plt.text(i, row["total_time_s"],
                 f"{row['total_time_s']:.1f}s",
                 ha="center", va="bottom")

    # Zoom y-axis slightly so differences are visible (not exaggerated)
    plt.ylim(5050, 5550)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "results_total_time_clean.png"), dpi=300)
    plt.close()


def plot_time_saved(df):
    no_pit = float(df.loc[df["strategy"] == "No pit", "total_time_s"].iloc[0])
    df["time_saved"] = no_pit - df["total_time_s"]

    df2 = df[df["strategy"] != "No pit"].copy()
    df2 = df2.sort_values("time_saved", ascending=False).reset_index(drop=True)

    label_map = {
        "Best 1-stop (BF)": "Best 1-stop",
        "Best 2-stop (BF, gap>=16)": "Best 2-stop",
        "DQN (best checkpoint)": "DQN"
    }

    df2["label"] = df2["strategy"].map(label_map)

    colors = ["steelblue"] * len(df2)
    for i, row in df2.iterrows():
        if row["label"] == "DQN":
            colors[i] = "darkorange"

    plt.figure(figsize=(8,5))
    plt.bar(df2["label"], df2["time_saved"], color=colors)

    plt.ylabel("Time saved vs no-pit (s)")
    plt.title("Benefit of strategy vs no-pit baseline")
    plt.grid(axis="y", alpha=0.3)

    for i, row in df2.iterrows():
        plt.text(i, row["time_saved"],
                 f"+{row['time_saved']:.1f}s",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "results_time_saved_clean.png"), dpi=300)
    plt.close()


def main():
    df = pd.read_csv(IN_PATH)
    plot_total_time(df.copy())
    plot_time_saved(df.copy())
    print("Saved clean comparison figures.")


if __name__ == "__main__":
    main()