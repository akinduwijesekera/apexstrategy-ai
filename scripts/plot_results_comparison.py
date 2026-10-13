import os
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "data/processed/baselines_summary.csv"
OUT_DIR = "reports/figures"
OUT_FNAME = "results_total_time_comparison.png"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_PATH)
    df = df.sort_values("total_time_s", ascending=True).reset_index(drop=True)

    best = float(df["total_time_s"].min())
    df["delta_vs_best_s"] = df["total_time_s"] - best
    df["pct_vs_best"] = 100.0 * df["delta_vs_best_s"] / best

    plt.figure()
    plt.bar(df["strategy"], df["total_time_s"])
    plt.ylabel("Total race time (s)")
    plt.title("Strategy comparison (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for i, row in df.iterrows():
        t = float(row["total_time_s"])
        d = float(row["delta_vs_best_s"])
        p = float(row["pct_vs_best"])
        pits = "" if pd.isna(row.get("pits")) else str(row.get("pits"))

        label = f"{t:.3f}s"
        if pits:
            label += f"\n{pits}"
        if d > 1e-9:
            label += f"\n(+{d:.3f}s, {p:.3f}%)"

        plt.text(i, t, label, ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_FNAME)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()