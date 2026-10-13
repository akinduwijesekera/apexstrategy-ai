import os
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "data/processed/baselines_summary.csv"
OUT_DIR = "reports/figures"
OUT_FNAME = "results_time_saved_vs_no_pit.png"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_PATH)

    no_pit = float(df.loc[df["strategy"] == "No pit", "total_time_s"].iloc[0])
    df["time_saved_s"] = no_pit - df["total_time_s"]

    # drop no-pit bar (it would be 0 saved)
    df2 = df[df["strategy"] != "No pit"].copy()
    df2 = df2.sort_values("time_saved_s", ascending=False).reset_index(drop=True)

    plt.figure()
    plt.bar(df2["strategy"], df2["time_saved_s"])
    plt.ylabel("Time saved vs no-pit (s)")
    plt.title("Benefit of pitting strategies vs no-pit baseline")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for i, row in df2.iterrows():
        s = float(row["time_saved_s"])
        pits = "" if pd.isna(row.get("pits")) else str(row.get("pits"))
        label = f"+{s:.3f}s"
        if pits:
            label += f"\n{pits}"
        plt.text(i, s, label, ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, OUT_FNAME)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()