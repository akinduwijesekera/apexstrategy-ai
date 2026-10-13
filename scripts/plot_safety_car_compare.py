import os
import pandas as pd
import matplotlib.pyplot as plt

RAW_CSV = "data/processed/safety_car_compare_raw.csv"
SUMMARY_CSV = "data/processed/safety_car_compare_summary.csv"

OUT_BAR = "data/processed/plots/safety_car_compare_mean_time.png"
OUT_BOX = "data/processed/plots/safety_car_compare_boxplot.png"


def main():
    os.makedirs("data/processed/plots", exist_ok=True)

    raw_df = pd.read_csv(RAW_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)

    # Bar chart: mean total time by experiment
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["experiment"], summary_df["mean_total_time"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean total race time (s)")
    plt.title("Safety Car comparison: mean race time")
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=200)
    plt.close()

    # Boxplot: total_time distribution by experiment
    plt.figure(figsize=(10, 5))
    raw_df.boxplot(column="total_time", by="experiment", grid=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Total race time (s)")
    plt.title("Safety Car comparison: race-time distribution")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT_BOX, dpi=200)
    plt.close()

    print("Saved:")
    print(f"  {OUT_BAR}")
    print(f"  {OUT_BOX}")


if __name__ == "__main__":
    main()