import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/processed/sc_sensitivity.csv"
OUT_TIME = "data/processed/plots/sc_sensitivity_time.png"
OUT_PIT = "data/processed/plots/sc_sensitivity_pit_under_sc.png"


def main():
    os.makedirs("data/processed/plots", exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    plt.figure(figsize=(8, 5))
    plt.plot(df["sc_prob_per_lap"], df["mean_total_time"], marker="o")
    plt.xlabel("Safety Car probability per lap")
    plt.ylabel("Mean total race time (s)")
    plt.title("Sensitivity: SC probability vs race time")
    plt.tight_layout()
    plt.savefig(OUT_TIME, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["sc_prob_per_lap"], df["pitted_under_sc_rate"], marker="o")
    plt.xlabel("Safety Car probability per lap")
    plt.ylabel("Pitted under SC rate")
    plt.title("Sensitivity: SC probability vs SC pit usage")
    plt.tight_layout()
    plt.savefig(OUT_PIT, dpi=200)
    plt.close()

    print("Saved:")
    print(f"  {OUT_TIME}")
    print(f"  {OUT_PIT}")


if __name__ == "__main__":
    main()
    