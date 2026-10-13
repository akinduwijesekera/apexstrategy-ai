import os
import pandas as pd
import matplotlib.pyplot as plt

TRACE_CSV = "data/processed/ddqn_safety_car_eval_trace.csv"
OUT_PLOT = "data/processed/plots/ddqn_safety_car_trace.png"


def main():
    os.makedirs("data/processed/plots", exist_ok=True)

    df = pd.read_csv(TRACE_CSV)

    plt.figure(figsize=(11, 5))
    plt.plot(df["lap"], df["lap_time"], label="Lap time")

    # Shade Safety Car laps
    sc_df = df[df["sc_active"] == 1]
    for lap in sc_df["lap"]:
        plt.axvspan(lap - 0.5, lap + 0.5, alpha=0.15)

    # Mark pit stops
    pit_df = df[df["pit"] == 1]
    plt.scatter(pit_df["lap"], pit_df["lap_time"], marker="x", s=80, label="Pit stop")

    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title("DDQN Safety-Car trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()

    print(f"Saved: {OUT_PLOT}")


if __name__ == "__main__":
    main()