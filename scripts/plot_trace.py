import os
import pandas as pd
import matplotlib.pyplot as plt

TRACE_PATH = "data/processed/dqn_eval_trace.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_trace(path=TRACE_PATH):
    df = pd.read_csv(path)

    # Make lap column consistent (you already have "lap", but keep this for robustness)
    if "lap" not in df.columns:
        if "lap_idx" in df.columns:
            df = df.rename(columns={"lap_idx": "lap"})
        elif "Lap" in df.columns:
            df = df.rename(columns={"Lap": "lap"})

    # Ensure sorting just in case
    df = df.sort_values("lap").reset_index(drop=True)
    return df


def pit_laps(df):
    if "pit" not in df.columns:
        return []
    pits = df.loc[df["pit"].astype(bool), "lap"].tolist()
    return sorted(set(int(x) for x in pits))


def plot_laptime(df, title="Lap time vs Lap", fname="lap_time_vs_lap.png"):
    if "lap_time" not in df.columns:
        print("No lap_time column found, skipping lap time plot.")
        return

    pits = pit_laps(df)
    plt.figure()
    plt.plot(df["lap"], df["lap_time"])

    for pl in pits:
        plt.axvline(pl, linestyle="--", alpha=0.6, linewidth=1)

    # Auto-tighten y-axis for cleaner visuals
    ymin = df["lap_time"].min() - 1
    ymax = df["lap_time"].max() + 1
    plt.ylim(ymin, ymax)

    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


def plot_tyre_life(df, title="Tyre life vs Lap", fname="tyre_life_vs_lap.png"):
    if "tyre_life" not in df.columns:
        print("No tyre_life column found, skipping tyre life plot.")
        return

    pits = pit_laps(df)
    plt.figure()
    plt.plot(df["lap"], df["tyre_life"])

    for pl in pits:
        plt.axvline(pl, linestyle="--", alpha=0.6, linewidth=1)

    plt.xlabel("Lap")
    plt.ylabel("Tyre life (laps)")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


def plot_compound(df, title="Compound over race", fname="compound_over_race.png"):
    if "compound" not in df.columns:
        print("No compound column found, skipping compound plot.")
        return

    comp_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    comp_vals = df["compound"].astype(str).map(lambda x: comp_map.get(x.upper(), None))

    if comp_vals.isna().any():
        missing = df.loc[comp_vals.isna(), "compound"].unique().tolist()
        print(f"Compound mapping missing values {missing}, skipping compound plot.")
        return

    plt.figure()
    plt.step(df["lap"], comp_vals, where="post")
    plt.yticks([0, 1, 2], ["SOFT", "MEDIUM", "HARD"])

    for pl in pit_laps(df):
        plt.axvline(pl, linestyle="--", alpha=0.6, linewidth=1)

    plt.xlabel("Lap")
    plt.ylabel("Compound")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


def plot_laptime_with_stints(df, title="Lap time vs Lap (stints annotated)", fname="lap_time_with_stints.png"):
    if "lap_time" not in df.columns or "compound" not in df.columns:
        print("Missing lap_time/compound columns, skipping stint-annotated plot.")
        return

    pits = pit_laps(df)

    plt.figure()
    plt.plot(df["lap"], df["lap_time"])

    for pl in pits:
        plt.axvline(pl, linestyle="--", alpha=0.6, linewidth=1)

    # Auto-tighten y-axis for cleaner visuals
    ymin = df["lap_time"].min() - 1
    ymax = df["lap_time"].max() + 1
    plt.ylim(ymin, ymax)

    # Stint boundaries:
    # If pit is logged on the lap where pit-loss applies (as in your trace),
    # then the next lap is the start of the new stint.
    stint_starts = [int(df["lap"].iloc[0])] + [p + 1 for p in pits]
    stint_ends = pits + [int(df["lap"].iloc[-1])]

    for s, e in zip(stint_starts, stint_ends):
        stint_df = df[(df["lap"] >= s) & (df["lap"] <= e)]
        if stint_df.empty:
            continue
        comp = str(stint_df["compound"].iloc[0])

        # Place label near the lower part of the stint curve so it doesn't overlap too much
        x = (s + e) / 2
        y = float(stint_df["lap_time"].min())
        plt.text(x, y, comp, ha="center", va="bottom")

    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


def main():
    df = load_trace()

    plot_laptime(df)
    plot_tyre_life(df)
    plot_compound(df)
    plot_laptime_with_stints(df)

    print(f"Saved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()