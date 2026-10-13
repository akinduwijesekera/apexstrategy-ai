import os
import pandas as pd

from config import SimConfig
from rl.evaluate_dqn import run_multi_eval


RAW_OUT = "data/processed/safety_car_compare_raw.csv"
SUMMARY_OUT = "data/processed/safety_car_compare_summary.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)

    n_episodes = 100

    cfg_no_sc = SimConfig(
        total_laps=52,
        enable_safety_car=False,
        sc_prob_per_lap=0.0,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.40,
        min_stint_laps=16,
    )

    cfg_sc = SimConfig(
        total_laps=52,
        enable_safety_car=True,
        sc_prob_per_lap=0.03,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.40,
        min_stint_laps=16,
    )

    model_no_sc = "models/ddqn_no_sc_10d_best.pt"
    model_sc = "models/ddqn_safety_car_best.pt"

    experiments = [
        {
            "label": "no_sc_train__no_sc_test",
            "model_path": model_no_sc,
            "config": cfg_no_sc,
        },
        {
            "label": "no_sc_train__sc_test",
            "model_path": model_no_sc,
            "config": cfg_sc,
        },
        {
            "label": "sc_train__sc_test",
            "model_path": model_sc,
            "config": cfg_sc,
        },
        {
            "label": "sc_train__no_sc_test",
            "model_path": model_sc,
            "config": cfg_no_sc,
        },
    ]

    all_rows = []

    for exp in experiments:
        print(f"\nRunning: {exp['label']}")

        df = run_multi_eval(
            model_path=exp["model_path"],
            config=exp["config"],
            n_episodes=n_episodes,
            seed_start=1000,
            save_csv=False,
        ).copy()

        df["experiment"] = exp["label"]
        all_rows.append(df)

        print(
            f"  mean total_time = {df['total_time'].mean():.3f} | "
            f"std = {df['total_time'].std():.3f} | "
            f"mean pits = {df['num_pits'].mean():.3f} | "
            f"SC rate = {df['safety_car_happened'].mean():.3f} | "
            f"pit under SC rate = {df['pitted_under_sc'].mean():.3f}"
        )

    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_df.to_csv(RAW_OUT, index=False)

    summary_df = (
        raw_df.groupby("experiment", as_index=False)
        .agg(
            mean_total_time=("total_time", "mean"),
            std_total_time=("total_time", "std"),
            mean_total_reward=("total_reward", "mean"),
            mean_num_pits=("num_pits", "mean"),
            safety_car_rate=("safety_car_happened", "mean"),
            pitted_under_sc_rate=("pitted_under_sc", "mean"),
        )
        .sort_values("mean_total_time")
    )

    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\nSaved:")
    print(f"  Raw results: {RAW_OUT}")
    print(f"  Summary:     {SUMMARY_OUT}")

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
