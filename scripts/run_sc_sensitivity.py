import os
import pandas as pd

from config import SimConfig
from rl.evaluate_dqn import run_multi_eval

OUT_CSV = "data/processed/sc_sensitivity.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)

    # Use a checkpoint trained with the CURRENT 16-dim observation space
    model_path = "models/ddqn_compare_best.pt"
    probs = [0.00, 0.01, 0.03, 0.05, 0.08]

    rows = []

    for p in probs:
        print(f"Running SC sensitivity for p={p:.2f}")

        cfg = SimConfig(
            total_laps=52,
            enable_safety_car=(p > 0.0),
            sc_prob_per_lap=p,
            sc_duration_laps_min=2,
            sc_duration_laps_max=5,
            sc_lap_time_multiplier=1.20,
            sc_pit_loss_multiplier=0.60,
            min_stint_laps=16,
        )

        df = run_multi_eval(
            model_path=model_path,
            config=cfg,
            n_episodes=100,
            seed_start=2000,
            save_csv=False,
        )

        rows.append({
            "sc_prob_per_lap": p,
            "mean_total_time": df["total_time"].mean(),
            "std_total_time": df["total_time"].std(),
            "mean_num_pits": df["num_pits"].mean(),
            "safety_car_rate": df["safety_car_happened"].mean(),
            "pitted_under_sc_rate": df["pitted_under_sc"].mean(),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
    