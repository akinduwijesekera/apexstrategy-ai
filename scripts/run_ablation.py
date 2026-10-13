import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from config import SimConfig
from rl.dqn_train import train_dqn


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    variants = [
        {
            "name": "full_system",
            "config": SimConfig(
                reward_shaping=True,
                min_stint_laps=16,
            ),
        },
        {
            "name": "no_reward_shaping",
            "config": SimConfig(
                reward_shaping=False,
                min_stint_laps=16,
            ),
        },
        {
            "name": "no_min_stint",
            "config": SimConfig(
                reward_shaping=True,
                min_stint_laps=0,
            ),
        },
    ]

    rows = []

    for variant in variants:
        name = variant["name"]
        config = variant["config"]

        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")

        final_model_path = f"models/ablation_{name}.pt"
        best_model_path = f"models/ablation_{name}_best.pt"

        result = train_dqn(
            config=config,
            final_model_path=final_model_path,
            best_model_path=best_model_path,
        )

        rows.append({
            "variant": name,
            "reward_shaping": config.reward_shaping,
            "min_stint_laps": config.min_stint_laps,
            "best_time": result["best_time"],
            "best_ep": result["best_ep"],
            "final_model_path": result["final_model_path"],
            "best_model_path": result["best_model_path"],
        })

    df = pd.DataFrame(rows)
    out_path = "data/processed/ablation_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved ablation summary to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
