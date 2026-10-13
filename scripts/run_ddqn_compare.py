import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import pandas as pd

from config import SimConfig
from rl.dqn_train import train_dqn


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = SimConfig()

    runs = [
        {
            "name": "DQN",
            "use_double_dqn": False,
            "final_model_path": "models/dqn_compare.pt",
            "best_model_path": "models/dqn_compare_best.pt",
        },
        {
            "name": "DoubleDQN",
            "use_double_dqn": True,
            "final_model_path": "models/ddqn_compare.pt",
            "best_model_path": "models/ddqn_compare_best.pt",
        },
    ]

    rows = []

    for run in runs:
        print(f"\n{'='*60}")
        print(f"Training {run['name']}")
        print(f"{'='*60}")

        result = train_dqn(
            config=config,
            final_model_path=run["final_model_path"],
            best_model_path=run["best_model_path"],
            use_double_dqn=run["use_double_dqn"],
        )

        rows.append({
            "algorithm": run["name"],
            "best_time": result["best_time"],
            "best_ep": result["best_ep"],
            "best_model_path": result["best_model_path"],
        })

    df = pd.DataFrame(rows)
    out_path = "data/processed/ddqn_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved comparison to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
    