import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from config import SimConfig
from env.baselines import run_plan, best_one_stop, best_two_stop
from rl.evaluate_dqn import run_greedy_eval


def pit_plan_str(pit_events):
    if not pit_events:
        return "no pits"
    return ", ".join([f"lap {lap} -> {comp}" for lap, _, comp in pit_events])


def main():
    os.makedirs("data/processed", exist_ok=True)

    pit_loss_values = [15.0, 25.0, 32.0, 40.0]
    rows = []

    # Use a checkpoint trained with the CURRENT 16-dim observation space
    model_path = "models/ddqn_compare_best.pt"

    for pit_loss in pit_loss_values:
        print(f"\n{'='*60}")
        print(f"Running sensitivity analysis for pit_loss_seconds = {pit_loss}")
        print(f"{'='*60}")

        config = SimConfig(pit_loss_seconds=pit_loss)

        # Baselines
        no_pit = run_plan({}, config=config)
        _, best1 = best_one_stop(config=config)
        _, best2 = best_two_stop(config=config, min_gap=config.min_stint_laps)

        # RL evaluation under this config
        eval_result = run_greedy_eval(
            model_path=model_path,
            config=config,
            save_trace=False,
        )

        dqn_time = eval_result["total_time"]
        dqn_pits = eval_result["pit_events"]

        rows.append({
            "pit_loss_seconds": pit_loss,
            "no_pit_time": float(no_pit),
            "best_1stop_time": float(best1[0]),
            "best_2stop_time": float(best2[0]),
            "ddqn_time": float(dqn_time),
            "ddqn_gap_to_best_2stop": float(dqn_time - best2[0]),
            "ddqn_pit_count": len(dqn_pits),
            "ddqn_pit_plan": pit_plan_str(dqn_pits),
        })

    df = pd.DataFrame(rows)
    out_path = "data/processed/sensitivity_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved sensitivity summary to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
    