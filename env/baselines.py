import itertools
import os
import pandas as pd

from config import SimConfig
from env.race_env import SimpleRaceEnv

# Actions: 0=stay, 1=SOFT, 2=MEDIUM, 3=HARD, 4=INTERMEDIATE, 5=WET
# Brute-force baselines use only dry-weather compounds by default.
# Pass wet_actions=True to include INTERMEDIATE and WET (useful when weather is enabled).
DRY_PIT_ACTIONS = [1, 2, 3]
WET_PIT_ACTIONS = [1, 2, 3, 4, 5]
PIT_ACTIONS = DRY_PIT_ACTIONS  # default — kept for backward compatibility


def run_plan(pit_plan, config: SimConfig | None = None):
    """
    pit_plan: dict mapping lap_number -> action (1/2/3)

    Returns a dict with:
      - total_time
      - used_compounds
      - illegal_tyre_strategy
      - tyre_rule_satisfied_final
      - tyre_rule_penalty_applied
    """
    config = config if config is not None else SimConfig()
    env = SimpleRaceEnv(config=config)
    env.reset()

    done = False
    info = None

    while not done:
        action = pit_plan.get(env.lap, 0)
        _, _, done, info = env.step(action)

    return {
        "total_time": float(info["total_time"]),
        "used_compounds": info.get("used_compounds", []),
        "illegal_tyre_strategy": bool(info.get("illegal_tyre_strategy", False)),
        "tyre_rule_satisfied_final": bool(info.get("tyre_rule_satisfied_final", True)),
        "tyre_rule_penalty_applied": float(info.get("tyre_rule_penalty_applied", 0.0)),
    }


def best_one_stop(config: SimConfig | None = None, min_lap=2, max_lap=None, wet_actions=False, legal_only=True):
    config = config if config is not None else SimConfig()
    if max_lap is None:
        max_lap = config.total_laps - 1

    actions = WET_PIT_ACTIONS if wet_actions else DRY_PIT_ACTIONS
    no_pit_result = run_plan({}, config=config)
    best = None  # (result_dict, pit_lap, action)

    for pit_lap in range(min_lap, max_lap):
        for a in actions:
            result = run_plan({pit_lap: a}, config=config)

            if legal_only and result["illegal_tyre_strategy"]:
                continue

            if best is None or result["total_time"] < best[0]["total_time"]:
                best = (result, pit_lap, a)

    return no_pit_result, best


def best_two_stop(config: SimConfig | None = None, min_gap=None, min_lap=2, max_lap=None, wet_actions=False, legal_only=True):
    """Brute-force 2-stop, respecting minimum gap between pit laps."""
    config = config if config is not None else SimConfig()
    if min_gap is None:
        min_gap = config.min_stint_laps
    if max_lap is None:
        max_lap = config.total_laps - 1

    actions = WET_PIT_ACTIONS if wet_actions else DRY_PIT_ACTIONS
    no_pit_result = run_plan({}, config=config)
    best = None  # (result_dict, pit_lap1, a1, pit_lap2, a2)

    for pit_lap1 in range(min_lap, max_lap):
        for pit_lap2 in range(pit_lap1 + min_gap, max_lap):
            for a1, a2 in itertools.product(actions, actions):
                result = run_plan({pit_lap1: a1, pit_lap2: a2}, config=config)

                if legal_only and result["illegal_tyre_strategy"]:
                    continue

                if best is None or result["total_time"] < best[0]["total_time"]:
                    best = (result, pit_lap1, a1, pit_lap2, a2)

    return no_pit_result, best


def read_dqn_trace(path="data/processed/ddqn_safety_car_eval_trace.csv"):
    df = pd.read_csv(path)

    total_time = float(df["total_time"].iloc[-1])
    pit_laps = df.loc[df["pit"] == 1, ["lap", "action", "compound"]].values.tolist()

    used_compounds = []
    if "used_compounds" in df.columns:
        last_used = df["used_compounds"].dropna().iloc[-1] if len(df["used_compounds"].dropna()) else ""
        used_compounds = [x.strip() for x in str(last_used).split(",") if x.strip()]

    illegal_tyre_strategy = False
    if "illegal_tyre_strategy" in df.columns:
        illegal_tyre_strategy = bool(df["illegal_tyre_strategy"].fillna(0).iloc[-1])

    tyre_rule_satisfied_final = True
    if "tyre_rule_satisfied_final" in df.columns:
        non_nan = df["tyre_rule_satisfied_final"].dropna()
        if len(non_nan):
            tyre_rule_satisfied_final = bool(non_nan.iloc[-1])

    tyre_rule_penalty_applied = 0.0
    if "tyre_rule_penalty_applied" in df.columns:
        tyre_rule_penalty_applied = float(df["tyre_rule_penalty_applied"].fillna(0.0).iloc[-1])

    return {
        "total_time": total_time,
        "pit_laps": pit_laps,
        "used_compounds": used_compounds,
        "illegal_tyre_strategy": illegal_tyre_strategy,
        "tyre_rule_satisfied_final": tyre_rule_satisfied_final,
        "tyre_rule_penalty_applied": tyre_rule_penalty_applied,
    }


def action_name(a):
    return {1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}.get(a, "??")


def save_summary_csv(rows, out_path="data/processed/baselines_summary.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved baseline summary to {out_path}")


def main():
    config = SimConfig()

    print("=== Baselines (current simulator settings) ===")
    print(f"Dry race: {config.is_dry_race}")
    print(f"Tyre rule penalty: {config.tyre_rule_penalty}")

    rows = []

    # No-pit
    no_pit_result = run_plan({}, config=config)
    no_pit = no_pit_result["total_time"]
    print(
        f"No pit total time: {no_pit:.3f} s | "
        f"used={no_pit_result['used_compounds']} | "
        f"illegal={no_pit_result['illegal_tyre_strategy']}"
    )
    rows.append({
        "strategy": "No pit",
        "total_time_s": float(no_pit),
        "pits": "",
        "used_compounds": ",".join(no_pit_result["used_compounds"]),
        "illegal_tyre_strategy": int(no_pit_result["illegal_tyre_strategy"]),
        "tyre_rule_satisfied_final": int(no_pit_result["tyre_rule_satisfied_final"]),
        "tyre_rule_penalty_applied": float(no_pit_result["tyre_rule_penalty_applied"]),
    })

    # Best legal 1-stop
    _, best1 = best_one_stop(config=config, legal_only=True)
    if best1 is not None:
        result1, lap1, a1 = best1
        t1 = result1["total_time"]
        pits1 = f"({lap1},{action_name(a1)})"
        print(
            f"Best LEGAL 1-stop: time={t1:.3f} s | pit lap={lap1} | tyre={action_name(a1)} | "
            f"used={result1['used_compounds']} | "
            f"illegal={result1['illegal_tyre_strategy']} | "
            f"improvement={no_pit - t1:.3f} s"
        )
        rows.append({
            "strategy": "Best legal 1-stop (BF)",
            "total_time_s": float(t1),
            "pits": pits1,
            "used_compounds": ",".join(result1["used_compounds"]),
            "illegal_tyre_strategy": int(result1["illegal_tyre_strategy"]),
            "tyre_rule_satisfied_final": int(result1["tyre_rule_satisfied_final"]),
            "tyre_rule_penalty_applied": float(result1["tyre_rule_penalty_applied"]),
        })

    # Best legal 2-stop
    min_gap = config.min_stint_laps
    _, best2 = best_two_stop(config=config, min_gap=min_gap, legal_only=True)
    if best2 is not None:
        result2, l1, aa1, l2, aa2 = best2
        t2 = result2["total_time"]
        pits2 = f"({l1},{action_name(aa1)}),({l2},{action_name(aa2)})"
        print(
            f"Best LEGAL 2-stop (gap>={min_gap}): time={t2:.3f} s | "
            f"pits=({l1},{action_name(aa1)}), ({l2},{action_name(aa2)}) | "
            f"used={result2['used_compounds']} | "
            f"illegal={result2['illegal_tyre_strategy']} | "
            f"improvement={no_pit - t2:.3f} s"
        )
        rows.append({
            "strategy": f"Best legal 2-stop (BF, gap>={min_gap})",
            "total_time_s": float(t2),
            "pits": pits2,
            "used_compounds": ",".join(result2["used_compounds"]),
            "illegal_tyre_strategy": int(result2["illegal_tyre_strategy"]),
            "tyre_rule_satisfied_final": int(result2["tyre_rule_satisfied_final"]),
            "tyre_rule_penalty_applied": float(result2["tyre_rule_penalty_applied"]),
        })

    # DQN / DDQN trace
    try:
        dqn_result = read_dqn_trace()
        dqn_time = dqn_result["total_time"]
        pit_laps = dqn_result["pit_laps"]

        print(
            f"DQN policy (from trace): time={dqn_time:.3f} s | "
            f"pits={pit_laps} | "
            f"used={dqn_result['used_compounds']} | "
            f"illegal={dqn_result['illegal_tyre_strategy']}"
        )
        print(f"DQN improvement vs no-pit: {no_pit - dqn_time:.3f} s")

        pits_str = ",".join([f"({int(lap)},{str(comp)})" for lap, _, comp in pit_laps])

        rows.append({
            "strategy": "DQN (best checkpoint)",
            "total_time_s": float(dqn_time),
            "pits": pits_str,
            "used_compounds": ",".join(dqn_result["used_compounds"]),
            "illegal_tyre_strategy": int(dqn_result["illegal_tyre_strategy"]),
            "tyre_rule_satisfied_final": int(dqn_result["tyre_rule_satisfied_final"]),
            "tyre_rule_penalty_applied": float(dqn_result["tyre_rule_penalty_applied"]),
        })
    except FileNotFoundError:
        print("DQN trace not found. Run: python -m rl.evaluate_dqn")

    save_summary_csv(rows, out_path="data/processed/baselines_summary.csv")


if __name__ == "__main__":
    main()
