import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from config import SimConfig
from env.gym_race_env import GymRaceEnv

MODEL_PATH = "models/ddqn_safety_car_best.pt"
TRACE_PATH = "data/processed/ddqn_safety_car_eval_trace.csv"


def default_sc_config() -> SimConfig:
    return SimConfig(
        total_laps=52,
        enable_safety_car=True,
        sc_prob_per_lap=0.03,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.60,
        min_stint_laps=16,
        is_dry_race=True,
        tyre_rule_penalty=1000.0,
    )


def default_no_sc_config() -> SimConfig:
    return SimConfig(
        total_laps=52,
        enable_safety_car=False,
        sc_prob_per_lap=0.0,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.60,
        min_stint_laps=16,
        is_dry_race=True,
        tyre_rule_penalty=1000.0,
    )


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def _load_model(model_path: str, obs_dim: int, n_actions: int, device: torch.device):
    model = QNet(obs_dim, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_greedy_eval(
    model_path: str = MODEL_PATH,
    config: SimConfig | None = None,
    seed: int = 123,
    save_trace: bool = True,
    trace_path: str = TRACE_PATH,
    device: str | None = None,
):
    """
    Runs one greedy (argmax-Q) episode using a saved model.

    Returns:
      - total_reward
      - total_time
      - pit_events
      - trace_df
      - trace_path
      - config
      - used_compounds
      - illegal_tyre_strategy
      - tyre_rule_satisfied_final
    """
    config = config if config is not None else default_sc_config()
    env = GymRaceEnv(config=config)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = _load_model(model_path, obs_dim, n_actions, device)

    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    pit_events = []
    trace = []

    lap = 1
    last_info = None
    stint_index = 0

    while not done:
        current_obs = obs.copy()

        with torch.no_grad():
            s = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0).to(device)
            qvals = model(s)
            action = int(torch.argmax(qvals, dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        last_info = info

        pit_this_lap = int(info.get("pit", False))
        sc_this_lap = int(info.get("sc_active", False))
        pitted_under_sc = int(pit_this_lap == 1 and sc_this_lap == 1)

        if pit_this_lap:
            pit_events.append((lap, action, info["compound"]))
            stint_index += 1

        row = {
            "lap": lap,
            "action": action,
            "pit": pit_this_lap,
            "compound": info["compound"],
            "used_compounds": ",".join(info.get("used_compounds", [])),
            "tyre_life": info["tyre_life"],
            "lap_time": info["lap_time"],
            "pred_norm": info["pred_norm"],
            "deg": info.get("deg", np.nan),
            "total_time": info["total_time"],
            "reward": float(reward),

            "stint_index": stint_index,

            "sc_active": sc_this_lap,
            "sc_remaining_laps": info.get("sc_remaining_laps", 0),
            "sc_start_lap": info.get("sc_start_lap", np.nan),
            "safety_car_happened": int(info.get("safety_car_happened", False)),
            "pit_loss_used": info.get("pit_loss_used", np.nan),
            "pitted_under_sc": pitted_under_sc,

            "is_dry_race": int(info.get("is_dry_race", True)),
            "tyre_rule_satisfied_so_far": int(info.get("tyre_rule_satisfied_so_far", False)),
            "illegal_tyre_strategy": int(info.get("illegal_tyre_strategy", False)),
            "tyre_rule_satisfied_final": int(info.get("tyre_rule_satisfied_final", False))
            if done else np.nan,
            "tyre_rule_penalty_applied": info.get("tyre_rule_penalty_applied", 0.0),
        }

        for i, val in enumerate(current_obs):
            row[f"state_{i}"] = float(val)

        trace.append(row)
        lap += 1

    df = pd.DataFrame(trace)

    if save_trace:
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        df.to_csv(trace_path, index=False)

    return {
        "total_reward": total_reward,
        "total_time": float(last_info["total_time"]) if last_info and "total_time" in last_info else None,
        "pit_events": pit_events,
        "trace_path": trace_path if save_trace else None,
        "trace_df": df,
        "config": config,
        "used_compounds": last_info.get("used_compounds", []) if last_info else [],
        "illegal_tyre_strategy": last_info.get("illegal_tyre_strategy", False) if last_info else None,
        "tyre_rule_satisfied_final": last_info.get("tyre_rule_satisfied_final", None) if last_info else None,
    }


def run_multi_eval(
    model_path: str = MODEL_PATH,
    config: SimConfig | None = None,
    n_episodes: int = 100,
    seed_start: int = 1000,
    device: str | None = None,
    save_csv: bool = False,
    csv_path: str = "data/processed/ddqn_safety_car_multi_eval.csv",
):
    """
    Runs multiple greedy evaluations and returns a dataframe of summary results.
    Useful when Safety Car makes the environment stochastic.
    """
    config = config if config is not None else default_sc_config()
    rows = []

    for i in range(n_episodes):
        seed = seed_start + i
        result = run_greedy_eval(
            model_path=model_path,
            config=config,
            seed=seed,
            save_trace=False,
            device=device,
        )

        trace_df = result["trace_df"]
        sc_happened = int(trace_df["safety_car_happened"].max()) if len(trace_df) else 0
        num_pits = len(result["pit_events"])

        pitted_under_sc = 0
        sc_start_lap = np.nan
        sc_laps = 0

        if len(trace_df):
            pitted_under_sc = int(trace_df["pitted_under_sc"].max())
            sc_laps = int(trace_df["sc_active"].sum())
            sc_start_series = trace_df.loc[trace_df["sc_active"] == 1, "sc_start_lap"]
            if len(sc_start_series) > 0:
                sc_start_lap = float(sc_start_series.iloc[0])

        rows.append({
            "episode": i,
            "seed": seed,
            "total_reward": result["total_reward"],
            "total_time": result["total_time"],
            "num_pits": num_pits,
            "used_compounds": ",".join(result["used_compounds"]),
            "illegal_tyre_strategy": int(result["illegal_tyre_strategy"]),
            "tyre_rule_satisfied_final": int(result["tyre_rule_satisfied_final"]),
            "safety_car_happened": sc_happened,
            "pitted_under_sc": pitted_under_sc,
            "sc_laps": sc_laps,
            "sc_start_lap": sc_start_lap,
        })

    df = pd.DataFrame(rows)

    if save_csv:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df


def summarise_multi_eval(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    summary = pd.DataFrame([{
        "mean_total_time": df["total_time"].mean(),
        "std_total_time": df["total_time"].std(),
        "mean_total_reward": df["total_reward"].mean(),
        "mean_num_pits": df["num_pits"].mean(),
        "illegal_strategy_rate": df["illegal_tyre_strategy"].mean(),
        "tyre_rule_satisfied_rate": df["tyre_rule_satisfied_final"].mean(),
        "safety_car_rate": df["safety_car_happened"].mean(),
        "pitted_under_sc_rate": df["pitted_under_sc"].mean(),
        "mean_sc_laps": df["sc_laps"].mean(),
    }])

    return summary


def main():
    config = default_sc_config()

    result = run_greedy_eval(
        model_path=MODEL_PATH,
        config=config,
    )

    print(f"Saved trace to {TRACE_PATH}")
    print("Total reward:", result["total_reward"])
    print("Total time (sim):", result["total_time"])
    print("Used compounds:", result["used_compounds"])
    print("Tyre rule satisfied:", result["tyre_rule_satisfied_final"])
    print("Illegal tyre strategy:", result["illegal_tyre_strategy"])
    print("Pit events (lap, action, compound):")
    for e in result["pit_events"]:
        print(e)


if __name__ == "__main__":
    main()