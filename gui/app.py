import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # NEW: needed for hatched Rectangle patches

from dotenv import load_dotenv
load_dotenv()

from config import SimConfig, CIRCUIT_PRESETS
from env.gym_race_env import GymRaceEnv
from env.baselines import run_plan, best_one_stop, best_two_stop, action_name
from explain.strategy_summary import summarise_trace


# ---------------------------
# Time formatting
# ---------------------------

def fmt_time(seconds: float) -> str:
    """Format a duration in seconds as [h:]mm:ss.ssss."""
    total_secs = abs(seconds)
    hours = int(total_secs) // 3600
    minutes = (int(total_secs) % 3600) // 60
    secs = total_secs - hours * 3600 - minutes * 60
    sign = "-" if seconds < 0 else ""
    if hours > 0:
        return f"{sign}{hours}:{minutes:02d}:{secs:07.4f}"
    return f"{sign}{minutes:02d}:{secs:07.4f}"


def fmt_trace_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy of the trace dataframe with time columns formatted."""
    display = df.copy()
    for col in ("lap_time", "total_time"):
        if col in display.columns:
            display[col] = display[col].apply(
                lambda v: fmt_time(v) if pd.notna(v) else ""
            )
    return display


# ---------------------------
# Helpers: model + evaluation
# ---------------------------

def load_qnet(obs_dim: int, n_actions: int, model_path: str, device: str = "cpu"):
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self, obs_dim=19, n_actions=6):
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

    model = QNet(obs_dim, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_greedy_policy(model_path: str, config: "SimConfig", seed: int = 123):
    import torch

    env = GymRaceEnv(config=config)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_qnet(obs_dim, n_actions, model_path, device=device)

    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    trace = []
    pit_events = []

    lap = 1
    stint_index = 0

    while not done:
        current_obs = obs.copy()

        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            qvals = model(s)
            action = int(torch.argmax(qvals, dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        pit_this_lap = int(info.get("pit", False))
        sc_this_lap = int(info.get("sc_active", False))
        vsc_this_lap = int(info.get("vsc_active", False))
        rf_this_lap = int(info.get("red_flag_active", False))
        pitted_under_sc = int(pit_this_lap == 1 and sc_this_lap == 1)
        pitted_under_vsc = int(info.get("pitted_under_vsc", False))
        pitted_under_rf = int(info.get("pitted_under_red_flag", False))

        if pit_this_lap:
            pit_events.append((lap, action, info["compound"]))
            stint_index += 1

        row = {
            "lap": lap,
            "action": action,
            "pit": pit_this_lap,
            "compound": info["compound"],
            "tyre_life": info["tyre_life"],
            "lap_time": info["lap_time"],
            "pred_norm": info["pred_norm"],
            "deg": info.get("deg", np.nan),
            "total_time": info["total_time"],
            "reward": float(reward),
            "stint_index": stint_index,
            # SC
            "sc_active": sc_this_lap,
            "sc_remaining_laps": info.get("sc_remaining_laps", 0),
            "sc_start_lap": info.get("sc_start_lap", np.nan),
            "safety_car_happened": int(info.get("safety_car_happened", False)),
            "sc_events_count": info.get("sc_events_count", 0),
            "pit_loss_used": info.get("pit_loss_used", np.nan),
            "pitted_under_sc": pitted_under_sc,
            # VSC
            "vsc_active": vsc_this_lap,
            "pitted_under_vsc": pitted_under_vsc,
            # Red Flag
            "red_flag_active": rf_this_lap,
            "pitted_under_red_flag": pitted_under_rf,
            "free_pit": int(info.get("free_pit", False)),
            # Weather
            "weather": info.get("weather", "dry"),
            "weather_penalty": info.get("weather_penalty", 0.0),
            # Opponents
            "gap_ahead_norm": info.get("gap_ahead_norm", 0.0),
            "gap_behind_norm": info.get("gap_behind_norm", 0.0),
        }

        for i, val in enumerate(current_obs):
            row[f"state_{i}"] = float(val)

        trace.append(row)
        lap += 1

    df = pd.DataFrame(trace)
    total_time = float(df["total_time"].iloc[-1])
    return df, total_time, total_reward, pit_events


def run_multi_policy(model_path: str, config: "SimConfig", n_runs: int = 20, seed_start: int = 1000):
    rows = []
    for i in range(n_runs):
        seed = seed_start + i
        df, total_time, total_reward, pit_events = run_greedy_policy(
            model_path=model_path, config=config, seed=seed,
        )
        sc_happened = int(df["safety_car_happened"].max()) if "safety_car_happened" in df.columns and len(df) else 0
        vsc_happened = int(df["vsc_active"].max()) if "vsc_active" in df.columns and len(df) else 0
        rf_happened = int(df["red_flag_active"].max()) if "red_flag_active" in df.columns and len(df) else 0
        pitted_under_sc = int(df["pitted_under_sc"].max()) if "pitted_under_sc" in df.columns and len(df) else 0
        rows.append({
            "run": i, "seed": seed, "total_time": total_time,
            "total_reward": total_reward, "num_pits": int(df["pit"].sum()),
            "safety_car_happened": sc_happened, "vsc_happened": vsc_happened,
            "red_flag_happened": rf_happened, "pitted_under_sc": pitted_under_sc,
        })
    return pd.DataFrame(rows)


# ---------------------------
# Plot helpers
# ---------------------------

def _vsc_hatch_rect(ax, x0, x1, y0, y1):
    """
    Draw a single diagonal-stripe Rectangle for VSC laps.
    Uses facecolor='none' so only the orange hatch lines are visible,
    layered over a very faint orange background tint (drawn separately via axvspan).
    """
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=0.5,
        edgecolor="orange",
        facecolor="none",
        hatch="////",          # four forward-slashes → dense diagonal stripes
        alpha=0.60,
        zorder=2,
        transform=ax.transData,
        clip_on=True,
    )
    ax.add_patch(rect)


def _shade_events(ax, df: pd.DataFrame):
    """
    Shade race-event laps behind the main plot line.

    SC        → solid yellow band  (unchanged)
    VSC       → faint orange tint  +  diagonal orange hatch stripes  (NEW)
    Red Flag  → solid red band     (unchanged)
    """
    # Snapshot y-limits BEFORE adding any patches so the rectangles span the
    # correct data range.  (axvspan is axis-fraction based so it is fine.)
    y0, y1 = ax.get_ylim()

    for col, color, label in [
        ("sc_active",       "yellow", "SC"),
        ("vsc_active",      "orange", "VSC"),
        ("red_flag_active", "red",    "Red Flag"),
    ]:
        if col not in df.columns:
            continue

        active_laps = df.loc[df[col] == 1, "lap"].tolist()
        for lap in active_laps:
            x0_lap, x1_lap = lap - 0.5, lap + 0.5
            if col == "vsc_active":
                # Faint background tint so it isn't completely empty
                ax.axvspan(x0_lap, x1_lap, alpha=0.10, color=color, zorder=1, linewidth=0)
                # Diagonal stripe overlay
                _vsc_hatch_rect(ax, x0_lap, x1_lap, y0, y1)
            else:
                ax.axvspan(x0_lap, x1_lap, alpha=0.15, color=color, zorder=1, label=None)


def plot_lap_time(df: pd.DataFrame, title="Lap time vs lap"):
    pits = df.loc[df["pit"] == 1, "lap"].tolist()

    fig, ax = plt.subplots()
    ax.plot(df["lap"], df["lap_time"], label="Lap time", zorder=5)
    _shade_events(ax, df)

    pit_df = df.loc[df["pit"] == 1]
    if len(pit_df):
        ax.scatter(pit_df["lap"], pit_df["lap_time"], marker="x", s=80,
                   color="black", label="Pit stop", zorder=6)

    for pl in pits:
        ax.axvline(pl, linestyle="--", alpha=0.4, linewidth=1, zorder=3)

    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_tyre_life(df: pd.DataFrame, title="Tyre life vs lap"):
    pits = df.loc[df["pit"] == 1, "lap"].tolist()

    fig, ax = plt.subplots()
    ax.plot(df["lap"], df["tyre_life"], label="Tyre life", zorder=5)
    _shade_events(ax, df)

    for pl in pits:
        ax.axvline(pl, linestyle="--", alpha=0.4, linewidth=1, zorder=3)

    ax.set_xlabel("Lap")
    ax.set_ylabel("Tyre life (laps)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_compound(df: pd.DataFrame, title="Compound over race"):
    comp_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    comp_vals = df["compound"].astype(str).map(lambda x: comp_map.get(x.upper(), np.nan))

    fig, ax = plt.subplots()
    ax.step(df["lap"], comp_vals, where="post", label="Compound", zorder=5)
    _shade_events(ax, df)

    pit_df = df.loc[df["pit"] == 1]
    if len(pit_df):
        ax.scatter(pit_df["lap"], comp_vals.loc[pit_df.index], marker="x",
                   s=80, color="black", label="Pit stop", zorder=6)

    ax.set_yticks([0, 1, 2, 3, 4], ["SOFT", "MEDIUM", "HARD", "INTER", "WET"])
    ax.set_xlabel("Lap")
    ax.set_ylabel("Compound")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_sc_timeline(df: pd.DataFrame, title="Neutralisation events"):
    """
    Binary timeline showing when each neutralisation type was active.

    SC        → solid gold fill     (unchanged)
    VSC       → diagonal orange hatch stripes  (NEW – replaces solid fill)
    Red Flag  → solid red fill      (unchanged)
    """
    fig, ax = plt.subplots()

    # --- SC: solid fill ---
    if "sc_active" in df.columns:
        ax.fill_between(df["lap"], df["sc_active"].clip(0, 1), step="post",
                        alpha=0.50, color="gold", label="SC", zorder=1)

    # --- VSC: faint tint + per-lap diagonal hatch stripes ---
    if "vsc_active" in df.columns:
        # Subtle background so the column isn't invisible when hatch is sparse
        ax.fill_between(df["lap"], df["vsc_active"].clip(0, 1), step="post",
                        alpha=0.10, color="orange", label=None, zorder=1)

        # Draw a hatched rectangle for every active VSC lap
        for lap in df.loc[df["vsc_active"] == 1, "lap"].tolist():
            rect = mpatches.Rectangle(
                (lap - 0.5, 0), 1.0, 1.0,
                linewidth=0.5,
                edgecolor="orange",
                facecolor="none",
                hatch="////",
                alpha=0.70,
                zorder=2,
                transform=ax.transData,
                clip_on=True,
            )
            ax.add_patch(rect)

        # Legend proxy for VSC (hatched patch without fill)
        vsc_proxy = mpatches.Patch(
            facecolor="none", edgecolor="orange",
            hatch="////", alpha=0.70, label="VSC",
        )
        # We'll collect all handles at the end; for now stash the proxy
        ax._vsc_legend_proxy = vsc_proxy  # type: ignore[attr-defined]

    # --- Red Flag: solid fill ---
    if "red_flag_active" in df.columns:
        ax.fill_between(df["lap"], df["red_flag_active"].clip(0, 1), step="post",
                        alpha=0.50, color="red", label="Red Flag", zorder=1)

    # --- Pit stop markers ---
    pit_df = df.loc[df["pit"] == 1]
    if len(pit_df):
        ax.scatter(pit_df["lap"], [1.02] * len(pit_df), marker="x", s=80,
                   color="black", label="Pit stop", zorder=5)

    # Build legend including the VSC hatch proxy
    handles, labels = ax.get_legend_handles_labels()
    if hasattr(ax, "_vsc_legend_proxy"):
        handles.insert(1, ax._vsc_legend_proxy)
        labels.insert(1, "VSC")
    ax.legend(handles=handles, labels=labels)

    ax.set_xlabel("Lap")
    ax.set_ylabel("Event active")
    ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    return fig


def plot_weather(df: pd.DataFrame, title="Weather over race"):
    weather_map = {"dry": 0, "damp": 1, "wet": 2}
    if "weather" not in df.columns:
        return None
    weather_vals = df["weather"].map(lambda x: weather_map.get(str(x).lower(), 0))

    fig, ax = plt.subplots()
    ax.step(df["lap"], weather_vals, where="post", color="steelblue", label="Weather", zorder=5)

    pit_df = df.loc[df["pit"] == 1]
    if len(pit_df):
        ax.scatter(pit_df["lap"], weather_vals.loc[pit_df.index], marker="x",
                   s=80, color="black", label="Pit stop", zorder=6)

    ax.set_yticks([0, 1, 2], ["Dry", "Damp", "Wet"])
    ax.set_xlabel("Lap")
    ax.set_ylabel("Weather")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_multi_run_distribution(df: pd.DataFrame, title="Multi-run total time distribution"):
    fig, ax = plt.subplots()
    ax.hist(df["total_time"], bins=min(10, max(5, len(df) // 2)))
    ax.set_xlabel("Total race time (s)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


# ---------------------------
# Explanation helpers
# ---------------------------

def load_baselines_dict(path="data/processed/baselines_summary.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {row["strategy"]: float(row["total_time_s"]) for _, row in df.iterrows()}


def llm_explain(summary: dict) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    prompt = f"""
You are an assistant that explains an F1 pit strategy chosen by a reinforcement learning agent.

Rules:
- Use ONLY the provided JSON. Do not invent numbers or tyre choices.
- When describing pit stops, you MUST use pit_plan_str exactly as written.
- When comparing strategies, use baselines_total_time_s exactly.
- If a Safety Car / VSC / Red Flag occurred, explicitly mention whether the policy pitted under it.

Write:
1) Headline using pit_plan_str
2) Why each stop likely happened (degradation vs pit loss trade-off)
3) Neutralisation events note (SC / VSC / Red Flag) if applicable
4) Weather note if weather changed during the race
5) Comparison using exact times in baselines_total_time_s
6) Limitations note

Race summary (JSON):
{json.dumps(summary, indent=2)}
""".strip()

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)

    try:
        return resp.output_text
    except Exception:
        texts = []
        for item in getattr(resp, "output", []):
            if getattr(item, "type", "") == "output_text":
                texts.append(getattr(item, "text", ""))
        return "\n".join(texts).strip()


def local_explain(summary: dict) -> str:
    pits = summary.get("pit_plan_str", "no pits")
    total = summary.get("total_time_s", None)
    sc_happened = summary.get("safety_car_happened", False)
    sc_start = summary.get("safety_car_start_lap", None)
    sc_duration = summary.get("safety_car_duration", None)
    pitted_under_sc = summary.get("pitted_under_safety_car", False)
    vsc_happened = summary.get("vsc_happened", False)
    rf_happened = summary.get("red_flag_happened", False)

    lines = []
    if total is not None:
        lines.append(f'Headline: The RL agent chose the pit strategy "{pits}" and finished in {fmt_time(total)}.')
    else:
        lines.append(f'Headline: The RL agent chose the pit strategy "{pits}".')

    lines.append("")
    lines.append("Rationale: The agent is trading pit loss against tyre degradation; pitting resets tyre life and stabilises lap times.")

    if sc_happened:
        sc_line = "Safety Car: A Safety Car occurred"
        if sc_start is not None:
            sc_line += f" around lap {sc_start}"
        if sc_duration is not None:
            sc_line += f" for {sc_duration} laps"
        sc_line += "."
        lines.append(sc_line)
        if pitted_under_sc:
            lines.append("The policy exploited the reduced pit-loss during the Safety Car by stopping under neutralised conditions.")
        else:
            lines.append("The policy did not pit under the Safety Car, suggesting the remaining stint structure still looked acceptable.")
    else:
        lines.append("Safety Car: No Safety Car occurred in this simulated race.")

    if vsc_happened:
        lines.append("VSC: A Virtual Safety Car period occurred, providing a partial pit-loss discount.")
    if rf_happened:
        lines.append("Red Flag: A Red Flag was shown, enabling a free tyre change under neutralisation.")

    weather_note = summary.get("weather_note", "")
    if weather_note:
        lines.append(f"Weather: {weather_note}")

    lines.append("Limitations: This is a simplified simulator; opponent strategies are rule-based and do not reflect real team decision-making.")
    return "\n".join(lines)


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="ApexStrategy AI", layout="wide")

st.title("ApexStrategy AI — F1 Pit Strategy Optimiser")
st.caption(
    "DDQN policy evaluation · Safety Car / VSC / Red Flag · "
    "Weather & dynamic track conditions · Multi-car opponent model · "
    "Cross-circuit generalisation"
)

MODEL_PRESETS = {
    "SC-aware DDQN": "models/ddqn_safety_car_best.pt",
    "No-SC DDQN baseline": "models/ddqn_no_sc_10d_best.pt",
    "Custom path": None,
}

CIRCUIT_OPTIONS = ["generic"] + sorted(CIRCUIT_PRESETS.keys())

with st.sidebar:
    st.header("Circuit")
    circuit_name = st.selectbox("Circuit preset", CIRCUIT_OPTIONS, index=0)
    if circuit_name != "generic":
        preset = CIRCUIT_PRESETS[circuit_name]
        st.info(
            f"Preset: laps={preset['total_laps']}, "
            f"base={preset['base_lap_time']}s, "
            f"pit_loss={preset['pit_loss_seconds']}s, "
            f"deg_k={preset['deg_k']}, deg_p={preset['deg_p']}"
        )

    st.divider()
    st.header("Simulation settings")
    _default = CIRCUIT_PRESETS.get(circuit_name, {})
    total_laps = st.number_input("Total laps", value=int(_default.get("total_laps", 52)), min_value=10, max_value=100, step=1)
    base_lap_time = st.number_input("Base lap time (s)", value=float(_default.get("base_lap_time", 95.0)), min_value=60.0, max_value=150.0, step=0.5)
    pit_loss_seconds = st.number_input("Pit loss (s)", value=float(_default.get("pit_loss_seconds", 32.0)), min_value=10.0, max_value=60.0, step=0.5)
    deg_k = st.number_input("Degradation k", value=float(_default.get("deg_k", 35.0)), min_value=1.0, max_value=100.0, step=1.0)
    deg_p = st.number_input("Degradation exponent", value=float(_default.get("deg_p", 1.7)), min_value=0.5, max_value=3.0, step=0.1)
    min_stint_laps = st.number_input("Minimum stint length", value=16, min_value=0, max_value=40, step=1)
    reward_shaping = st.checkbox("Reward shaping enabled", value=True)
    seed = st.number_input("Seed", value=123, min_value=0, max_value=10_000, step=1)

    st.divider()
    st.header("Safety Car (SC)")
    enable_safety_car = st.checkbox("Enable Safety Car", value=True)
    sc_prob_per_lap = st.number_input("SC prob/lap", value=0.03, min_value=0.0, max_value=0.20, step=0.01, format="%.2f")
    sc_duration_laps_min = st.number_input("SC min duration", value=2, min_value=1, max_value=10, step=1)
    sc_duration_laps_max = st.number_input("SC max duration", value=5, min_value=1, max_value=10, step=1)
    sc_lap_time_multiplier = st.number_input("SC lap-time multiplier", value=1.20, min_value=1.0, max_value=2.0, step=0.05)
    sc_pit_loss_multiplier = st.number_input("SC pit-loss multiplier", value=0.40, min_value=0.1, max_value=1.0, step=0.05)
    max_sc_events = st.number_input("Max SC events per race", value=3, min_value=1, max_value=10, step=1)

    st.divider()
    st.header("Virtual Safety Car (VSC)")
    enable_vsc = st.checkbox("Enable VSC", value=True)
    vsc_prob_per_lap = st.number_input("VSC prob/lap", value=0.04, min_value=0.0, max_value=0.20, step=0.01, format="%.2f")
    vsc_duration_laps_min = st.number_input("VSC min duration", value=2, min_value=1, max_value=8, step=1)
    vsc_duration_laps_max = st.number_input("VSC max duration", value=4, min_value=1, max_value=8, step=1)
    vsc_lap_time_multiplier = st.number_input("VSC lap-time multiplier", value=1.10, min_value=1.0, max_value=1.5, step=0.05)
    vsc_pit_loss_multiplier = st.number_input("VSC pit-loss multiplier", value=0.70, min_value=0.1, max_value=1.0, step=0.05)

    st.divider()
    st.header("Red Flag")
    enable_red_flag = st.checkbox("Enable Red Flag", value=True)
    red_flag_prob_per_lap = st.number_input("Red Flag prob/lap", value=0.01, min_value=0.0, max_value=0.10, step=0.005, format="%.3f")
    red_flag_duration_laps = st.number_input("Red Flag duration (laps)", value=3, min_value=1, max_value=10, step=1)
    red_flag_free_pit = st.checkbox("Free tyre change under Red Flag", value=True)

    st.divider()
    st.header("Weather")
    enable_weather = st.checkbox("Enable weather", value=False)
    initial_weather = st.selectbox("Initial weather", ["dry", "damp", "wet"], index=0)
    rain_probability = st.number_input("Rain prob/lap (dry→damp)", value=0.02, min_value=0.0, max_value=0.20, step=0.01, format="%.2f")

    st.divider()
    st.header("Opponent Model")
    enable_opponent_model = st.checkbox("Enable opponents", value=False)
    num_opponents = st.number_input("Number of opponents", value=3, min_value=1, max_value=10, step=1)
    undercut_bonus_per_lap = st.number_input("Undercut bonus/lap", value=0.5, min_value=0.0, max_value=5.0, step=0.1)
    traffic_penalty_per_lap = st.number_input("Traffic penalty/lap", value=0.3, min_value=0.0, max_value=5.0, step=0.1)

    st.divider()
    st.header("Model")
    model_choice = st.selectbox("Model preset", list(MODEL_PRESETS.keys()), index=0)

    if model_choice == "Custom path":
        model_path = st.text_input("Custom model path", value="models/ddqn_safety_car_best.pt")
    else:
        model_path = MODEL_PRESETS[model_choice]
        st.text_input("Resolved model path", value=model_path, disabled=True)

    st.divider()
    st.header("Run mode")
    run_mode = st.radio("Evaluation mode", ["Single run", "Multi-run"], index=0)
    n_runs = st.number_input("Number of runs", value=20, min_value=2, max_value=200, step=1)
    multi_seed_start = st.number_input("Multi-run seed start", value=1000, min_value=0, max_value=100000, step=1)

    st.divider()
    use_llm = st.checkbox("Use LLM explanation (requires OPENAI_API_KEY + quota)", value=False)

    st.divider()
    save_trace = st.checkbox("Save trace to CSV", value=True)
    trace_out_path = st.text_input("Trace output path", value="data/processed/ddqn_gui_eval_trace.csv")

# Build config from sidebar
config = SimConfig(
    circuit_name=circuit_name,
    total_laps=int(total_laps),
    pit_loss_seconds=float(pit_loss_seconds),
    base_lap_time=float(base_lap_time),
    deg_k=float(deg_k),
    deg_p=float(deg_p),
    min_stint_laps=int(min_stint_laps),
    reward_shaping=bool(reward_shaping),
    # SC
    enable_safety_car=bool(enable_safety_car),
    sc_prob_per_lap=float(sc_prob_per_lap),
    sc_duration_laps_min=int(sc_duration_laps_min),
    sc_duration_laps_max=int(sc_duration_laps_max),
    sc_lap_time_multiplier=float(sc_lap_time_multiplier),
    sc_pit_loss_multiplier=float(sc_pit_loss_multiplier),
    max_sc_events=int(max_sc_events),
    # VSC
    enable_vsc=bool(enable_vsc),
    vsc_prob_per_lap=float(vsc_prob_per_lap),
    vsc_duration_laps_min=int(vsc_duration_laps_min),
    vsc_duration_laps_max=int(vsc_duration_laps_max),
    vsc_lap_time_multiplier=float(vsc_lap_time_multiplier),
    vsc_pit_loss_multiplier=float(vsc_pit_loss_multiplier),
    # Red Flag
    enable_red_flag=bool(enable_red_flag),
    red_flag_prob_per_lap=float(red_flag_prob_per_lap),
    red_flag_duration_laps=int(red_flag_duration_laps),
    red_flag_free_pit=bool(red_flag_free_pit),
    # Weather
    enable_weather=bool(enable_weather),
    initial_weather=str(initial_weather),
    rain_probability=float(rain_probability),
    # Opponents
    enable_opponent_model=bool(enable_opponent_model),
    num_opponents=int(num_opponents),
    undercut_bonus_per_lap=float(undercut_bonus_per_lap),
    traffic_penalty_per_lap=float(traffic_penalty_per_lap),
)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Run DDQN policy")
    run_btn = st.button("Run evaluation", type="primary")

with colB:
    st.subheader("Baselines")
    baseline_btn = st.button("Compute brute-force baselines")

# Session state
for key, default in {
    "trace_df": None,
    "total_time": None,
    "baseline_summary": None,
    "explanation": None,
    "pit_events": None,
    "multi_df": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Run evaluation
if run_btn:
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
    else:
        if run_mode == "Single run":
            with st.spinner("Running single greedy DDQN evaluation..."):
                df, total_time, total_reward, pit_events = run_greedy_policy(
                    model_path=model_path,
                    config=config,
                    seed=int(seed),
                )
                st.session_state.trace_df = df
                st.session_state.total_time = total_time
                st.session_state.pit_events = pit_events
                st.session_state.multi_df = None

                if save_trace:
                    trace_dir = os.path.dirname(trace_out_path)
                    if trace_dir:
                        os.makedirs(trace_dir, exist_ok=True)
                    df.to_csv(trace_out_path, index=False)

            sc_happened = bool(df["safety_car_happened"].max()) if "safety_car_happened" in df.columns else False
            vsc_happened = bool(df["vsc_active"].max()) if "vsc_active" in df.columns else False
            rf_happened = bool(df["red_flag_active"].max()) if "red_flag_active" in df.columns else False
            pitted_under_sc = bool(df["pitted_under_sc"].max()) if "pitted_under_sc" in df.columns else False

            flags = []
            if sc_happened:
                flags.append(f"SC={'pitted' if pitted_under_sc else 'no pit'}")
            if vsc_happened:
                flags.append("VSC")
            if rf_happened:
                flags.append("RedFlag")

            st.success(
                f"DDQN finished in {fmt_time(total_time)} | "
                f"pits={[(l, c) for (l, a, c) in pit_events]} | "
                + (" | ".join(flags) if flags else "no neutralisation events")
            )

        else:
            with st.spinner("Running multi-seed DDQN evaluation..."):
                multi_df = run_multi_policy(
                    model_path=model_path,
                    config=config,
                    n_runs=int(n_runs),
                    seed_start=int(multi_seed_start),
                )
                st.session_state.multi_df = multi_df
                st.session_state.trace_df = None
                st.session_state.total_time = None
                st.session_state.pit_events = None

            st.success("Multi-run evaluation complete.")

# Compute baselines
if baseline_btn:
    with st.spinner("Computing baselines (brute force)..."):
        no_pit_time = run_plan({}, config=config)
        _, best1 = best_one_stop(config=config)
        _, best2 = best_two_stop(config=config, min_gap=config.min_stint_laps)

        summary = [
            ("No pit", float(no_pit_time)),
            ("Best 1-stop (BF)", float(best1[0])),
            (f"Best 2-stop (BF, gap>={config.min_stint_laps})", float(best2[0])),
        ]

        st.session_state.baseline_summary = summary

        os.makedirs("data/processed", exist_ok=True)
        rows = [
            {"strategy": "No pit", "total_time_s": float(no_pit_time), "pits": ""},
            {
                "strategy": "Best 1-stop (BF)",
                "total_time_s": float(best1[0]),
                "pits": f"({best1[1]},{action_name(best1[2])})"
            },
            {
                "strategy": f"Best 2-stop (BF, gap>={config.min_stint_laps})",
                "total_time_s": float(best2[0]),
                "pits": f"({best2[1]},{action_name(best2[2])}),({best2[3]},{action_name(best2[4])})"
            },
        ]
        if st.session_state.total_time is not None:
            rows.append({
                "strategy": "DDQN",
                "total_time_s": float(st.session_state.total_time),
                "pits": ""
            })

        pd.DataFrame(rows).to_csv("data/processed/baselines_summary.csv", index=False)

    st.success("Baselines computed and saved to data/processed/baselines_summary.csv")

st.divider()

# Single-run display
df = st.session_state.trace_df
if df is not None:
    st.subheader("Single-run DDQN trace")
    st.dataframe(fmt_trace_df(df), use_container_width=True, height=260)

    c0, c1, c2, c3, c4, c5 = st.columns(6)
    with c0:
        st.metric("Total time", fmt_time(st.session_state.total_time))
    with c1:
        st.metric("Pit stops", int(df["pit"].sum()))
    with c2:
        sc_count = int(df.get("sc_events_count", pd.Series([0])).max()) if "sc_events_count" in df.columns else 0
        st.metric("SC events", sc_count)
    with c3:
        vsc = "Yes" if ("vsc_active" in df.columns and int(df["vsc_active"].max())) else "No"
        st.metric("VSC occurred", vsc)
    with c4:
        rf = "Yes" if ("red_flag_active" in df.columns and int(df["red_flag_active"].max())) else "No"
        st.metric("Red Flag", rf)
    with c5:
        wx = df["weather"].iloc[-1] if "weather" in df.columns else "dry"
        st.metric("Final weather", str(wx).capitalize())

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.pyplot(plot_lap_time(df, "Lap time vs lap (DDQN)"))
    with r1c2:
        st.pyplot(plot_sc_timeline(df, "Neutralisation events"))

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.pyplot(plot_tyre_life(df, "Tyre life vs lap (DDQN)"))
    with r2c2:
        st.pyplot(plot_compound(df, "Compound over race (DDQN)"))

    if enable_weather and "weather" in df.columns:
        fig_wx = plot_weather(df, "Weather over race")
        if fig_wx is not None:
            st.pyplot(fig_wx)

# Multi-run display
multi_df = st.session_state.multi_df
if multi_df is not None:
    st.subheader("Multi-run DDQN evaluation")
    st.dataframe(multi_df, use_container_width=True, height=260)

    c0, c1, c2, c3, c4, c5 = st.columns(6)
    with c0:
        st.metric("Mean total time", fmt_time(multi_df["total_time"].mean()))
    with c1:
        st.metric("Std total time (s)", f"{multi_df['total_time'].std():.3f}")
    with c2:
        st.metric("Mean pit count", f"{multi_df['num_pits'].mean():.2f}")
    with c3:
        st.metric("SC rate", f"{multi_df['safety_car_happened'].mean():.2f}")
    with c4:
        vsc_rate = multi_df["vsc_happened"].mean() if "vsc_happened" in multi_df.columns else 0.0
        st.metric("VSC rate", f"{vsc_rate:.2f}")
    with c5:
        rf_rate = multi_df["red_flag_happened"].mean() if "red_flag_happened" in multi_df.columns else 0.0
        st.metric("Red Flag rate", f"{rf_rate:.2f}")

    st.pyplot(plot_multi_run_distribution(multi_df, "Multi-run total time distribution"))

# Show baselines table
if st.session_state.baseline_summary is not None:
    st.subheader("Baseline comparison")
    bdf = pd.DataFrame(
        [
            (strategy, fmt_time(t))
            for strategy, t in st.session_state.baseline_summary
        ],
        columns=["Strategy", "Total time"],
    )
    if st.session_state.total_time is not None:
        bdf = pd.concat(
            [
                bdf,
                pd.DataFrame(
                    [["DDQN (current run)", fmt_time(st.session_state.total_time)]],
                    columns=["Strategy", "Total time"],
                ),
            ],
            ignore_index=True,
        )
    st.table(bdf)

st.divider()

# Explanation
st.subheader("Strategy explanation")
explain_btn = st.button("Generate explanation")

if explain_btn:
    trace_path = trace_out_path

    if df is not None and (not save_trace or not os.path.exists(trace_path)):
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        df.to_csv(trace_path, index=False)

    if df is None and multi_df is not None:
        st.warning("Explanation is only available for single-run mode. Switch to Single run and evaluate first.")
    elif not os.path.exists(trace_path):
        st.error("No trace available. Run a single DDQN evaluation first.")
    else:
        with st.spinner("Building summary..."):
            summary = summarise_trace(trace_path)

            baselines = load_baselines_dict()
            if baselines:
                summary["baselines_total_time_s"] = baselines

            pit_laps = summary.get("pit_laps", [])
            pit_comps = summary.get("pit_compounds", [])
            summary["pit_plan_str"] = ", ".join(
                [f"lap {l} -> {c}" for l, c in zip(pit_laps, pit_comps)]
            ) if pit_laps else "no pits"

            if df is not None and len(df):
                summary["safety_car_happened"] = bool(df["safety_car_happened"].max()) if "safety_car_happened" in df.columns else False
                summary["vsc_happened"] = bool(df["vsc_active"].max()) if "vsc_active" in df.columns else False
                summary["red_flag_happened"] = bool(df["red_flag_active"].max()) if "red_flag_active" in df.columns else False

                if summary["safety_car_happened"]:
                    sc_df = df[df["sc_active"] == 1]
                    if len(sc_df):
                        summary["safety_car_start_lap"] = int(sc_df["lap"].iloc[0])
                        summary["safety_car_duration"] = int(sc_df["sc_active"].sum())
                summary["pitted_under_safety_car"] = bool(df["pitted_under_sc"].max()) if "pitted_under_sc" in df.columns else False

                if "weather" in df.columns:
                    weather_counts = df["weather"].value_counts().to_dict()
                    if len(weather_counts) > 1:
                        summary["weather_note"] = "Weather changed during the race: " + str(weather_counts)

        if use_llm:
            try:
                with st.spinner("Calling LLM..."):
                    txt = llm_explain(summary)
                st.session_state.explanation = txt
            except Exception as e:
                st.warning(f"LLM unavailable ({e}). Falling back to local explanation.")
                st.session_state.explanation = local_explain(summary)
        else:
            st.session_state.explanation = local_explain(summary)

if st.session_state.explanation:
    st.text_area("Explanation output", st.session_state.explanation, height=260)

st.caption(
    "Legend: yellow = Safety Car (solid) | orange //// = VSC (diagonal stripes) | red = Red Flag | x = pit stop  "
    "· Tip: enable opponent model + weather for the most realistic strategy behaviour."
)
