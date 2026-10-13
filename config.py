from dataclasses import dataclass


# Per-circuit presets — approximate values calibrated to 2023 F1 season data.
# Loaded automatically when circuit_name is set to a non-"generic" key.
CIRCUIT_PRESETS: dict[str, dict] = {
    "bahrain":     {"total_laps": 57, "base_lap_time": 95.0,  "pit_loss_seconds": 30.0, "deg_k": 42.0, "deg_p": 1.80},
    "jeddah":      {"total_laps": 50, "base_lap_time": 90.0,  "pit_loss_seconds": 29.0, "deg_k": 30.0, "deg_p": 1.65},
    "melbourne":   {"total_laps": 58, "base_lap_time": 82.0,  "pit_loss_seconds": 33.0, "deg_k": 32.0, "deg_p": 1.70},
    "baku":        {"total_laps": 51, "base_lap_time": 103.0, "pit_loss_seconds": 31.0, "deg_k": 25.0, "deg_p": 1.60},
    "miami":       {"total_laps": 57, "base_lap_time": 91.0,  "pit_loss_seconds": 32.0, "deg_k": 38.0, "deg_p": 1.75},
    "monaco":      {"total_laps": 78, "base_lap_time": 74.0,  "pit_loss_seconds": 38.0, "deg_k": 14.0, "deg_p": 1.45},
    "barcelona":   {"total_laps": 66, "base_lap_time": 80.0,  "pit_loss_seconds": 31.0, "deg_k": 40.0, "deg_p": 1.80},
    "montreal":    {"total_laps": 70, "base_lap_time": 76.0,  "pit_loss_seconds": 30.0, "deg_k": 30.0, "deg_p": 1.65},
    "silverstone": {"total_laps": 52, "base_lap_time": 90.0,  "pit_loss_seconds": 31.0, "deg_k": 33.0, "deg_p": 1.70},
    "budapest":    {"total_laps": 70, "base_lap_time": 80.0,  "pit_loss_seconds": 32.0, "deg_k": 36.0, "deg_p": 1.72},
    "spa":         {"total_laps": 44, "base_lap_time": 107.0, "pit_loss_seconds": 30.0, "deg_k": 38.0, "deg_p": 1.75},
    "zandvoort":   {"total_laps": 72, "base_lap_time": 75.0,  "pit_loss_seconds": 33.0, "deg_k": 37.0, "deg_p": 1.73},
    "monza":       {"total_laps": 53, "base_lap_time": 82.0,  "pit_loss_seconds": 27.0, "deg_k": 45.0, "deg_p": 1.85},
    "singapore":   {"total_laps": 61, "base_lap_time": 103.0, "pit_loss_seconds": 35.0, "deg_k": 27.0, "deg_p": 1.60},
    "suzuka":      {"total_laps": 53, "base_lap_time": 93.0,  "pit_loss_seconds": 32.0, "deg_k": 36.0, "deg_p": 1.75},
    "losail":      {"total_laps": 57, "base_lap_time": 84.0,  "pit_loss_seconds": 30.0, "deg_k": 44.0, "deg_p": 1.82},
    "austin":      {"total_laps": 56, "base_lap_time": 96.0,  "pit_loss_seconds": 31.0, "deg_k": 39.0, "deg_p": 1.78},
    "mexico":      {"total_laps": 71, "base_lap_time": 79.0,  "pit_loss_seconds": 30.0, "deg_k": 28.0, "deg_p": 1.62},
    "interlagos":  {"total_laps": 71, "base_lap_time": 73.0,  "pit_loss_seconds": 30.0, "deg_k": 34.0, "deg_p": 1.70},
    "las_vegas":   {"total_laps": 50, "base_lap_time": 96.0,  "pit_loss_seconds": 29.0, "deg_k": 25.0, "deg_p": 1.60},
    "abu_dhabi":   {"total_laps": 58, "base_lap_time": 87.0,  "pit_loss_seconds": 29.0, "deg_k": 33.0, "deg_p": 1.70},
}


@dataclass
class SimConfig:
    # --- Circuit ---
    # Set to a key in CIRCUIT_PRESETS to auto-load that circuit's parameters.
    # "generic" keeps the values defined below unchanged.
    circuit_name: str = "generic"

    # Core race parameters (overridden by circuit preset when circuit_name != "generic")
    total_laps: int = 52
    pit_loss_seconds: float = 32.0
    base_lap_time: float = 95.0

    # Degradation model
    deg_k: float = 35.0
    deg_p: float = 1.7

    # Constraints / reward design
    min_stint_laps: int = 16
    reward_shaping: bool = True

    # Extra strategic shaping
    pit_stop_penalty: float = 3.0
    same_compound_pit_penalty: float = 10.0
    extra_stop_penalty: float = 40.0

    # FIA tyre legality
    is_dry_race: bool = True
    tyre_rule_penalty: float = 1000.0

    # --- Full Safety Car (SC) ---
    enable_safety_car: bool = True
    sc_prob_per_lap: float = 0.03
    sc_duration_laps_min: int = 2
    sc_duration_laps_max: int = 5
    sc_lap_time_multiplier: float = 1.20
    sc_pit_loss_multiplier: float = 0.40
    max_sc_events: int = 3              # multiple SCs allowed per race

    # --- Virtual Safety Car (VSC) ---
    enable_vsc: bool = True
    vsc_prob_per_lap: float = 0.04      # slightly more frequent than full SC
    vsc_duration_laps_min: int = 2
    vsc_duration_laps_max: int = 4
    vsc_lap_time_multiplier: float = 1.10   # milder slowdown than SC
    vsc_pit_loss_multiplier: float = 0.70   # smaller discount than SC

    # --- Red Flag ---
    enable_red_flag: bool = True
    red_flag_prob_per_lap: float = 0.01     # rare event
    red_flag_duration_laps: int = 3         # neutralisation laps before restart
    red_flag_free_pit: bool = True          # tyre change allowed without time penalty

    # --- Opponent Model ---
    enable_opponent_model: bool = False
    num_opponents: int = 3
    undercut_bonus_per_lap: float = 0.5     # per-lap reward bonus for undercutting
    traffic_penalty_per_lap: float = 0.3    # per-lap penalty for being stuck in traffic

    # --- Weather & Dynamic Track Conditions ---
    enable_weather: bool = False
    initial_weather: str = "dry"            # "dry", "damp", or "wet"
    rain_probability: float = 0.02          # per-lap probability of dry->damp transition

    def __post_init__(self):
        # Apply circuit preset — overrides base params for all non-generic circuits
        if self.circuit_name != "generic" and self.circuit_name in CIRCUIT_PRESETS:
            for k, v in CIRCUIT_PRESETS[self.circuit_name].items():
                setattr(self, k, v)

