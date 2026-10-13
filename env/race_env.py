import numpy as np

from config import SimConfig
from lstm.predict_lstm import load_model, predict_next_laptime_norm

# All available compounds (indices 1-5 map to pit actions 1-5)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]

# 5-dim one-hot for observation vector
COMPOUND_TO_ONEHOT = {
    "SOFT":   np.array([1., 0., 0., 0., 0.], dtype=np.float32),
    "MEDIUM": np.array([0., 1., 0., 0., 0.], dtype=np.float32),
    "HARD":   np.array([0., 0., 1., 0., 0.], dtype=np.float32),
    "INTER":  np.array([0., 0., 0., 1., 0.], dtype=np.float32),
    "WET":    np.array([0., 0., 0., 0., 1.], dtype=np.float32),
}

# Map INTER/WET to SOFT for LSTM (trained only on dry compounds)
_LSTM_COMPOUND_OH = {
    "SOFT":   np.array([1., 0., 0.], dtype=np.float32),
    "MEDIUM": np.array([0., 1., 0.], dtype=np.float32),
    "HARD":   np.array([0., 0., 1.], dtype=np.float32),
    "INTER":  np.array([1., 0., 0.], dtype=np.float32),
    "WET":    np.array([1., 0., 0.], dtype=np.float32),
}

WEATHER_TO_FLOAT = {"dry": 0.0, "damp": 0.5, "wet": 1.0}

# Wet-weather penalty if agent runs a dry compound in damp/wet conditions
_WET_COMPOUND_PENALTY = {"dry": 0.0, "damp": 0.5, "wet": 2.0}


class SimpleRaceEnv:
    """
    Single-car race simulator using the trained LSTM to predict next LapTimeNorm.

    Action space (6):
      0 = stay out
      1 = pit → SOFT
      2 = pit → MEDIUM
      3 = pit → HARD
      4 = pit → INTER
      5 = pit → WET

    Observation vector (19-dim):
      [lap_frac, laps_remaining_frac, tyre_life_norm,
       comp_soft, comp_medium, comp_hard, comp_inter, comp_wet,
       last_pred, slope,
       sc_active, vsc_active, red_flag_active,
       weather_state,
       gap_opp1, gap_opp2, gap_opp3,
       pit_count_norm, tyre_rule_satisfied_flag]

    FIA dry-race tyre rule:
      In a dry race, the driver must use at least two distinct dry compounds.
      If not satisfied by race end, a large terminal penalty is applied.
    """

    N_ACTIONS = 6

    def __init__(self, config: SimConfig | None = None):
        self.config = config if config is not None else SimConfig()

        self.total_laps = self.config.total_laps
        self.pit_loss_seconds = self.config.pit_loss_seconds
        self.base_lap_time = self.config.base_lap_time
        self.deg_k = self.config.deg_k
        self.deg_p = self.config.deg_p
        self.min_stint_laps = self.config.min_stint_laps
        self.reward_shaping = self.config.reward_shaping

        # Strategic shaping
        self.pit_stop_penalty = self.config.pit_stop_penalty
        self.same_compound_pit_penalty = self.config.same_compound_pit_penalty
        self.extra_stop_penalty = self.config.extra_stop_penalty

        # Safety Car config
        self.enable_safety_car = self.config.enable_safety_car
        self.sc_prob_per_lap = self.config.sc_prob_per_lap
        self.sc_duration_laps_min = self.config.sc_duration_laps_min
        self.sc_duration_laps_max = self.config.sc_duration_laps_max
        self.sc_lap_time_multiplier = self.config.sc_lap_time_multiplier
        self.sc_pit_loss_multiplier = self.config.sc_pit_loss_multiplier
        self.max_sc_events = getattr(self.config, "max_sc_events", 3)

        # VSC config
        self.enable_vsc = getattr(self.config, "enable_vsc", True)
        self.vsc_prob_per_lap = getattr(self.config, "vsc_prob_per_lap", 0.04)
        self.vsc_duration_laps_min = getattr(self.config, "vsc_duration_laps_min", 2)
        self.vsc_duration_laps_max = getattr(self.config, "vsc_duration_laps_max", 4)
        self.vsc_lap_time_multiplier = getattr(self.config, "vsc_lap_time_multiplier", 1.10)
        self.vsc_pit_loss_multiplier = getattr(self.config, "vsc_pit_loss_multiplier", 0.70)

        # Red Flag config
        self.enable_red_flag = getattr(self.config, "enable_red_flag", True)
        self.red_flag_prob_per_lap = getattr(self.config, "red_flag_prob_per_lap", 0.01)
        self.red_flag_duration_laps = getattr(self.config, "red_flag_duration_laps", 3)
        self.red_flag_free_pit = getattr(self.config, "red_flag_free_pit", True)

        # Tyre rule config
        self.is_dry_race = getattr(self.config, "is_dry_race", True)
        self.tyre_rule_penalty = getattr(self.config, "tyre_rule_penalty", 1000.0)

        # Weather config
        self.enable_weather = getattr(self.config, "enable_weather", False)
        self.initial_weather = getattr(self.config, "initial_weather", "dry")
        self.rain_probability = getattr(self.config, "rain_probability", 0.02)

        # Opponent model config
        self.enable_opponent_model = getattr(self.config, "enable_opponent_model", False)
        self.num_opponents = getattr(self.config, "num_opponents", 3)
        self.undercut_bonus_per_lap = getattr(self.config, "undercut_bonus_per_lap", 0.5)
        self.traffic_penalty_per_lap = getattr(self.config, "traffic_penalty_per_lap", 0.3)

        # Load LSTM once
        self.model, self.device = load_model()

        # RNG for stochastic race events
        self.rng = np.random.default_rng(42)

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.lap = 1
        self.compound = "MEDIUM"
        self.tyre_life = 1
        self.pit_cooldown = 0
        self.pit_count = 0

        # Track compounds used for FIA dry-race tyre legality
        self.used_compounds = {self.compound}
        self.tyre_rule_satisfied = False
        self.illegal_tyre_strategy = False

        # History window for LSTM: shape (10, 5)
        # Features: [LapTimeNorm, TyreLifeNorm, onehotS, onehotM, onehotH]
        self.window = np.zeros((10, 5), dtype=np.float32)

        self.total_time = 0.0
        self.laps_since_pit = 999

        # Safety Car state
        self.sc_active = False
        self.sc_remaining_laps = 0
        self.sc_start_lap = None
        self.sc_events_count = 0

        # VSC state
        self.vsc_active = False
        self.vsc_remaining_laps = 0

        # Red Flag state
        self.red_flag_active = False
        self.red_flag_remaining_laps = 0
        self.free_pit_available = False

        # Weather state
        self.weather = self.initial_weather if self.enable_weather else "dry"

        # Opponent gaps (normalized, positive = ahead, negative = behind)
        self._init_opponents()

        return self._get_state()

    # ------------------------------------------------------------------
    # Opponent helpers
    # ------------------------------------------------------------------

    def _init_opponents(self):
        if not self.enable_opponent_model:
            self.opponent_gaps = [0.0, 0.0, 0.0]
            return
        # Simple rule-based 3-car opponents
        # Gaps in seconds; positive = opponent is ahead of agent
        n = max(self.num_opponents, 3)
        self.opponent_gaps = list(self.rng.uniform(-30.0, 30.0, size=n).tolist())

    def _update_opponents(self, agent_lap_time: float):
        if not self.enable_opponent_model:
            self.opponent_gaps = [0.0, 0.0, 0.0]
            return
        updated = []
        for gap in self.opponent_gaps:
            # Opponents run a fixed pace slightly varied from the agent
            opp_lap_time = self.base_lap_time + float(self.rng.uniform(-1.5, 1.5))
            delta = agent_lap_time - opp_lap_time
            updated.append(gap + delta)
        self.opponent_gaps = updated

    def _gap_obs(self):
        gaps = self.opponent_gaps[:3] if len(self.opponent_gaps) >= 3 else (
            self.opponent_gaps + [0.0] * (3 - len(self.opponent_gaps))
        )
        max_gap = 60.0
        return [float(np.clip(g / max_gap, -1.0, 1.0)) for g in gaps]

    # ------------------------------------------------------------------
    # Safety Car / VSC / Red Flag helpers
    # ------------------------------------------------------------------

    def _maybe_trigger_neutralisation(self):
        # Only trigger if no neutralisation is currently active
        if self.sc_active or self.vsc_active or self.red_flag_active:
            return
        if self.lap < 5 or self.lap > self.total_laps - 5:
            return

        # Red Flag (rarest)
        if self.enable_red_flag and self.rng.random() < self.red_flag_prob_per_lap:
            self.red_flag_active = True
            self.red_flag_remaining_laps = self.red_flag_duration_laps
            if self.red_flag_free_pit:
                self.free_pit_available = True
            return

        # Full Safety Car
        if (self.enable_safety_car
                and self.sc_events_count < self.max_sc_events
                and self.rng.random() < self.sc_prob_per_lap):
            self.sc_active = True
            self.sc_events_count += 1
            self.sc_start_lap = self.lap
            self.sc_remaining_laps = int(
                self.rng.integers(self.sc_duration_laps_min, self.sc_duration_laps_max + 1)
            )
            return

        # VSC (more common, less severe)
        if self.enable_vsc and self.rng.random() < self.vsc_prob_per_lap:
            self.vsc_active = True
            self.vsc_remaining_laps = int(
                self.rng.integers(self.vsc_duration_laps_min, self.vsc_duration_laps_max + 1)
            )

    def _update_neutralisation(self):
        if self.sc_active:
            self.sc_remaining_laps -= 1
            if self.sc_remaining_laps <= 0:
                self.sc_active = False
                self.sc_remaining_laps = 0

        if self.vsc_active:
            self.vsc_remaining_laps -= 1
            if self.vsc_remaining_laps <= 0:
                self.vsc_active = False
                self.vsc_remaining_laps = 0

        if self.red_flag_active:
            self.red_flag_remaining_laps -= 1
            if self.red_flag_remaining_laps <= 0:
                self.red_flag_active = False
                self.red_flag_remaining_laps = 0

    # ------------------------------------------------------------------
    # Weather helper
    # ------------------------------------------------------------------

    def _update_weather(self):
        if not self.enable_weather:
            return
        r = float(self.rng.random())
        if self.weather == "dry":
            if r < self.rain_probability:
                self.weather = "damp"
        elif self.weather == "damp":
            if r < self.rain_probability:
                self.weather = "wet"
            elif r < 0.1:
                self.weather = "dry"
        elif self.weather == "wet":
            if r < 0.05:
                self.weather = "damp"

    # ------------------------------------------------------------------
    # Tyre rule helpers
    # ------------------------------------------------------------------

    def _dry_compounds_used(self):
        return self.used_compounds & {"SOFT", "MEDIUM", "HARD"}

    def _check_tyre_rule(self):
        if not self.is_dry_race:
            return True
        return len(self._dry_compounds_used()) >= 2

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_state(self):
        lap_frac = self.lap / self.total_laps
        laps_remaining_frac = (self.total_laps - self.lap) / self.total_laps
        tyre_life_norm = min(self.tyre_life, 60) / 60.0
        comp_oh = COMPOUND_TO_ONEHOT[self.compound]

        last_pred = float(self.window[-1, 0])
        prev_pred = float(self.window[-2, 0])
        slope = last_pred - prev_pred

        sc_flag = float(self.sc_active)
        vsc_flag = float(self.vsc_active)
        rf_flag = float(self.red_flag_active)
        weather_val = WEATHER_TO_FLOAT.get(self.weather, 0.0)

        gap_obs = self._gap_obs()
        pit_count_norm = min(self.pit_count, 5) / 5.0
        tyre_rule_ok = float(self._check_tyre_rule())

        return np.array(
            [
                lap_frac,
                laps_remaining_frac,
                tyre_life_norm,
                *comp_oh,           # 5 values
                last_pred,
                slope,
                sc_flag,
                vsc_flag,
                rf_flag,
                weather_val,
                *gap_obs,           # 3 values
                pit_count_norm,
                tyre_rule_ok,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        # Trigger neutralisation events before this lap runs
        self._maybe_trigger_neutralisation()
        self._update_weather()

        # Enforce minimum stint length (unless red flag free pit)
        free_pit_this_lap = False
        if self.red_flag_active and self.free_pit_available:
            free_pit_this_lap = True
            self.free_pit_available = False

        if self.laps_since_pit < self.min_stint_laps and action != 0 and not free_pit_this_lap:
            action = 0

        pit = action != 0
        pit_loss = 0.0
        pit_penalty = 0.0
        same_compound_penalty = 0.0
        extra_pit_penalty_now = 0.0
        weather_penalty = 0.0
        pitted_under_sc = False
        pitted_under_vsc = False
        pitted_under_rf = False

        if pit:
            old_compound = self.compound
            new_compound = COMPOUNDS[action - 1]
            pit_number = self.pit_count + 1

            self.compound = new_compound
            self.used_compounds.add(self.compound)
            self.tyre_life = 1
            self.pit_count = pit_number

            pit_penalty = self.pit_stop_penalty
            if new_compound == old_compound:
                same_compound_penalty = self.same_compound_pit_penalty
            if pit_number > 2:
                extra_pit_penalty_now = self.extra_stop_penalty

            # Reset LSTM window on new stint
            self.window[:] = 0.0
            tyre_life_norm = min(self.tyre_life, 60) / 60.0
            lstm_oh = _LSTM_COMPOUND_OH[self.compound]
            self.window[-1] = np.array([0.0, tyre_life_norm, *lstm_oh], dtype=np.float32)

            # Pit loss (discounted under SC/VSC/red flag)
            if free_pit_this_lap:
                pit_loss = 0.0
                pitted_under_rf = True
            elif self.sc_active:
                pit_loss = self.pit_loss_seconds * self.sc_pit_loss_multiplier
                pitted_under_sc = True
            elif self.vsc_active:
                pit_loss = self.pit_loss_seconds * self.vsc_pit_loss_multiplier
                pitted_under_vsc = True
            else:
                pit_loss = self.pit_loss_seconds

            self.laps_since_pit = 0
        else:
            self.tyre_life += 1
            self.laps_since_pit += 1

        # LSTM prediction (window uses 3-compound dry encoding)
        tyre_life_norm = min(self.tyre_life, 60) / 60.0
        lstm_oh = _LSTM_COMPOUND_OH[self.compound]

        pred_norm = predict_next_laptime_norm(self.window, model=self.model, device=self.device)

        new_row = np.array([pred_norm, tyre_life_norm, *lstm_oh], dtype=np.float32)
        self.window = np.roll(self.window, shift=-1, axis=0)
        self.window[-1] = new_row

        # Tyre degradation
        age = min(self.tyre_life, 60)
        deg = self.deg_k * ((age / 60.0) ** self.deg_p)

        # Base lap time
        running_lap_time = self.base_lap_time + pred_norm + deg

        # Neutralisation lap-time multiplier
        if self.sc_active:
            running_lap_time *= self.sc_lap_time_multiplier
        elif self.vsc_active:
            running_lap_time *= self.vsc_lap_time_multiplier
        elif self.red_flag_active:
            running_lap_time *= self.sc_lap_time_multiplier  # treat like SC during flag laps

        # Weather penalty: running dry compounds in wet conditions
        if self.weather in ("damp", "wet") and self.compound in ("SOFT", "MEDIUM", "HARD"):
            weather_penalty = _WET_COMPOUND_PENALTY[self.weather] * self.base_lap_time * 0.01

        lap_time = running_lap_time + pit_loss

        # Opponent interaction
        self._update_opponents(lap_time)
        gaps = self._gap_obs()
        undercut_bonus = 0.0
        traffic_penalty = 0.0
        if self.enable_opponent_model:
            # Reward if we just pitted and are now ahead of an opponent who hasn't
            if pit and pitted_under_sc and gaps[0] < 0:
                undercut_bonus = self.undercut_bonus_per_lap
            # Traffic penalty if closely behind an opponent
            if gaps[0] > -0.1 and gaps[0] < 0:
                traffic_penalty = self.traffic_penalty_per_lap

        strategy_penalty = (
            pit_penalty + same_compound_penalty + extra_pit_penalty_now + weather_penalty
            - undercut_bonus + traffic_penalty
        )

        if self.reward_shaping:
            base_r = (pred_norm + deg) * (
                self.sc_lap_time_multiplier if self.sc_active
                else (self.vsc_lap_time_multiplier if self.vsc_active else 1.0)
            )
            reward = -(base_r + pit_loss + strategy_penalty)
        else:
            reward = -(lap_time + strategy_penalty)

        self.total_time += lap_time + strategy_penalty

        info = {
            "lap": self.lap,
            "lap_time": lap_time,
            "pred_norm": pred_norm,
            "deg": deg,
            "pit": pit,
            "pit_count": self.pit_count,
            "compound": self.compound,
            "used_compounds": sorted(list(self.used_compounds)),
            "tyre_life": self.tyre_life,
            "total_time": self.total_time,
            "reward_shaping": self.reward_shaping,
            "min_stint_laps": self.min_stint_laps,
            # SC
            "sc_active": self.sc_active,
            "sc_remaining_laps": self.sc_remaining_laps,
            "sc_start_lap": self.sc_start_lap,
            "sc_events_count": self.sc_events_count,
            "safety_car_happened": self.sc_events_count > 0,
            "pitted_under_sc": pitted_under_sc,
            # VSC
            "vsc_active": self.vsc_active,
            "pitted_under_vsc": pitted_under_vsc,
            # Red Flag
            "red_flag_active": self.red_flag_active,
            "pitted_under_red_flag": pitted_under_rf,
            "free_pit": free_pit_this_lap,
            # Weather
            "weather": self.weather,
            "weather_penalty": weather_penalty,
            # Opponents
            "gap_ahead_norm": gaps[0] if gaps else 0.0,
            "gap_behind_norm": gaps[1] if len(gaps) > 1 else 0.0,
            # Pit details
            "pit_loss_used": pit_loss,
            "pit_stop_penalty_used": pit_penalty,
            "same_compound_pit_penalty_used": same_compound_penalty,
            "extra_pit_penalty_now_used": extra_pit_penalty_now,
            "is_dry_race": self.is_dry_race,
            "tyre_rule_satisfied_so_far": self._check_tyre_rule(),
            "illegal_tyre_strategy": False,
        }

        self.lap += 1
        done = self.lap > self.total_laps

        if done:
            self.tyre_rule_satisfied = self._check_tyre_rule()
            self.illegal_tyre_strategy = self.is_dry_race and not self.tyre_rule_satisfied

            tyre_rule_penalty_applied = 0.0
            if self.illegal_tyre_strategy:
                reward -= self.tyre_rule_penalty
                self.total_time += self.tyre_rule_penalty
                tyre_rule_penalty_applied = self.tyre_rule_penalty

            info["tyre_rule_satisfied_final"] = self.tyre_rule_satisfied
            info["illegal_tyre_strategy"] = self.illegal_tyre_strategy
            info["tyre_rule_penalty_applied"] = tyre_rule_penalty_applied
            info["extra_stop_penalty_applied"] = 0.0

        self._update_neutralisation()

        return self._get_state(), reward, done, info


def main():
    config = SimConfig(
        total_laps=52,
        enable_safety_car=True,
        sc_prob_per_lap=0.03,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.40,
        min_stint_laps=16,
        pit_stop_penalty=3.0,
        same_compound_pit_penalty=10.0,
        extra_stop_penalty=40.0,
        is_dry_race=True,
        tyre_rule_penalty=1000.0,
    )
    env = SimpleRaceEnv(config=config)
    state = env.reset(seed=123)

    print("Initial state:", state)
    print("State dim:", state.shape[0])

    done = False
    pitted_under_sc = False

    while not done:
        if env.sc_active and not pitted_under_sc:
            action = 1
            pitted_under_sc = True
        else:
            action = 0

        state, reward, done, info = env.step(action)

        print(
            f"Lap {info['lap']:02d} | action={action} | "
            f"lap_time={info['lap_time']:.3f} | reward={reward:.3f} | "
            f"tyre={info['compound']} age={info['tyre_life']} | "
            f"used={info['used_compounds']} | pits={info['pit_count']} | "
            f"SC={info['sc_active']} VSC={info['vsc_active']} RF={info['red_flag_active']} | "
            f"pit={info['pit']} | pit_loss={info['pit_loss_used']:.3f}"
        )

    print("Total time:", env.total_time)
    print("Illegal tyre strategy:", env.illegal_tyre_strategy)


if __name__ == "__main__":
    main()
