import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import SimConfig
from env.race_env import SimpleRaceEnv


class GymRaceEnv(gym.Env):
    """
    Gymnasium wrapper around SimpleRaceEnv.

    Observation: dynamic state vector from simulator
    Action space: 4 discrete actions
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig | None = None):
        super().__init__()

        self.config = config if config is not None else SimConfig()
        self.sim = SimpleRaceEnv(config=self.config)

        sample_obs = self.sim.reset().astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.sim.N_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed).astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.sim.step(int(action))
        obs = obs.astype(np.float32)

        terminated = done
        truncated = False
        return obs, float(reward), terminated, truncated, info


def main():
    config = SimConfig(
        total_laps=52,
        enable_safety_car=True,
        sc_prob_per_lap=0.03,
        sc_duration_laps_min=2,
        sc_duration_laps_max=5,
        sc_lap_time_multiplier=1.20,
        sc_pit_loss_multiplier=0.40,
        is_dry_race=True,
        tyre_rule_penalty=1000.0,
    )
    env = GymRaceEnv(config=config)

    obs, info = env.reset(seed=123)
    print("obs:", obs)
    print("obs shape:", obs.shape)

    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        print(
            f"action={action} "
            f"lap_time={info['lap_time']:.3f} "
            f"reward={reward:.3f} "
            f"compound={info.get('compound')} "
            f"used={info.get('used_compounds')} "
            f"sc={info.get('sc_active', False)} "
            f"sc_rem={info.get('sc_remaining_laps', 0)} "
            f"pit={info.get('pit', False)} "
            f"pit_loss={info.get('pit_loss_used', 0.0):.3f} "
            f"illegal={info.get('illegal_tyre_strategy', False)}"
        )

    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()
