from collections import deque
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import SimConfig
from env.gym_race_env import GymRaceEnv


# ----------------------------
# Helper: evaluate greedily (no exploration) and return total_time
# ----------------------------
def evaluate_greedy(env, qnet, device, seed: int = 123):
    obs, _ = env.reset(seed=seed)
    done = False
    last_info = None

    while not done:
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = int(torch.argmax(qnet(s), dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_info = info

    return last_info["total_time"], last_info


# ----------------------------
# Hyperparameters
# ----------------------------
SEED = 42
EPISODES = 400
GAMMA = 0.99

BATCH_SIZE = 64
REPLAY_SIZE = 50_000
MIN_REPLAY_TO_LEARN = 2_000

LR = 1e-3
TARGET_UPDATE_EVERY = 200  # steps

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8_000  # linear decay over steps


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s = np.stack([b.s for b in batch], axis=0)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch], axis=0)
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


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


def linear_epsilon(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)


def train_dqn(
    config: SimConfig | None = None,
    seed: int = SEED,
    episodes: int = EPISODES,
    final_model_path: str = "models/dqn_qnet.pt",
    best_model_path: str = "models/dqn_qnet_best.pt",
    use_double_dqn: bool = False,
):
    """
    Train DQN or Double DQN using the provided simulator config.
    """
    config = config if config is not None else SimConfig()

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GymRaceEnv(config=config)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Algorithm:", "Double DQN" if use_double_dqn else "DQN")
    print("Observation dim:", obs_dim)
    print("Safety Car enabled:", config.enable_safety_car)
    print("Dry race:", config.is_dry_race)
    print("Tyre rule penalty:", config.tyre_rule_penalty)
    print("Final model path:", final_model_path)
    print("Best model path:", best_model_path)

    q = QNet(obs_dim, n_actions).to(device)
    q_tgt = QNet(obs_dim, n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=LR)
    buffer = ReplayBuffer(REPLAY_SIZE)

    global_step = 0
    recent_returns = deque(maxlen=20)

    best_time = float("inf")
    best_ep = -1

    final_dir = os.path.dirname(final_model_path)
    best_dir = os.path.dirname(best_model_path)

    if final_dir:
        os.makedirs(final_dir, exist_ok=True)
    if best_dir:
        os.makedirs(best_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        pit_count = 0

        while not done:
            eps = linear_epsilon(global_step)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = q(s_t)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            if info.get("pit", False):
                pit_count += 1
            done = terminated or truncated

            buffer.push(Transition(obs, action, reward, next_obs, done))

            obs = next_obs
            ep_return += reward
            global_step += 1

            if len(buffer) >= MIN_REPLAY_TO_LEARN:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)

                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a, dtype=torch.int64).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)

                q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if use_double_dqn:
                        next_actions = q(s2).argmax(dim=1, keepdim=True)
                        next_q_target = q_tgt(s2).gather(1, next_actions).squeeze(1)
                        target = r + GAMMA * (1.0 - d) * next_q_target
                    else:
                        max_q_s2 = q_tgt(s2).max(dim=1)[0]
                        target = r + GAMMA * (1.0 - d) * max_q_s2

                loss = nn.MSELoss()(q_sa, target)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if global_step % TARGET_UPDATE_EVERY == 0:
                    q_tgt.load_state_dict(q.state_dict())

        recent_returns.append(ep_return)

        if ep % 10 == 0:
            avg20 = sum(recent_returns) / len(recent_returns)
            print(
                f"Episode {ep:03d} | return={ep_return:.1f} | avg20={avg20:.1f} | "
                f"eps={linear_epsilon(global_step):.3f} | pits={pit_count} | buffer={len(buffer)}"
            )

        if ep % 20 == 0:
            eval_time, eval_info = evaluate_greedy(env, q, device, seed=123)
            if eval_time < best_time:
                best_time = eval_time
                best_ep = ep
                torch.save(q.state_dict(), best_model_path)
                print(
                    f"  >> New BEST model at ep {ep}: total_time={best_time:.3f} "
                    f"(saved to {best_model_path}) | "
                    f"SC happened={eval_info.get('safety_car_happened', False)} | "
                    f"SC start lap={eval_info.get('sc_start_lap', None)} | "
                    f"used_compounds={eval_info.get('used_compounds', [])} | "
                    f"illegal_tyre_strategy={eval_info.get('illegal_tyre_strategy', False)}"
                )

    torch.save(q.state_dict(), final_model_path)
    print(f"Saved model to {final_model_path}")
    print(f"Best greedy total_time={best_time:.3f} achieved at ep {best_ep} ({best_model_path})")

    return {
        "best_time": best_time,
        "best_ep": best_ep,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "config": config,
        "episodes": episodes,
        "seed": seed,
        "use_double_dqn": use_double_dqn,
    }


def main():
    experiments = [
        {
            "name": "DDQN no-SC baseline (tyre-rule aware)",
            "config": SimConfig(
                total_laps=52,
                enable_safety_car=False,
                sc_prob_per_lap=0.0,
                sc_duration_laps_min=2,
                sc_duration_laps_max=5,
                sc_lap_time_multiplier=1.20,
                sc_pit_loss_multiplier=0.40,
                min_stint_laps=16,
                is_dry_race=True,
                tyre_rule_penalty=1000.0,
            ),
            "final_model_path": "models/ddqn_no_sc_tyre_rule.pt",
            "best_model_path": "models/ddqn_no_sc_tyre_rule_best.pt",
            "use_double_dqn": True,
        },
        {
            "name": "DDQN Safety Car aware + tyre-rule aware",
            "config": SimConfig(
                total_laps=52,
                enable_safety_car=True,
                sc_prob_per_lap=0.03,
                sc_duration_laps_min=2,
                sc_duration_laps_max=5,
                sc_lap_time_multiplier=1.20,
                sc_pit_loss_multiplier=0.40,
                min_stint_laps=16,
                is_dry_race=True,
                tyre_rule_penalty=1000.0,
            ),
            "final_model_path": "models/ddqn_safety_car.pt",
            "best_model_path": "models/ddqn_safety_car_best.pt",
            "use_double_dqn": True,
        },
    ]

    all_results = []

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"Training: {exp['name']}")
        print("=" * 80)

        result = train_dqn(
            config=exp["config"],
            final_model_path=exp["final_model_path"],
            best_model_path=exp["best_model_path"],
            use_double_dqn=exp["use_double_dqn"],
        )
        all_results.append((exp["name"], result))

    print("\n" + "=" * 80)
    print("FINAL TRAINING SUMMARY")
    print("=" * 80)
    for name, result in all_results:
        print(f"\n{name}")
        print(f"  Best time:   {result['best_time']:.3f}")
        print(f"  Best episode:{result['best_ep']}")
        print(f"  Final model: {result['final_model_path']}")
        print(f"  Best model:  {result['best_model_path']}")
        print(f"  Algorithm:   {'Double DQN' if result['use_double_dqn'] else 'DQN'}")


if __name__ == "__main__":
    main()