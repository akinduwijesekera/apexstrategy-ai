import numpy as np
from env.race_env import SimpleRaceEnv

# Actions: 0=stay, 1=pit soft, 2=pit medium, 3=pit hard
PIT_ACTION = 2  # try MEDIUM first


def run_strategy(pit_lap=None, pit_action=PIT_ACTION):
    env = SimpleRaceEnv(total_laps=52, pit_loss_seconds=28.0, base_lap_time=95.0)
    env.reset()

    done = False
    while not done:
        action = 0
        if pit_lap is not None and env.lap == pit_lap:
            action = pit_action
        _, _, done, info = env.step(action)

    return info["total_time"]


def main():
    no_pit_time = run_strategy(pit_lap=None)
    print("No pit time:", no_pit_time)

    best = (None, float("inf"))
    for pit_lap in range(2, 52):  # don't pit lap 1 or after finish
        t = run_strategy(pit_lap=pit_lap)
        if t < best[1]:
            best = (pit_lap, t)

    print("Best 1-stop:", best)
    print("1-stop improvement vs no-pit:", no_pit_time - best[1])


if __name__ == "__main__":
    main()