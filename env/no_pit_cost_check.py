from env.race_env import SimpleRaceEnv

def main():
    env = SimpleRaceEnv(total_laps=52, pit_loss_seconds=22.0, base_lap_time=95.0)
    env.reset()

    lap_times = []
    done = False
    while not done:
        _, _, done, info = env.step(0)  # always stay out
        lap_times.append(info["lap_time"])

    print("Total time:", info["total_time"])
    print("First lap time:", lap_times[0])
    print("Last lap time:", lap_times[-1])
    print("Increase last-first:", lap_times[-1] - lap_times[0])

if __name__ == "__main__":
    main()