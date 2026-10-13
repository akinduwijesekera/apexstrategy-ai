import fastf1
import pandas as pd

# Enable caching (very important)
fastf1.Cache.enable_cache("data/cache")

print("Loading 2023 Silverstone Race session...")

session = fastf1.get_session(2023, "Silverstone", "R")
session.load()

print("Session loaded successfully!")

laps = session.laps

print("\nTotal laps loaded:", len(laps))

print("\nSample lap data:")
print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound']].head())