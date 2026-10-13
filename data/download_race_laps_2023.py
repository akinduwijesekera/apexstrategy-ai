import shutil
import time
from pathlib import Path

import fastf1
import pandas as pd


SEASON = 2023
RACES = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Azerbaijan",
    "Miami",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "Qatar",
    "United States",
    "Mexico",
    "Brazil",
    "Las Vegas",
    "Abu Dhabi",
]

OUT_DIR = Path("data/raw")
CACHE_DIR = Path("data/cache")
USE_FULL_SEASON = True

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3
CLEAR_CACHE_ON_RETRY = True


def get_full_season_races(season: int) -> list[str]:
    """Return all championship race names for the season, excluding testing."""
    schedule = fastf1.get_event_schedule(season)

    if "EventFormat" in schedule.columns:
        schedule = schedule[schedule["EventFormat"] != "testing"]

    if "Session5" in schedule.columns:
        schedule = schedule[schedule["Session5"] == "Race"]

    return schedule["EventName"].tolist()



def get_races_to_download() -> list[str]:
    """Return either the manual race list or the full season list."""
    if USE_FULL_SEASON:
        return get_full_season_races(SEASON)
    return RACES


def reset_cache():
    """Delete and recreate the local FastF1 cache directory."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))



def load_session_laps_with_retry(race_name: str) -> pd.DataFrame:
    """Load laps for a race, retrying if FastF1 returns incomplete session data."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1 and CLEAR_CACHE_ON_RETRY:
                print(f"Retry {attempt}/{MAX_RETRIES} for {race_name}: clearing cache and trying again...")
                reset_cache()
                time.sleep(RETRY_WAIT_SECONDS)

            session = fastf1.get_session(SEASON, race_name, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            laps = session.laps
            if laps is None or laps.empty:
                raise ValueError(f"No lap data available for {race_name}")

            return laps

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed for {race_name}: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(
        f"Could not load laps for {race_name} after {MAX_RETRIES} attempts: {last_error}"
    )


def timedelta_to_seconds(td) -> float:
    if pd.isna(td):
        return float("nan")
    return td.total_seconds()


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()

    keep_cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "Stint",
        "TyreLife",
        "FreshTyre",
        "IsAccurate",
        "PitInTime",
        "PitOutTime",
        "TrackStatus",
    ]
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    if "LapTime" in df.columns:
        df = df.dropna(subset=["LapTime"])

    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]

    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]
    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]

    df["LapTimeSeconds"] = df["LapTime"].apply(timedelta_to_seconds)
    df = df[(df["LapTimeSeconds"] > 50) & (df["LapTimeSeconds"] < 200)]

    if "LapNumber" in df.columns:
        df["LapNumber"] = df["LapNumber"].astype(int)

    return df.reset_index(drop=True)


def download_and_save_race(race_name: str) -> dict:
    print(f"\n=== {SEASON} {race_name} (Race) ===")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = race_name.lower().replace(" ", "_").replace("-", "_")
    out_path = OUT_DIR / f"{SEASON}_{safe_name}_race_laps.csv"

    laps = load_session_laps_with_retry(race_name)

    cleaned = clean_laps(laps)
    cleaned.to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows: {len(cleaned)}")
    return {
        "race": race_name,
        "status": "success",
        "rows": len(cleaned),
        "path": str(out_path),
    }


def print_summary(results: list[dict]):
    total_requested = len(results)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]

    total_rows = sum(r.get("rows", 0) for r in successes)

    print("\n" + "=" * 60)
    print(f"SUMMARY FOR {SEASON}")
    print("=" * 60)
    print(f"Races requested : {total_requested}")
    print(f"Successful      : {len(successes)}")
    print(f"Failed          : {len(failures)}")
    print(f"Total rows saved: {total_rows}")

    if successes:
        print("\nSuccessful races:")
        for r in successes:
            print(f" - {r['race']}: {r['rows']} rows -> {r['path']}")

    if failures:
        print("\nFailed races:")
        for r in failures:
            print(f" - {r['race']}: {r['error']}")


def main():
    reset_cache()

    races = get_races_to_download()

    results = []

    for race in races:
        try:
            result = download_and_save_race(race)
            results.append(result)
        except Exception as e:
            print(f"Failed for {race}: {e}")
            results.append(
                {
                    "race": race,
                    "status": "failed",
                    "error": str(e),
                }
            )

    print(f"\nProcessed {len(races)} races for {SEASON}")
    print_summary(results)


if __name__ == "__main__":
    main()
