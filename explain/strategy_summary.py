import pandas as pd


def summarise_trace(trace_path="data/processed/dqn_eval_trace.csv"):
    df = pd.read_csv(trace_path)

    total_time = float(df["total_time"].iloc[-1])

    pit_rows = df[df["pit"] == 1]
    pit_laps = pit_rows["lap"].tolist()
    pit_compounds = pit_rows["compound"].tolist()

    pit_plan_str = (
        ", ".join([f"lap {l} -> {c}" for l, c in zip(pit_laps, pit_compounds)])
        if pit_laps
        else "no pits"
    )

    # Compounds used across the whole race
    used_compounds = sorted(df["compound"].dropna().astype(str).unique().tolist())
    num_unique_compounds = len(used_compounds)

    # Tyre legality fields if present in trace
    is_dry_race = bool(df["is_dry_race"].iloc[-1]) if "is_dry_race" in df.columns else True
    tyre_rule_satisfied_final = (
        bool(df["tyre_rule_satisfied_final"].dropna().iloc[-1])
        if "tyre_rule_satisfied_final" in df.columns and df["tyre_rule_satisfied_final"].dropna().shape[0] > 0
        else (num_unique_compounds >= 2 if is_dry_race else True)
    )
    illegal_tyre_strategy = (
        bool(df["illegal_tyre_strategy"].iloc[-1])
        if "illegal_tyre_strategy" in df.columns
        else (is_dry_race and not tyre_rule_satisfied_final)
    )
    tyre_rule_penalty_applied = (
        float(df["tyre_rule_penalty_applied"].iloc[-1])
        if "tyre_rule_penalty_applied" in df.columns
        else 0.0
    )

    # Build stint segments: pit is logged on the pit-loss lap, next lap starts new stint
    stints = []
    start_lap = int(df["lap"].iloc[0])

    for pit_lap in pit_laps:
        stint_df = df[(df["lap"] >= start_lap) & (df["lap"] <= pit_lap)]
        if not stint_df.empty:
            stints.append(stint_df)
        start_lap = int(pit_lap) + 1

    final_stint = df[df["lap"] >= start_lap]
    if not final_stint.empty:
        stints.append(final_stint)

    # Degradation stats excluding pit-loss laps
    stint_summaries = []
    for i, s in enumerate(stints, start=1):
        comp = str(s["compound"].iloc[0])
        start = int(s["lap"].iloc[0])
        end = int(s["lap"].iloc[-1])

        s_nopit = s[s["pit"] == 0] if "pit" in s.columns else s
        if s_nopit.empty:
            s_nopit = s

        lap_time_start = float(s_nopit["lap_time"].iloc[0])
        lap_time_end = float(s_nopit["lap_time"].iloc[-1])
        lap_time_delta = lap_time_end - lap_time_start
        mean_lap_time = float(s_nopit["lap_time"].mean())

        stint_summaries.append({
            "stint": i,
            "compound": comp,
            "start_lap": start,
            "end_lap": end,
            "lap_time_start": lap_time_start,
            "lap_time_end": lap_time_end,
            "lap_time_delta": lap_time_delta,
            "mean_lap_time": mean_lap_time,
            "n_laps_used_for_stats": int(len(s_nopit)),
        })

    return {
        "total_time_s": total_time,
        "pit_laps": pit_laps,
        "pit_compounds": pit_compounds,
        "pit_plan_str": pit_plan_str,
        "used_compounds": used_compounds,
        "num_unique_compounds": num_unique_compounds,
        "is_dry_race": is_dry_race,
        "tyre_rule_satisfied_final": tyre_rule_satisfied_final,
        "illegal_tyre_strategy": illegal_tyre_strategy,
        "tyre_rule_penalty_applied": tyre_rule_penalty_applied,
        "stints": stint_summaries,
    }