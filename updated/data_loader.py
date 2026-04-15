"""
Stages 1-3: Load raw data, apply 2026 results, clean, engineer features.
"""
import pandas as pd
import numpy as np
from config import (DATA_RAW, DATA_UPDATED, RESULTS_2026, RESULT_ORDER,
                    DROP_COLS, SHOOT_COLS, ADV_COLS, MIN_GP, MIN_MPG)
from utils import log_step


def load_and_update():
    """Stage 1: Load CSV and patch 2026 tournament results."""
    log_step(f"Loading {DATA_RAW} ...")
    df = pd.read_csv(DATA_RAW)
    log_step(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    log_step(f"Seasons: {df['season_year'].min()} – {df['season_year'].max()}")

    updated = 0
    for team, (wins, result) in RESULTS_2026.items():
        mask = (df["season_year"] == 2026) & (df["team"] == team)
        if mask.sum() > 0:
            df.loc[mask, "games_won"] = wins
            df.loc[mask, "tournament_result"] = result
            updated += mask.sum()
    log_step(f"Updated {updated} rows for 2026 tournament results")

    dist = df[df["season_year"] == 2026].groupby("tournament_result")["team"].nunique()
    expected = {"R68":4,"R64":32,"R32":16,"S16":8,"E8":4,"F4":2,"NCG":1,"Champion":1}
    print("\n  2026 result distribution:")
    for r in ["R68","R64","R32","S16","E8","F4","NCG","Champion"]:
        n = dist.get(r, 0)
        flag = " <<" if n != expected.get(r, 0) else ""
        print(f"    {r:<10s} {n:>4d}  (expected {expected.get(r, '?')}){flag}")

    df.to_csv(DATA_UPDATED, index=False)
    log_step(f"Saved {DATA_UPDATED}")
    return df


def clean(df):
    """Stage 2: Filter, impute, create target variable."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    log_step("Dropped metadata columns")

    before = len(df)
    df = df[(df["games_played"] >= MIN_GP) & (df["minutes_per_game"] >= MIN_MPG)].copy()
    log_step(f"Filtered: {before} -> {len(df)} players (>={MIN_GP} GP, >={MIN_MPG} MPG)")

    for col in SHOOT_COLS + ADV_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df["conference"]    = df["conference"].fillna("Unknown")
    df["position"]      = df["position"].fillna("Unknown")
    df["games_started"] = df["games_started"].fillna(0)
    df["tov_per_game"]  = df["tov_per_game"].fillna(0)
    log_step("Imputed missing values with median")

    df["result_numeric"] = df["tournament_result"].map(RESULT_ORDER).fillna(0).astype(int)
    df["deep_run"] = (df["result_numeric"] >= 3).astype(int)

    # Upset contribution flag
    seed_win_map = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,
                    9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1}
    df["upset_contribution"] = (
        (df["tournament_seed"].map(seed_win_map) == 1) & (df["games_won"] > 0)
    ).astype(int)

    log_step(f"Deep run (Sweet 16+): {df['deep_run'].sum()} players ({df['deep_run'].mean():.1%})")
    log_step(f"Unique teams: {df['team'].nunique()}, Unique players: {df['player_name'].nunique()}")
    return df


def engineer_features(df):
    """Stage 3: Create composite features."""
    df["scoring_efficiency"]  = df["pts_per_game"] * df["ts_pct"]
    df["playmaking_index"]    = df["ast_per_game"] - df["tov_per_game"]
    df["defensive_impact"]    = df["stl_per_game"] + df["blk_per_game"]
    df["rebounding_rate"]     = df["trb_per_game"] / df["minutes_per_game"].clip(lower=1) * 40
    df["usage_efficiency"]    = df["per"] / df["usg_pct"].clip(lower=1) * 100
    df["starter_ratio"]       = df["games_started"] / df["games_played"].clip(lower=1)
    df["win_shares_per_game"] = df["ws"] / df["games_played"].clip(lower=1)
    df["team_win_pct"]        = df["team_wins"] / (df["team_wins"] + df["team_losses"]).clip(lower=1)
    log_step("Created 7 engineered features")
    return df


def load_pipeline():
    """Run stages 1-3 and return the cleaned, engineered DataFrame."""
    df = load_and_update()
    df = clean(df)
    df = engineer_features(df)
    return df
