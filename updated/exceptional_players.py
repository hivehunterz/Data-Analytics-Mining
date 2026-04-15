"""
Stage 7: Exceptional player identification via weighted z-score impact.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, HOLDOUT_YEAR
from utils import log_step, save_fig


def run_exceptional_identification(df):
    """Compute impact scores and identify exceptional / upset-star players."""
    df_pre = df[df["season_year"] < HOLDOUT_YEAR]
    seed_expected_wins = df_pre.groupby("tournament_seed")["games_won"].mean()
    df["expected_wins"]      = df["tournament_seed"].map(seed_expected_wins)
    df["wins_over_expected"] = df["games_won"] - df["expected_wins"]
    log_step(f"Expected wins computed from 2011-{HOLDOUT_YEAR-1} ({len(df_pre)} players)")

    print("\n  Expected wins by seed:")
    for seed in range(1, 17):
        if seed in seed_expected_wins.index:
            print(f"    Seed {seed:2d}: {seed_expected_wins[seed]:.2f}")

    impact_cols = ["bpm","dbpm","obpm","ws","dws","ows","per","efg_pct","ts_pct"]
    pre_means = {c: df_pre[c].mean() for c in impact_cols}
    pre_stds  = {c: df_pre[c].std()  for c in impact_cols}

    impact_z = pd.DataFrame()
    for col in impact_cols:
        impact_z[col] = (df[col] - pre_means[col]) / pre_stds[col]

    weights = {"bpm":0.635,"dbpm":0.717,"obpm":0.419,"ws":0.315,"dws":0.307,
               "ows":0.281,"per":0.191,"efg_pct":0.139,"ts_pct":0.136}
    total_w = sum(weights.values())
    df["impact_score"] = sum(impact_z[c]*weights[c] for c in impact_cols) / total_w

    df["is_exceptional"] = (
        (df["impact_score"] >= df["impact_score"].quantile(0.95)) &
        (df["wins_over_expected"] > 0)
    )
    df["upset_star"] = (
        (df["upset_contribution"] == 1) &
        (df["impact_score"] >= df["impact_score"].quantile(0.85))
    )

    exceptional = df[df["is_exceptional"]].sort_values("impact_score", ascending=False)
    upset_stars = df[df["upset_star"]].sort_values("impact_score", ascending=False)
    print(f"\n  Exceptional players: {len(exceptional)}  |  Upset stars: {len(upset_stars)}")

    print(f"\n  --- TOP 25 EXCEPTIONAL TOURNAMENT PERFORMERS ---")
    print(f"  {'Player':<22s} {'Team':<18s} {'Yr':>4s} {'Seed':>4s} {'Result':<10s} "
          f"{'Won':>3s} {'PPG':>5s} {'BPM':>5s} {'DBPM':>5s} {'Score':>6s}")
    print("  " + "-"*95)
    for _, row in exceptional.head(25).iterrows():
        print(f"  {row['player_name']:<22s} {row['team']:<18s} {row['season_year']:>4.0f} "
              f"{row['tournament_seed']:>4d} {row['tournament_result']:<10s} "
              f"{row['games_won']:>3d} {row['pts_per_game']:>5.1f} "
              f"{row['bpm']:>5.1f} {row['dbpm']:>5.1f} {row['impact_score']:>6.2f}")

    print(f"\n  --- TOP 15 UPSET STARS ---")
    print(f"  {'Player':<22s} {'Team':<18s} {'Yr':>4s} {'Seed':>4s} {'Result':<10s} "
          f"{'PPG':>5s} {'BPM':>5s} {'DBPM':>5s} {'Score':>6s}")
    print("  " + "-"*80)
    for _, row in upset_stars.head(15).iterrows():
        print(f"  {row['player_name']:<22s} {row['team']:<18s} {row['season_year']:>4.0f} "
              f"{row['tournament_seed']:>4d} {row['tournament_result']:<10s} "
              f"{row['pts_per_game']:>5.1f} {row['bpm']:>5.1f} {row['dbpm']:>5.1f} "
              f"{row['impact_score']:>6.2f}")

    exc_out = exceptional[["player_name","team","season_year","tournament_seed",
        "tournament_result","games_won","expected_wins","wins_over_expected",
        "pts_per_game","trb_per_game","ast_per_game","bpm","dbpm","obpm",
        "ws","per","impact_score"]].round(3)
    exc_out.to_csv(f"{OUTPUT_DIR}/exceptional_players.csv", index=False)
    log_step(f"[saved] exceptional_players.csv  ({len(exc_out)} rows)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["impact_score"], bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].axvline(df["impact_score"].quantile(0.95), color="red", linestyle="--",
                    label=f"95th pct ({df['impact_score'].quantile(0.95):.2f})")
    axes[0].set_title("Impact Score Distribution")
    axes[0].legend()
    exc_by_seed = exceptional.groupby("tournament_seed").size()
    exc_rate = (exc_by_seed / df.groupby("tournament_seed").size() * 100).fillna(0)
    axes[1].bar(exc_rate.index, exc_rate.values, color="#e74c3c", alpha=0.7)
    axes[1].set_title("% Exceptional Players by Seed")
    axes[1].set_xticks(range(1,17))
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("% Flagged Exceptional")
    plt.tight_layout()
    save_fig("stage7_exceptional_players")

    return df
