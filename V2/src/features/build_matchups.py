"""
Build pairwise matchup training data.

Convention (Brendan style): for each tournament game, the two teams are
ordered by TeamID with Team1.id < Team2.id. Features are Team1 - Team2
differences. Label y = 1 if Team1 won, else 0. This makes the target
symmetric and prevents "team A always wins" bias.
"""
import numpy as np
import pandas as pd


def _merge_features(
    game_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    team_col: str,
    feature_cols: list,
    suffix: str,
) -> pd.DataFrame:
    sub = feat_df[["Season", "TeamID"] + feature_cols].copy()
    sub = sub.rename(columns={"TeamID": team_col, **{c: c + suffix for c in feature_cols}})
    return game_df.merge(sub, on=["Season", team_col], how="left")


def build_tourney_matchups(
    tourney_compact: pd.DataFrame,
    feat_df: pd.DataFrame,
    feature_cols: list,
    seeds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each tournament game, produce one canonical row with:
      - Season, DayNum
      - T1, T2  (T1 < T2 by TeamID)
      - diff_{feature}  = T1_feature - T2_feature
      - label           = 1 if T1 won else 0
      - T1_SeedNum, T2_SeedNum  (if seeds_df provided)
    """
    g = tourney_compact[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    g["T1"] = g[["WTeamID", "LTeamID"]].min(axis=1)
    g["T2"] = g[["WTeamID", "LTeamID"]].max(axis=1)
    g["label"] = (g["WTeamID"] == g["T1"]).astype(int)
    g = g.drop(columns=["WTeamID", "LTeamID"])

    g = _merge_features(g, feat_df, "T1", feature_cols, "_T1")
    g = _merge_features(g, feat_df, "T2", feature_cols, "_T2")

    for c in feature_cols:
        g[f"diff_{c}"] = g[f"{c}_T1"] - g[f"{c}_T2"]
    # Drop intermediate _T1/_T2 columns unless needed downstream
    g = g.drop(columns=[f"{c}_T1" for c in feature_cols] +
                       [f"{c}_T2" for c in feature_cols])

    if seeds_df is not None:
        seeds = seeds_df[["Season", "TeamID", "SeedNum"]]
        g = g.merge(seeds.rename(columns={"TeamID": "T1", "SeedNum": "T1_SeedNum"}),
                    on=["Season", "T1"], how="left")
        g = g.merge(seeds.rename(columns={"TeamID": "T2", "SeedNum": "T2_SeedNum"}),
                    on=["Season", "T2"], how="left")
        g["diff_SeedNum"] = g["T1_SeedNum"] - g["T2_SeedNum"]

    return g


def build_submission_matchups(
    sample_sub: pd.DataFrame,
    feat_df: pd.DataFrame,
    feature_cols: list,
    seeds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Same as build_tourney_matchups but for Stage 2 submission pairs.
    sample_sub has columns Season, T1, T2 (T1 < T2 already).
    No label (unknown outcomes).
    """
    g = sample_sub[["Season", "T1", "T2"]].copy()
    g = _merge_features(g, feat_df, "T1", feature_cols, "_T1")
    g = _merge_features(g, feat_df, "T2", feature_cols, "_T2")
    for c in feature_cols:
        g[f"diff_{c}"] = g[f"{c}_T1"] - g[f"{c}_T2"]
    g = g.drop(columns=[f"{c}_T1" for c in feature_cols] +
                       [f"{c}_T2" for c in feature_cols])

    if seeds_df is not None:
        seeds = seeds_df[["Season", "TeamID", "SeedNum"]]
        g = g.merge(seeds.rename(columns={"TeamID": "T1", "SeedNum": "T1_SeedNum"}),
                    on=["Season", "T1"], how="left")
        g = g.merge(seeds.rename(columns={"TeamID": "T2", "SeedNum": "T2_SeedNum"}),
                    on=["Season", "T2"], how="left")
        g["diff_SeedNum"] = g["T1_SeedNum"] - g["T2_SeedNum"]
    return g
