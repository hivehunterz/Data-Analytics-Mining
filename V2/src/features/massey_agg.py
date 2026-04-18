"""
Massey ordinal aggregation — Kevin's dynamic system selection.

For each training season window, select all SystemName values that have
full coverage (≥95% of target seasons). Aggregate each team's latest
ranking across those systems into mean / median / min.

Kevin's finding: dynamic selection beat hand-picked subsets by 0.00246 Brier.
"""
import numpy as np
import pandas as pd


def find_eligible_systems(
    massey: pd.DataFrame,
    seasons: list,
    coverage_threshold: float = 0.95,
) -> list:
    """
    Return SystemName values that appear in at least `coverage_threshold`
    of the target seasons.
    """
    counts = (
        massey[massey["Season"].isin(seasons)]
        .groupby("SystemName")["Season"].nunique()
    )
    threshold = int(np.ceil(coverage_threshold * len(seasons)))
    eligible = counts[counts >= threshold].index.tolist()
    return eligible


def aggregate_massey(
    massey: pd.DataFrame,
    seasons: list | None = None,
    systems: list | None = None,
    latest_window_days: int = 14,
) -> pd.DataFrame:
    """
    Aggregate Massey ordinals per (Season, TeamID).

    Parameters
    ----------
    massey : pd.DataFrame
        Columns: Season, RankingDayNum, SystemName, TeamID, OrdinalRank
    seasons : list, optional
        If provided, restrict aggregation to these seasons.
    systems : list, optional
        If provided, use only these SystemNames. Else use all.
    latest_window_days : int
        Keep only ranks from last N RankingDayNum of each season (Brendan's
        "last 2 weeks" filter).

    Returns
    -------
    DataFrame with columns Season, TeamID, MasseyMean, MasseyMedian, MasseyMin.
    """
    df = massey.copy()
    if seasons is not None:
        df = df[df["Season"].isin(seasons)]
    if systems is not None:
        df = df[df["SystemName"].isin(systems)]

    # Keep only last two weeks of each season
    max_day = df.groupby("Season")["RankingDayNum"].transform("max")
    df = df[df["RankingDayNum"] >= max_day - latest_window_days]

    # Per (Season, SystemName, TeamID), take latest RankingDayNum
    latest = (
        df.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"])
        .tail(1)
    )

    # Aggregate across systems
    agg = latest.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        MasseyMean="mean", MasseyMedian="median", MasseyMin="min",
    ).reset_index()
    return agg
