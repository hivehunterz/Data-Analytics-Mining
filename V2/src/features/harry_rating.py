"""
Harry's custom rating + opponent quality points won features.

Harry's harry_Rating formula:
    harry_Rating = NetEff
                   * (1 + opp_quality_MinMax)     # SOS scaler
                   * power_conf_MinMax            # power conference bonus
                   * top12_MinMax                 # AP Poll week 6 top-12

Scaler ranges (from his writeup):
    men:   opp_pts in [-0.55, 0.55]
           power_conf in [1, 1.3]
           top12      in [1, 1.2]
    women: opp_pts in [-0.5, 0.5]
           power_conf in [1, 1.1]
           top12      skipped (no women's poll)

Opponent quality points:
    Tier 1 (seed <=4 in this year's tournament): 6
    Tier 2 (seed 5-16 in this year's tournament): 4
    Tier 3 (secondary tournament):               2
    Tier 4 (no tournament):                       0.25

opp_qlty_pts_won = sum of those points on each team's WINS.
Then MinMax-scale per season/gender into [-0.55, 0.55] etc.

Power conferences (men):
    big_ten, acc, sec, big_twelve, big_east, pac_twelve
Power conferences (women): same list applies (all have strong programs).

AP Poll top 12 is not in the Kaggle competition data, so we skip
top12_MinMax (set to 1.0 neutral). The lift from this feature is
marginal; it's mainly a men's signal Harry added for prestige-not-
efficiency upside.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


POWER_CONFS = {
    "big_ten", "acc", "sec", "big_twelve", "big_east", "pac_twelve",
}


def _assign_opp_quality_points(opp_seed: float, opp_in_secondary: bool) -> float:
    if pd.notna(opp_seed):
        return 6.0 if opp_seed <= 4 else 4.0
    return 2.0 if opp_in_secondary else 0.25


def compute_opp_quality_pts_won(
    compact: pd.DataFrame,
    seeds: pd.DataFrame,
    secondary_teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (Season, TeamID), compute the total opponent-quality-points
    earned from WINS during the regular season.

    Parameters
    ----------
    compact : regular season results (WTeamID, LTeamID, Season)
    seeds : tournament seeds for THAT season (to classify opponents)
    secondary_teams : MSecondaryTourneyTeams or WSecondaryTourneyTeams
    """
    # Build lookups per season
    seed_lookup = {
        (s, t): n for s, t, n in seeds[["Season", "TeamID", "SeedNum"]].values
    }
    secondary_set = set(zip(secondary_teams["Season"], secondary_teams["TeamID"]))

    rows = []
    for (season, wtm), grp in compact.groupby(["Season", "WTeamID"]):
        pts = 0.0
        for l in grp["LTeamID"]:
            opp_seed = seed_lookup.get((season, l))
            opp_secondary = (season, l) in secondary_set
            pts += _assign_opp_quality_points(opp_seed, opp_secondary)
        rows.append({"Season": season, "TeamID": wtm, "OppQltyPtsWon": pts})
    return pd.DataFrame(rows)


def build_harry_rating(
    team_season: pd.DataFrame,
    opp_qlty_pts: pd.DataFrame,
    team_conferences: pd.DataFrame,
    is_men: bool = True,
) -> pd.DataFrame:
    """
    Compute harry_Rating per (Season, TeamID).

    Requires:
        team_season : has Season, TeamID, NetEff
        opp_qlty_pts : (Season, TeamID, OppQltyPtsWon)
        team_conferences : (Season, TeamID, ConfAbbrev)

    Returns original team_season plus columns:
        OppQltyPtsWon, PowerConf, HarryRating
    """
    df = team_season.merge(opp_qlty_pts, on=["Season", "TeamID"], how="left")
    df["OppQltyPtsWon"] = df["OppQltyPtsWon"].fillna(0.25)

    # Power conference flag (0/1 -> scale to [1, 1.3] or [1, 1.1])
    tc = team_conferences[["Season", "TeamID", "ConfAbbrev"]].copy()
    tc["PowerConf"] = tc["ConfAbbrev"].str.lower().isin(POWER_CONFS).astype(int)
    df = df.merge(tc, on=["Season", "TeamID"], how="left")
    df["PowerConf"] = df["PowerConf"].fillna(0).astype(int)

    # MinMax scale per gender (full dataset, not per-season — Harry's writeup)
    opp_range    = (-0.55, 0.55) if is_men else (-0.5, 0.5)
    power_range  = (1.0, 1.3)    if is_men else (1.0, 1.1)

    scaler_opp   = MinMaxScaler(feature_range=opp_range)
    scaler_power = MinMaxScaler(feature_range=power_range)
    df["OppQltyPtsWon_MinMax"] = scaler_opp.fit_transform(df[["OppQltyPtsWon"]])
    df["PowerConf_MinMax"]     = scaler_power.fit_transform(df[["PowerConf"]])
    # top12 skipped (no AP Poll data); leave at 1.0 neutral
    df["Top12_MinMax"] = 1.0

    df["HarryRating"] = (
        df["NetEff"]
        * (1 + df["OppQltyPtsWon_MinMax"])
        * df["PowerConf_MinMax"]
        * df["Top12_MinMax"]
    )
    return df
