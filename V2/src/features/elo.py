"""
MOV-weighted Elo ratings with season carry-over.

Design blends Brendan's MOV formula with Kevin's 0.75 season mean reversion:

    mov_mult = log(max(mov,1)+1) * 2.2 / ((w_elo - l_elo) * 0.001 + 2.2)
    season_start_elo = 0.75 * last_season_end_elo + 0.25 * 1500

Home court adds 100 Elo points to the home side when computing expected
win probability. Base K = 20, multiplied by mov_mult per game.
"""
import numpy as np
import pandas as pd

from config import ELO_INITIAL, ELO_K, ELO_HOME_ADV, ELO_MEAN_REVERSION


def compute_elo_ratings(
    compact: pd.DataFrame,
    k: int = ELO_K,
    home_adv: int = ELO_HOME_ADV,
    mean_reversion: float = ELO_MEAN_REVERSION,
    initial: float = ELO_INITIAL,
) -> pd.DataFrame:
    """
    Run Elo through every regular-season game in chronological order and
    emit END-OF-SEASON rating per (Season, TeamID).

    Parameters
    ----------
    compact : pd.DataFrame
        Must have Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc.

    Returns
    -------
    pd.DataFrame with columns [Season, TeamID, Elo]
    """
    required = {"Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc"}
    missing = required - set(compact.columns)
    if missing:
        raise ValueError(f"compact missing columns: {missing}")

    elo: dict[int, float] = {}      # TeamID -> current Elo
    last_season_elo: dict[int, float] = {}
    snapshots: list[dict] = []

    for season, games in compact.sort_values(["Season", "DayNum"]).groupby("Season"):
        # Carry-over: blend last-season end Elo with initial
        for team in list(elo.keys()):
            elo[team] = mean_reversion * elo[team] + (1 - mean_reversion) * initial
        # Teams new to this season default to initial
        for team in set(games["WTeamID"]).union(games["LTeamID"]):
            if team not in elo:
                elo[team] = initial

        for _, g in games.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            w_elo, l_elo = elo[w], elo[l]
            # Home court advantage (only applies when venue is home for one)
            if g["WLoc"] == "H":
                w_eff, l_eff = w_elo + home_adv, l_elo
            elif g["WLoc"] == "A":
                w_eff, l_eff = w_elo, l_elo + home_adv
            else:
                w_eff, l_eff = w_elo, l_elo

            # Expected win prob for the actual winner
            w_exp = 1.0 / (1.0 + 10 ** ((l_eff - w_eff) / 400))

            # Margin-of-victory multiplier (Brendan / 538-style)
            mov = abs(g["WScore"] - g["LScore"])
            # Use actual (unadjusted) elo diff for the autocorrelation term
            elo_diff = w_elo - l_elo
            mov_mult = np.log(max(mov, 1) + 1) * 2.2 / (elo_diff * 0.001 + 2.2)

            delta = k * mov_mult * (1 - w_exp)
            elo[w] += delta
            elo[l] -= delta

        # Snapshot end-of-season before applying carry-over next iteration
        for team, rating in elo.items():
            snapshots.append({"Season": season, "TeamID": team, "Elo": rating})
        last_season_elo = elo.copy()

    return pd.DataFrame(snapshots)


def compute_elo_slope(
    compact: pd.DataFrame,
    k: int = ELO_K,
    home_adv: int = ELO_HOME_ADV,
    mean_reversion: float = ELO_MEAN_REVERSION,
    initial: float = ELO_INITIAL,
) -> pd.DataFrame:
    """
    In-season Elo slope per (Season, TeamID): rate of change per 10 game days.
    Approximate via a linear regression of Elo-over-time during the season.

    Useful as a separate "elo_slope" feature (Kevin's women's feature).
    """
    elo: dict[int, float] = {}
    tracker: list[dict] = []

    for season, games in compact.sort_values(["Season", "DayNum"]).groupby("Season"):
        for team in list(elo.keys()):
            elo[team] = mean_reversion * elo[team] + (1 - mean_reversion) * initial
        for team in set(games["WTeamID"]).union(games["LTeamID"]):
            if team not in elo:
                elo[team] = initial

        for _, g in games.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            w_elo, l_elo = elo[w], elo[l]
            if g["WLoc"] == "H":
                w_eff, l_eff = w_elo + home_adv, l_elo
            elif g["WLoc"] == "A":
                w_eff, l_eff = w_elo, l_elo + home_adv
            else:
                w_eff, l_eff = w_elo, l_elo
            w_exp = 1.0 / (1.0 + 10 ** ((l_eff - w_eff) / 400))
            mov = abs(g["WScore"] - g["LScore"])
            elo_diff = w_elo - l_elo
            mov_mult = np.log(max(mov, 1) + 1) * 2.2 / (elo_diff * 0.001 + 2.2)
            delta = k * mov_mult * (1 - w_exp)
            elo[w] += delta
            elo[l] -= delta
            tracker.append({
                "Season": season, "DayNum": g["DayNum"],
                "TeamID": w, "Elo": elo[w],
            })
            tracker.append({
                "Season": season, "DayNum": g["DayNum"],
                "TeamID": l, "Elo": elo[l],
            })

    df = pd.DataFrame(tracker)
    # For each (Season, TeamID), fit slope = dElo/dDay
    out = []
    for (s, t), grp in df.groupby(["Season", "TeamID"]):
        if len(grp) < 5:
            slope = 0.0
        else:
            x = grp["DayNum"].values
            y = grp["Elo"].values
            slope = float(np.polyfit(x, y, 1)[0])
        out.append({"Season": s, "TeamID": t, "EloSlope": slope})
    return pd.DataFrame(out)
