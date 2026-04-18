"""
Build per-team-season statistics from detailed regular-season results.

Each game contributes one row where the focal team is the winner, plus one
row where the focal team is the loser. Aggregated per (Season, TeamID).

Outputs include Four Factors (Dean Oliver), per-100-possession efficiency,
per-game box score averages, tempo, win percentage, and point differential.
"""
import numpy as np
import pandas as pd


POSS_FTA_COEF = 0.475   # Dean Oliver's possession estimator coefficient


def _team_game_rows(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each game into two team-perspective rows (winner + loser),
    renaming W* -> own_, L* -> opp_ (and vice-versa).

    NOTE: WLoc is NOT a box-score stat; it's the winner's court location.
    Handle it specially so we don't accidentally create `own_Loc`.
    """
    # All W-prefix columns EXCEPT WLoc are box-score fields to swap
    w_box = [c for c in detailed.columns if c.startswith("W") and c != "WLoc"]
    l_box = [c for c in detailed.columns if c.startswith("L")]

    # Winner-as-focal
    w = detailed.rename(columns={
        **{c: "own_" + c[1:] for c in w_box},
        **{c: "opp_" + c[1:] for c in l_box},
    }).copy()
    # Focal court location: winner's perspective = WLoc as-is
    w["focal_loc"] = w["WLoc"]
    w["TeamID"] = w["own_TeamID"]
    w["OppID"]  = w["opp_TeamID"]
    w["Win"]    = 1

    # Loser-as-focal
    l = detailed.rename(columns={
        **{c: "own_" + c[1:] for c in l_box},
        **{c: "opp_" + c[1:] for c in w_box},
    }).copy()
    l["focal_loc"] = l["WLoc"].map({"H": "A", "A": "H", "N": "N"})
    l["TeamID"] = l["own_TeamID"]
    l["OppID"]  = l["opp_TeamID"]
    l["Win"]    = 0

    keep = ["Season", "DayNum", "TeamID", "OppID", "focal_loc", "Win",
            "NumOT"] + [c for c in w.columns if c.startswith(("own_", "opp_"))]
    keep = [c for c in keep if c in w.columns]
    return pd.concat([w[keep], l[keep]], ignore_index=True)


def _add_game_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Add possession counts and per-game efficiency to each team-game row."""
    own_poss = df["own_FGA"] - df["own_OR"] + df["own_TO"] + POSS_FTA_COEF * df["own_FTA"]
    opp_poss = df["opp_FGA"] - df["opp_OR"] + df["opp_TO"] + POSS_FTA_COEF * df["opp_FTA"]
    poss = (own_poss + opp_poss) / 2   # same for both teams in a game
    df["Poss"]    = poss.clip(lower=1)
    df["OffEff"]  = df["own_Score"] / df["Poss"] * 100
    df["DefEff"]  = df["opp_Score"] / df["Poss"] * 100
    df["NetEff"]  = df["OffEff"] - df["DefEff"]

    # Four Factors
    df["EFGPct"]    = (df["own_FGM"] + 0.5 * df["own_FGM3"]) / df["own_FGA"].clip(lower=1)
    df["TORate"]    = df["own_TO"] / df["Poss"]
    df["ORPct"]     = df["own_OR"] / (df["own_OR"] + df["opp_DR"]).clip(lower=1)
    df["FTRate"]    = df["own_FTA"] / df["own_FGA"].clip(lower=1)
    # Opponent Four Factors (i.e. defense)
    df["OppEFGPct"] = (df["opp_FGM"] + 0.5 * df["opp_FGM3"]) / df["opp_FGA"].clip(lower=1)
    df["OppTORate"] = df["opp_TO"] / df["Poss"]
    df["DRPct"]     = df["own_DR"] / (df["own_DR"] + df["opp_OR"]).clip(lower=1)
    df["OppFTRate"] = df["opp_FTA"] / df["opp_FGA"].clip(lower=1)

    # Shooting splits
    df["FGPct"]    = df["own_FGM"] / df["own_FGA"].clip(lower=1)
    df["FG3Pct"]   = df["own_FGM3"] / df["own_FGA3"].clip(lower=1)
    df["FTPct"]    = df["own_FTM"] / df["own_FTA"].clip(lower=1)
    df["OppFGPct"] = df["opp_FGM"] / df["opp_FGA"].clip(lower=1)
    return df


def build_team_season(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detailed regular-season results into one row per (Season, TeamID).
    """
    tg = _team_game_rows(detailed)
    tg = _add_game_efficiency(tg)

    agg_mean = [
        "OffEff", "DefEff", "NetEff", "Poss",
        "EFGPct", "TORate", "ORPct", "FTRate",
        "OppEFGPct", "OppTORate", "DRPct", "OppFTRate",
        "FGPct", "FG3Pct", "FTPct", "OppFGPct",
    ]
    agg_sum_count = ["Win"]
    agg_pergame = [
        "own_Ast", "own_TO", "own_Stl", "own_Blk", "own_OR", "own_DR",
        "own_Score", "opp_Score",
    ]

    grp = tg.groupby(["Season", "TeamID"])

    out = grp[agg_mean].mean()
    out["Tempo"]      = grp["Poss"].mean()
    out["GamesPlayed"]= grp.size()
    out["Wins"]       = grp["Win"].sum()
    out["WinPct"]     = out["Wins"] / out["GamesPlayed"]
    out["AvgPts"]     = grp["own_Score"].mean()
    out["AvgOppPts"]  = grp["opp_Score"].mean()
    out["PointDiff"]  = out["AvgPts"] - out["AvgOppPts"]
    out["AstPerGame"] = grp["own_Ast"].mean()
    out["TOPerGame"]  = grp["own_TO"].mean()
    out["StlPerGame"] = grp["own_Stl"].mean()
    out["BlkPerGame"] = grp["own_Blk"].mean()
    out["ORPerGame"]  = grp["own_OR"].mean()
    out["DRPerGame"]  = grp["own_DR"].mean()

    return out.reset_index()


def build_from_compact(compact: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback feature builder when detailed box scores aren't available
    (pre-2003 men, pre-2010 women). Produces WinPct, PointDiff, AvgPts,
    AvgOppPts only.
    """
    rows = []
    for role, w_col_team, w_col_score, l_col_team, l_col_score, win_flag in [
        ("W", "WTeamID", "WScore", "LTeamID", "LScore", 1),
        ("L", "LTeamID", "LScore", "WTeamID", "WScore", 0),
    ]:
        r = compact[["Season", w_col_team, w_col_score, l_col_score]].copy()
        r.columns = ["Season", "TeamID", "own_Score", "opp_Score"]
        r["Win"] = win_flag
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)
    grp = tg.groupby(["Season", "TeamID"])
    out = pd.DataFrame({
        "GamesPlayed": grp.size(),
        "Wins": grp["Win"].sum(),
        "AvgPts": grp["own_Score"].mean(),
        "AvgOppPts": grp["opp_Score"].mean(),
    }).reset_index()
    out["WinPct"]    = out["Wins"] / out["GamesPlayed"]
    out["PointDiff"] = out["AvgPts"] - out["AvgOppPts"]
    return out
