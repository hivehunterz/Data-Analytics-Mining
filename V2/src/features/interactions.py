"""
Feature interactions that capture non-linearity LR cannot otherwise model.

Kevin's list (men's only):
    SeedNum x pt_diff
    massey_rank x barthag
    close_win_pct x pt_diff
    SeedNum x massey
    ncsos x massey
    srs x ap_rank

We implement the ones that use features we already have computed. Skip
barthag/ap_rank (Barttorvik / external not yet wired in).
"""
import pandas as pd


def add_interactions_men(team_season: pd.DataFrame,
                          massey: pd.DataFrame,
                          custom: pd.DataFrame,
                          seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and add interaction columns. Returns updated team_season with new cols.

    New columns added:
        Seed_x_MasseyMean
        SRS_x_WinPct
        ColleyRating_x_PointDiff
    """
    df = team_season.merge(massey, on=["Season", "TeamID"], how="left")
    df = df.merge(custom, on=["Season", "TeamID"], how="left")
    # Seeds are per-season, per-team; most teams never get a seed so fill NaN
    seed_map = seeds[["Season", "TeamID", "SeedNum"]]
    df = df.merge(seed_map, on=["Season", "TeamID"], how="left")
    # Teams not in a given tournament get seed=17 (one beyond max seed)
    df["SeedNum"] = df["SeedNum"].fillna(17)

    df["Seed_x_MasseyMean"] = df["SeedNum"] * df["MasseyMean"]
    df["SRS_x_WinPct"]      = df["SRS"]     * df["WinPct"]
    df["Colley_x_PointDiff"]= df["ColleyRating"] * df["PointDiff"]
    return df


def add_interactions_women(team_season: pd.DataFrame,
                            custom: pd.DataFrame,
                            elo: pd.DataFrame,
                            seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Women's-only interactions. Kevin found these helpful:
        elo x colley
        last_n_pt_diff x elo x colley
        last_n_pt_diff x colley_rank

    We have Elo and Colley; skip last_n (not yet computed as a feature).
    """
    df = team_season.merge(custom, on=["Season", "TeamID"], how="left")
    df = df.merge(elo[["Season", "TeamID", "Elo"]], on=["Season", "TeamID"], how="left")
    seed_map = seeds[["Season", "TeamID", "SeedNum"]]
    df = df.merge(seed_map, on=["Season", "TeamID"], how="left")
    df["SeedNum"] = df["SeedNum"].fillna(17)

    df["Elo_x_Colley"]     = df["Elo"]     * df["ColleyRating"]
    df["SRS_x_Colley"]     = df["SRS"]     * df["ColleyRating"]
    df["PointDiff_x_Colley"]= df["PointDiff"] * df["ColleyRating"]
    return df
