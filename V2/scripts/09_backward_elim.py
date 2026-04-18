"""
Stage 9: Backward-elimination feature pruning on men's and women's LR.

This was Kevin's biggest single-round CV gain (-0.007 Brier).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import numpy as np

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.models.train_lr import make_lr_pipeline
from src.validation.backward_elim import backward_elimination
from src.utils import log


BRENDAN_FEATURES = [
    "NetEff", "OffEff", "DefEff",
    "EFGPct", "TORate", "ORPct", "FTRate",
    "OppEFGPct", "OppTORate", "DRPct", "OppFTRate",
    "FGPct", "FG3Pct", "FTPct", "OppFGPct",
    "AstPerGame", "TOPerGame", "StlPerGame", "BlkPerGame",
    "ORPerGame", "DRPerGame", "Tempo",
    "WinPct", "PointDiff", "AvgPts", "AvgOppPts",
]


def _prepare_matchups(team_df, elo_df, custom_df, massey_df,
                      tourney_df, seeds_df, seasons, extra_features):
    feat = team_df.merge(elo_df, on=["Season", "TeamID"], how="left")
    feat = feat.merge(custom_df, on=["Season", "TeamID"], how="left")
    if massey_df is not None:
        feat = feat.merge(massey_df, on=["Season", "TeamID"], how="left")
    features = (BRENDAN_FEATURES + ["Elo", "EloSlope"] +
                ["ColleyRating", "SRS", "GLMQuality"] + extra_features)

    tourney = tourney_df[tourney_df["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds_df)
    return matchups, features


def _run(label: str, matchups: pd.DataFrame, features: list, C: float):
    log(f"\n=== {label} backward elimination ===")
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X_df = matchups[diff_cols]
    y = matchups["label"].values
    seasons_arr = matchups["Season"].values

    def model_factory():
        return make_lr_pipeline(C=C)

    result = backward_elimination(
        model_factory, X_df, y, seasons_arr,
        threshold=0.0005, max_iters=25, verbose=True,
    )

    log(f"\n{label} final: {len(result['final_features'])} features, "
        f"Brier {result['final_brier']:.5f}")
    log(f"kept: {result['final_features']}")

    with open(CV_REPORTS / f"{label.lower()}_backward_elim.json", "w") as f:
        json.dump({
            "final_features": result["final_features"],
            "final_brier": result["final_brier"],
            "history": result["history"],
        }, f, indent=2, default=str)


def main():
    # Men
    m_matchups, m_feats = _prepare_matchups(
        pd.read_parquet(PROCESSED / "m_team_season.parquet"),
        pd.read_parquet(PROCESSED / "m_elo.parquet"),
        pd.read_parquet(PROCESSED / "m_ratings_custom.parquet"),
        pd.read_parquet(PROCESSED / "m_massey_agg.parquet"),
        kl.load_m_tourney_compact(),
        kl.load_m_seeds(),
        MEN_SEASONS_FULL,
        extra_features=["MasseyMean", "MasseyMedian", "MasseyMin"],
    )
    _run("Men", m_matchups, m_feats, LR_C_MEN)

    # Women
    w_matchups, w_feats = _prepare_matchups(
        pd.read_parquet(PROCESSED / "w_team_season.parquet"),
        pd.read_parquet(PROCESSED / "w_elo.parquet"),
        pd.read_parquet(PROCESSED / "w_ratings_custom.parquet"),
        None,  # no Massey for women
        kl.load_w_tourney_compact(),
        kl.load_w_seeds(),
        WOMEN_SEASONS_FULL,
        extra_features=[],
    )
    _run("Women", w_matchups, w_feats, LR_C_WOMEN)


if __name__ == "__main__":
    main()
