"""
Stage 8: Add Colley + SRS + GLMQuality to the LR model and measure CV gain.

Expected (from plan): -0.003 both genders. This is the biggest gain for
women's since they have no Massey and these ratings fill that gap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, save_cv_report


BRENDAN_FEATURES = [
    "NetEff", "OffEff", "DefEff",
    "EFGPct", "TORate", "ORPct", "FTRate",
    "OppEFGPct", "OppTORate", "DRPct", "OppFTRate",
    "FGPct", "FG3Pct", "FTPct", "OppFGPct",
    "AstPerGame", "TOPerGame", "StlPerGame", "BlkPerGame",
    "ORPerGame", "DRPerGame", "Tempo",
    "WinPct", "PointDiff", "AvgPts", "AvgOppPts",
]
CUSTOM_RATINGS = ["ColleyRating", "SRS", "GLMQuality"]


def _run(label: str, team_path: str, elo_path: str, custom_path: str,
         tourney_loader, seeds_loader, seasons: list, C: float,
         extra_features: list):
    log(f"\n=== {label} ===")
    t = pd.read_parquet(PROCESSED / team_path)
    e = pd.read_parquet(PROCESSED / elo_path)
    cr = pd.read_parquet(PROCESSED / custom_path)
    feat = t.merge(e, on=["Season", "TeamID"], how="left")
    feat = feat.merge(cr, on=["Season", "TeamID"], how="left")
    if label == "Men":
        mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
        feat = feat.merge(mass, on=["Season", "TeamID"], how="left")

    features = BRENDAN_FEATURES + ["Elo", "EloSlope"] + CUSTOM_RATINGS + extra_features

    tourney = tourney_loader()
    tourney = tourney[tourney["Season"].isin(seasons)]
    seeds = seeds_loader()
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    seasons_arr = matchups["Season"].values
    log(f"  rows: {len(matchups)}  features: {len(diff_cols)}")

    lr = make_lr_pipeline(C=C)
    res = loso_cv_brier(lr, X, y, seasons_arr, return_oof=False)
    log(f"  [LR + custom ratings] Brier={res['mean_brier']:.5f}  "
        f"logloss={res['mean_logloss']:.5f}")
    save_cv_report(f"{label.lower()}_lr_custom", res, CV_REPORTS)
    return res


def main():
    _run(
        label="Men",
        team_path="m_team_season.parquet",
        elo_path="m_elo.parquet",
        custom_path="m_ratings_custom.parquet",
        tourney_loader=kl.load_m_tourney_compact,
        seeds_loader=kl.load_m_seeds,
        seasons=MEN_SEASONS_FULL,
        C=LR_C_MEN,
        extra_features=["MasseyMean", "MasseyMedian", "MasseyMin"],
    )
    _run(
        label="Women",
        team_path="w_team_season.parquet",
        elo_path="w_elo.parquet",
        custom_path="w_ratings_custom.parquet",
        tourney_loader=kl.load_w_tourney_compact,
        seeds_loader=kl.load_w_seeds,
        seasons=WOMEN_SEASONS_FULL,
        C=LR_C_WOMEN,
        extra_features=[],   # no Massey for women
    )


if __name__ == "__main__":
    main()
