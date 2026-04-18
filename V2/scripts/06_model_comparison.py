"""
Stage 6: Compare THREE model families on the men's + Massey feature set:
  1. Logistic Regression (Kevin's primary)
  2. XGBoost shallow (Harry's primary)
  3. LR + XGBoost 40/60 blend (Brendan's primary)

Goal: empirically verify whether LR beats XGB on our data (Kevin's finding)
or whether the blend helps (Brendan's finding). This settles the debate
about whether we need ensembles or just LR.

Output: comparison table across genders + per-season brier breakdown.
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
from src.models.train_xgb import make_xgb_pipeline
from src.models.blend import LRxXGBBlend
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
MASSEY_FEATURES = ["MasseyMean", "MasseyMedian", "MasseyMin"]


def _load_men_features():
    t = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    e = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    feat = t.merge(e, on=["Season", "TeamID"], how="left")
    feat = feat.merge(m, on=["Season", "TeamID"], how="left")
    features = BRENDAN_FEATURES + ["Elo", "EloSlope"] + MASSEY_FEATURES
    return feat, features


def _load_women_features():
    t = pd.read_parquet(PROCESSED / "w_team_season.parquet")
    e = pd.read_parquet(PROCESSED / "w_elo.parquet")
    feat = t.merge(e, on=["Season", "TeamID"], how="left")
    features = BRENDAN_FEATURES + ["Elo", "EloSlope"]   # no Massey for women
    return feat, features


def _run(label: str, feat: pd.DataFrame, features: list, tourney: pd.DataFrame,
         seeds: pd.DataFrame, seasons: list, C: float) -> dict:
    log(f"\n=== {label} ===")
    tourney = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    seasons_arr = matchups["Season"].values
    log(f"  rows: {len(matchups)}  features: {len(diff_cols)}")

    results = {}

    # 1. Logistic Regression
    lr = make_lr_pipeline(C=C)
    r_lr = loso_cv_brier(lr, X, y, seasons_arr, return_oof=False)
    log(f"  [LR]         Brier={r_lr['mean_brier']:.5f}  "
        f"logloss={r_lr['mean_logloss']:.5f}")
    results["LR"] = r_lr

    # 2. XGBoost (Harry config, but faster)
    xgb = make_xgb_pipeline(max_depth=2, learning_rate=0.05, n_estimators=500)
    r_xgb = loso_cv_brier(xgb, X, y, seasons_arr, return_oof=False)
    log(f"  [XGB]        Brier={r_xgb['mean_brier']:.5f}  "
        f"logloss={r_xgb['mean_logloss']:.5f}")
    results["XGB"] = r_xgb

    # 3. LR + XGB blend (Brendan 40/60)
    blend = LRxXGBBlend(
        lr_pipeline=make_lr_pipeline(C=C),
        xgb_pipeline=make_xgb_pipeline(max_depth=2, learning_rate=0.05, n_estimators=500),
        w_lr=0.40, w_xgb=0.60,
        clip_lo=0.02, clip_hi=0.98,
    )
    r_blend = loso_cv_brier(blend, X, y, seasons_arr, return_oof=False)
    log(f"  [LR+XGB.40/60] Brier={r_blend['mean_brier']:.5f}  "
        f"logloss={r_blend['mean_logloss']:.5f}")
    results["Blend_40_60"] = r_blend

    # Save each
    for name, r in results.items():
        save_cv_report(f"{label.lower()}_{name.lower().replace('/','_')}", r, CV_REPORTS)

    return results


def main():
    log("=== MEN ===")
    mfeat, mfeatures = _load_men_features()
    _run("Men",
         mfeat, mfeatures,
         kl.load_m_tourney_compact(), kl.load_m_seeds(),
         MEN_SEASONS_FULL, LR_C_MEN)

    log("\n=== WOMEN ===")
    wfeat, wfeatures = _load_women_features()
    _run("Women",
         wfeat, wfeatures,
         kl.load_w_tourney_compact(), kl.load_w_seeds(),
         WOMEN_SEASONS_FULL, LR_C_WOMEN)


if __name__ == "__main__":
    main()
