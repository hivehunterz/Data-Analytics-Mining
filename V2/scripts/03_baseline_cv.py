"""
Stage 3: Baseline LOSO CV for men's and women's models.

Two configurations:
  A. Seed-only: diff_SeedNum alone
  B. Brendan-baseline: SeedNum + efficiency + Elo + Four Factors +
     shooting splits + recent form

Reports Brier, log-loss, per-season breakdown. Saves to outputs/cv_reports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, save_cv_report


BRENDAN_FEATURES = [
    # Efficiency
    "NetEff", "OffEff", "DefEff",
    # Four Factors (offensive)
    "EFGPct", "TORate", "ORPct", "FTRate",
    # Opponent Four Factors (defensive)
    "OppEFGPct", "OppTORate", "DRPct", "OppFTRate",
    # Shooting splits
    "FGPct", "FG3Pct", "FTPct", "OppFGPct",
    # Per-game
    "AstPerGame", "TOPerGame", "StlPerGame", "BlkPerGame",
    "ORPerGame", "DRPerGame", "Tempo",
    # Overall
    "WinPct", "PointDiff", "AvgPts", "AvgOppPts",
]


def _prepare(
    team_season: pd.DataFrame,
    elo: pd.DataFrame,
    tourney: pd.DataFrame,
    seeds: pd.DataFrame,
    seasons: list,
) -> tuple[pd.DataFrame, list]:
    """
    Merge team-season + Elo, filter to target seasons, build tourney matchups.
    Returns the matchup DataFrame and the list of feature columns (pre-diff).
    """
    feat = team_season.merge(elo, on=["Season", "TeamID"], how="left")

    features = BRENDAN_FEATURES + ["Elo", "EloSlope"]

    # Only train on seasons where tourney data is available
    tourney_in = tourney[tourney["Season"].isin(seasons)].copy()
    matchups = build_tourney_matchups(
        tourney_compact=tourney_in,
        feat_df=feat,
        feature_cols=features,
        seeds_df=seeds,
    )
    return matchups, features


def _cv_config(name: str, matchups: pd.DataFrame, diff_cols: list, C: float):
    X = matchups[diff_cols].values
    y = matchups["label"].values
    seasons = matchups["Season"].values
    model = make_lr_pipeline(C=C)
    res = loso_cv_brier(model, X, y, seasons)
    log(f"  [{name}] Brier={res['mean_brier']:.5f} "
        f"±{res['std_brier']:.5f}  logloss={res['mean_logloss']:.5f}")
    return res


def _run_gender(label: str, team_path: Path, elo_path: Path,
                tourney_loader, seeds_loader, seasons: list, C: float) -> dict:
    log(f"\n=== {label} ===")
    log("loading processed features...")
    team_season = pd.read_parquet(team_path)
    elo         = pd.read_parquet(elo_path)
    tourney     = tourney_loader()
    seeds       = seeds_loader()

    matchups, features = _prepare(team_season, elo, tourney, seeds, seasons)
    log(f"  training rows: {len(matchups)} across seasons "
        f"{matchups['Season'].min()}-{matchups['Season'].max()}")

    # === A. Seed-only ===
    res_seed = _cv_config("seed_only", matchups, ["diff_SeedNum"], C)

    # === B. Brendan baseline ===
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    res_brendan = _cv_config("brendan_base", matchups, diff_cols, C)

    # Save reports
    save_cv_report(f"{label.lower()}_seed_only", res_seed, CV_REPORTS)
    save_cv_report(f"{label.lower()}_brendan_base", res_brendan, CV_REPORTS)

    return {"seed_only": res_seed, "brendan_base": res_brendan,
            "features": features}


def main():
    # Men
    _run_gender(
        label="Men",
        team_path=PROCESSED / "m_team_season.parquet",
        elo_path=PROCESSED / "m_elo.parquet",
        tourney_loader=kl.load_m_tourney_compact,
        seeds_loader=kl.load_m_seeds,
        seasons=MEN_SEASONS_FULL,
        C=LR_C_MEN,
    )

    # Women
    _run_gender(
        label="Women",
        team_path=PROCESSED / "w_team_season.parquet",
        elo_path=PROCESSED / "w_elo.parquet",
        tourney_loader=kl.load_w_tourney_compact,
        seeds_loader=kl.load_w_seeds,
        seasons=WOMEN_SEASONS_FULL,
        C=LR_C_WOMEN,
    )


if __name__ == "__main__":
    main()
