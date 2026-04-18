"""
Stage 5: LOSO CV for men's model with Massey features added to baseline.

Compare to Stage 3 Brendan baseline: Men's 0.190 -> Massey should push
toward 0.185 (Kevin's finding: dynamic aggregation beat hand-picked by
0.00246, so we expect a meaningful drop).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, LR_C_MEN
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

MASSEY_FEATURES = ["MasseyMean", "MasseyMedian", "MasseyMin"]


def main():
    team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    tourney = kl.load_m_tourney_compact()
    seeds   = kl.load_m_seeds()

    feat = team.merge(elo, on=["Season", "TeamID"], how="left")
    feat = feat.merge(mass, on=["Season", "TeamID"], how="left")

    features = BRENDAN_FEATURES + ["Elo", "EloSlope"] + MASSEY_FEATURES

    tourney = tourney[tourney["Season"].isin(MEN_SEASONS_FULL)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    log(f"matchups: {len(matchups)}")

    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    seasons = matchups["Season"].values

    model = make_lr_pipeline(C=LR_C_MEN)
    res = loso_cv_brier(model, X, y, seasons)
    log(f"Men + Massey: Brier={res['mean_brier']:.5f} ±{res['std_brier']:.5f}  "
        f"logloss={res['mean_logloss']:.5f}")
    save_cv_report("men_with_massey", res, CV_REPORTS)

    # Per-season report
    print("\n  Per-season Brier:")
    for s, v in sorted(res["per_season"].items()):
        print(f"    {s}: {v['brier']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
