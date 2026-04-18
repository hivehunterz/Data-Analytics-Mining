"""
Stage 19: Add Barttorvik features to men's model, retune C, backward-elim.

Barttorvik coverage limits us to 2008+ for training if we use them.
Kevin restricted to 2015+ but earlier years might work. We'll try both.

Expected gain: ~0.01-0.02 Brier per Kevin's feature importance (adjoe,
adjde, WAB, barthag were in his top features).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import numpy as np

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, MEN_SEASONS_BARTT,
                     LR_C_MEN)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.validation.backward_elim import backward_elimination
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
BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


def _assemble_men_with_bartt():
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_bart = pd.read_parquet(PROCESSED / "m_barttorvik.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()

    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    # Merge Barttorvik on (Season, TeamID)
    m_feat = m_feat.merge(
        m_bart[["Season", "TeamID"] + BARTT_COLS],
        on=["Season", "TeamID"], how="left",
    )
    return m_feat, m_seeds


def _run(label: str, seasons: list, features: list, m_feat, m_seeds, tourney):
    tourney_in = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney_in, m_feat, features, seeds_df=m_seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    s = matchups["Season"].values
    log(f"\n{label}: rows={len(matchups)}, features={len(diff_cols)}")

    # Raw CV
    res = loso_cv_brier(make_lr_pipeline(C=LR_C_MEN), X, y, s, return_oof=False)
    log(f"  LR (C={LR_C_MEN}) Brier: {res['mean_brier']:.5f}")

    # C sweep
    best_c, best_b = LR_C_MEN, res['mean_brier']
    for c in [0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        r = loso_cv_brier(make_lr_pipeline(C=c), X, y, s, return_oof=False)
        if r['mean_brier'] < best_b:
            best_c, best_b = c, r['mean_brier']
    log(f"  best C: {best_c}, Brier={best_b:.5f}")

    # Backward elim
    log("  running backward elimination...")
    X_df = matchups[diff_cols]
    be = backward_elimination(
        lambda: make_lr_pipeline(C=best_c),
        X_df, y, s, threshold=0.0003, max_iters=15, verbose=False,
    )
    log(f"  after pruning ({len(be['final_features'])} features): "
        f"Brier={be['final_brier']:.5f}")
    return best_c, be


def main():
    m_feat, m_seeds = _assemble_men_with_bartt()
    log(f"men feat shape: {m_feat.shape}")
    log(f"Barttorvik non-null rate: "
        f"{m_feat['adjoe'].notna().mean():.1%}")

    tourney = kl.load_m_tourney_compact()

    all_features = (
        BRENDAN_FEATURES +
        ["Elo", "EloSlope"] +
        ["ColleyRating", "SRS", "GLMQuality"] +
        ["MasseyMean", "MasseyMedian", "MasseyMin"] +
        ["OppQltyPtsWon", "PowerConf", "HarryRating"] +
        ["Seed_x_MasseyMean", "SRS_x_WinPct", "Colley_x_PointDiff"] +
        BARTT_COLS
    )

    # Option A: full window 2003-2025, Barttorvik features median-imputed pre-2008
    best_c_A, be_A = _run(
        "FULL 2003-2025 (impute missing Barttorvik)",
        MEN_SEASONS_FULL, all_features, m_feat, m_seeds, tourney,
    )

    # Option B: restricted to 2015+
    best_c_B, be_B = _run(
        "RESTRICTED 2015-2025 (Barttorvik always available)",
        MEN_SEASONS_BARTT, all_features, m_feat, m_seeds, tourney,
    )

    summary = {
        "A_full": {"best_c": best_c_A, "final_brier": be_A["final_brier"],
                   "features": be_A["final_features"]},
        "B_restricted": {"best_c": best_c_B, "final_brier": be_B["final_brier"],
                          "features": be_B["final_features"]},
    }
    with open(CV_REPORTS / "men_barttorvik.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSUMMARY:")
    print(f"  Option A (2003-2025, impute):  C={best_c_A}  Brier={be_A['final_brier']:.5f}  "
          f"({len(be_A['final_features'])} feats)")
    print(f"  Option B (2015-2025, complete): C={best_c_B}  Brier={be_B['final_brier']:.5f}  "
          f"({len(be_B['final_features'])} feats)")


if __name__ == "__main__":
    main()
