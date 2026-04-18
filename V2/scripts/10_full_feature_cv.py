"""
Stage 10: Full-feature LOSO CV.

Adds: opp_qlty_pts_won, harry_Rating, interactions. Uses backward-elim kept
features as the LR input. This is the final CV before market blending and
submission generation.

Expected gains over Stage 9 (Men 0.18957 / Women 0.13853):
  - harry_rating + opp_qlty_pts_won: -0.003 men (Harry's feature r=0.59)
  - interactions: -0.002 (Kevin's non-linearity capture)
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
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import (compute_opp_quality_pts_won, build_harry_rating)
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


def main():
    # ── Men's ────────────────────────────────────────────────────────
    log("=== MEN: assembling all features ===")
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_seeds = kl.load_m_seeds()
    m_tourney = kl.load_m_tourney_compact()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg_compact = kl.load_m_regular_compact()

    log("men: computing opp_qlty_pts_won...")
    m_opq = compute_opp_quality_pts_won(m_reg_compact, m_seeds, m_secondary)

    log("men: building harry_Rating...")
    m_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_hr = m_hr.drop(columns=[c for c in m_hr.columns if c not in m_team.columns]
                     + ["HarryRating", "OppQltyPtsWon", "PowerConf",
                        "OppQltyPtsWon_MinMax", "PowerConf_MinMax", "Top12_MinMax"],
                     errors="ignore")
    # Re-do the merge so we keep the extra cols
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)

    log("men: adding interactions...")
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")

    m_features = (
        BRENDAN_FEATURES +
        ["Elo", "EloSlope"] +
        ["ColleyRating", "SRS", "GLMQuality"] +
        ["MasseyMean", "MasseyMedian", "MasseyMin"] +
        ["OppQltyPtsWon", "PowerConf", "HarryRating"] +
        ["Seed_x_MasseyMean", "SRS_x_WinPct", "Colley_x_PointDiff"]
    )
    log(f"  men features count: {len(m_features)}")

    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_FULL)]
    m_matchups = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    m_diff = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]

    X_m = m_matchups[m_diff]
    y_m = m_matchups["label"].values
    s_m = m_matchups["Season"].values

    log("men: baseline full-feature CV...")
    res_full_m = loso_cv_brier(make_lr_pipeline(C=LR_C_MEN),
                                X_m.values, y_m, s_m, return_oof=False)
    log(f"  Men full Brier: {res_full_m['mean_brier']:.5f}")
    save_cv_report("men_full_features", res_full_m, CV_REPORTS)

    log("men: backward elimination on full feature set...")
    be_m = backward_elimination(
        lambda: make_lr_pipeline(C=LR_C_MEN),
        X_m, y_m, s_m, threshold=0.0003, max_iters=15, verbose=True,
    )
    log(f"  Men pruned Brier: {be_m['final_brier']:.5f}  "
        f"({len(be_m['final_features'])} features kept)")
    with open(CV_REPORTS / "men_full_pruned.json", "w") as f:
        json.dump({
            "final_features": be_m["final_features"],
            "final_brier": be_m["final_brier"],
            "history": be_m["history"],
        }, f, indent=2, default=str)

    # ── Women's ──────────────────────────────────────────────────────
    log("\n=== WOMEN: assembling all features ===")
    w_team = pd.read_parquet(PROCESSED / "w_team_season.parquet")
    w_elo  = pd.read_parquet(PROCESSED / "w_elo.parquet")
    w_cust = pd.read_parquet(PROCESSED / "w_ratings_custom.parquet")
    w_seeds = kl.load_w_seeds()
    w_tourney = kl.load_w_tourney_compact()
    w_confs = kl.load_w_conferences()
    w_secondary = kl._read("WSecondaryTourneyTeams.csv")
    w_reg_compact = kl.load_w_regular_compact()

    log("women: computing opp_qlty_pts_won...")
    w_opq = compute_opp_quality_pts_won(w_reg_compact, w_seeds, w_secondary)

    log("women: building harry_Rating (is_men=False)...")
    w_team_hr = build_harry_rating(w_team, w_opq, w_confs, is_men=False)

    log("women: adding interactions...")
    w_inter = add_interactions_women(w_team_hr, w_cust, w_elo, w_seeds)
    w_feat = w_inter.merge(w_elo[["Season", "TeamID", "EloSlope"]],
                           on=["Season", "TeamID"], how="left")

    w_features = (
        BRENDAN_FEATURES +
        ["Elo", "EloSlope"] +
        ["ColleyRating", "SRS", "GLMQuality"] +
        ["OppQltyPtsWon", "PowerConf", "HarryRating"] +
        ["Elo_x_Colley", "SRS_x_Colley", "PointDiff_x_Colley"]
    )
    log(f"  women features count: {len(w_features)}")

    w_tour_in = w_tourney[w_tourney["Season"].isin(WOMEN_SEASONS_FULL)]
    w_matchups = build_tourney_matchups(w_tour_in, w_feat, w_features, seeds_df=w_seeds)
    w_diff = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]

    X_w = w_matchups[w_diff]
    y_w = w_matchups["label"].values
    s_w = w_matchups["Season"].values

    log("women: baseline full-feature CV...")
    res_full_w = loso_cv_brier(make_lr_pipeline(C=LR_C_WOMEN),
                                X_w.values, y_w, s_w, return_oof=False)
    log(f"  Women full Brier: {res_full_w['mean_brier']:.5f}")
    save_cv_report("women_full_features", res_full_w, CV_REPORTS)

    log("women: backward elimination on full feature set...")
    be_w = backward_elimination(
        lambda: make_lr_pipeline(C=LR_C_WOMEN),
        X_w, y_w, s_w, threshold=0.0003, max_iters=15, verbose=True,
    )
    log(f"  Women pruned Brier: {be_w['final_brier']:.5f}  "
        f"({len(be_w['final_features'])} features kept)")
    with open(CV_REPORTS / "women_full_pruned.json", "w") as f:
        json.dump({
            "final_features": be_w["final_features"],
            "final_brier": be_w["final_brier"],
            "history": be_w["history"],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
