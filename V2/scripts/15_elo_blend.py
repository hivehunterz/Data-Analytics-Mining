"""
Stage 15: Blend LR predictions with Elo-derived logistic probabilities.

Elo's built-in win probability:  p_elo = 1 / (1 + 10^((Elo_B - Elo_A) / 400))

This is an independent probability calculation that doesn't use any of
the other features. Blending it with the LR prediction can help if Elo
captures signal the LR misses.

Sweep blend weight alpha: p_final = alpha * p_lr + (1-alpha) * p_elo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from config import (PROCESSED, CV_REPORTS, SUBMISSIONS, TRUTH_DIR,
                     MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN, MEN_ID_CUTOFF,
                     CLIP_LO, CLIP_HI)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.postprocess.clip import clip_predictions
from src.utils import log, brier


def _elo_prob(elo_a: np.ndarray, elo_b: np.ndarray) -> np.ndarray:
    """Probability team A beats team B per 538's Elo formula."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def _load_pruned_features(gender: str) -> list:
    with open(CV_REPORTS / f"{gender}_full_pruned.json") as f:
        payload = json.load(f)
    feats = [c.replace("diff_", "") for c in payload["final_features"]]
    return [f for f in feats if f != "SeedNum"]


def _run_gender(label, feat, seeds, tourney_loader, elo_df, seasons, C, features):
    tourney = tourney_loader()
    tourney = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    s = matchups["Season"].values

    # LR OOF
    res = loso_cv_brier(make_lr_pipeline(C=C), X, y, s, return_oof=True)
    lr_oof = res["oof_preds"]

    # Elo OOF: simple formula using pre-game Elos
    elo_map = dict(zip(zip(elo_df["Season"], elo_df["TeamID"]), elo_df["Elo"]))
    m = matchups.copy()
    m["elo_T1"] = m.apply(lambda r: elo_map.get((r["Season"], r["T1"]), 1500), axis=1)
    m["elo_T2"] = m.apply(lambda r: elo_map.get((r["Season"], r["T2"]), 1500), axis=1)
    elo_oof = _elo_prob(m["elo_T1"].values, m["elo_T2"].values)

    log(f"{label}: LR OOF Brier = {brier(y, lr_oof):.5f}")
    log(f"{label}: Elo OOF Brier = {brier(y, elo_oof):.5f}")

    log(f"\n{label}: blend sweep:")
    best_a, best_b = 1.0, brier(y, np.clip(lr_oof, CLIP_LO, CLIP_HI))
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        blended = alpha * lr_oof + (1 - alpha) * elo_oof
        blended = np.clip(blended, CLIP_LO, CLIP_HI)
        b = brier(y, blended)
        log(f"  alpha={alpha:.2f}: Brier={b:.5f}")
        if b < best_b:
            best_a, best_b = alpha, b
    log(f"{label} best: alpha={best_a}, Brier={best_b:.5f}")
    return best_a, best_b


def main():
    log("=== MEN ===")
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()
    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    m_features = _load_pruned_features("men")
    m_alpha, m_brier = _run_gender("Men", m_feat, m_seeds, kl.load_m_tourney_compact,
                                     m_elo, MEN_SEASONS_FULL, LR_C_MEN, m_features)

    log("\n=== WOMEN ===")
    w_team = pd.read_parquet(PROCESSED / "w_team_season.parquet")
    w_elo  = pd.read_parquet(PROCESSED / "w_elo.parquet")
    w_cust = pd.read_parquet(PROCESSED / "w_ratings_custom.parquet")
    w_seeds = kl.load_w_seeds()
    w_confs = kl.load_w_conferences()
    w_secondary = kl._read("WSecondaryTourneyTeams.csv")
    w_reg = kl.load_w_regular_compact()
    w_opq = compute_opp_quality_pts_won(w_reg, w_seeds, w_secondary)
    w_team_hr = build_harry_rating(w_team, w_opq, w_confs, is_men=False)
    w_inter = add_interactions_women(w_team_hr, w_cust, w_elo, w_seeds)
    w_feat = w_inter.merge(w_elo[["Season", "TeamID", "EloSlope"]],
                           on=["Season", "TeamID"], how="left")
    w_features = _load_pruned_features("women")
    w_alpha, w_brier = _run_gender("Women", w_feat, w_seeds, kl.load_w_tourney_compact,
                                     w_elo, WOMEN_SEASONS_FULL, LR_C_WOMEN, w_features)

    with open(CV_REPORTS / "elo_blend.json", "w") as f:
        json.dump({"men_alpha": m_alpha, "men_brier": m_brier,
                   "women_alpha": w_alpha, "women_brier": w_brier},
                  f, indent=2)


if __name__ == "__main__":
    main()
