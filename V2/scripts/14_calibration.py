"""
Stage 14: Try calibration methods on OOF predictions.

Methods:
  1. Temperature scaling: p' = sigmoid(logit(p) / T). T>1 dampens confidence.
  2. Isotonic regression: non-parametric monotonic map p -> p'.
  3. Platt scaling: LR on logit(p) -> y.

We check if any of these improve OOF Brier over the tuned LR.
The risk (from Kevin's writeup): isotonic fit on OOF may not match full-
model distribution, so we VERIFY by training on LOSO OOF preds and
evaluating on a separately-held split.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN, CLIP_LO, CLIP_HI)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, brier


def _load_pruned_features(gender: str) -> list:
    with open(CV_REPORTS / f"{gender}_full_pruned.json") as f:
        payload = json.load(f)
    feats = [c.replace("diff_", "") for c in payload["final_features"]]
    return [f for f in feats if f != "SeedNum"]


def _assemble_men():
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
    return m_feat, m_seeds


def _assemble_women():
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
    return w_feat, w_seeds


def _calibrate(oof: np.ndarray, y: np.ndarray) -> dict:
    """Try temperature, isotonic, Platt. Return best."""
    base = np.clip(oof, CLIP_LO, CLIP_HI)
    base_brier = brier(y, base)
    log(f"  baseline (clipped) Brier: {base_brier:.5f}")

    # Temperature scaling
    best_T, best_T_brier = 1.0, base_brier
    for T in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
        logit = np.log(base / (1 - base))
        p = 1 / (1 + np.exp(-logit / T))
        p = np.clip(p, CLIP_LO, CLIP_HI)
        b = brier(y, p)
        if b < best_T_brier:
            best_T, best_T_brier = T, b
    log(f"  temperature T={best_T:.2f}: Brier={best_T_brier:.5f}  "
        f"(delta={best_T_brier - base_brier:+.5f})")

    # Isotonic (OOF -> y)
    iso = IsotonicRegression(y_min=CLIP_LO, y_max=CLIP_HI, out_of_bounds="clip")
    iso.fit(base, y)
    p_iso = iso.predict(base)
    iso_brier = brier(y, p_iso)
    log(f"  isotonic (in-sample fit): Brier={iso_brier:.5f}  "
        f"(delta={iso_brier - base_brier:+.5f})   WARNING: in-sample; may not generalize")

    # Platt (LR on logit)
    logit = np.log(base / (1 - base)).reshape(-1, 1)
    platt = LogisticRegression().fit(logit, y)
    p_platt = platt.predict_proba(logit)[:, 1]
    p_platt = np.clip(p_platt, CLIP_LO, CLIP_HI)
    platt_brier = brier(y, p_platt)
    log(f"  Platt (in-sample fit):    Brier={platt_brier:.5f}  "
        f"(delta={platt_brier - base_brier:+.5f})")

    return {
        "base_brier": base_brier,
        "temperature": {"T": best_T, "brier": best_T_brier},
        "isotonic": {"brier": iso_brier},
        "platt": {"brier": platt_brier},
    }


def main():
    log("=== MEN ===")
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned_features("men")
    tourney = kl.load_m_tourney_compact()
    tourney = tourney[tourney["Season"].isin(MEN_SEASONS_FULL)]
    matchups = build_tourney_matchups(tourney, m_feat, m_features, seeds_df=m_seeds)
    diff_cols = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]
    X, y, s = matchups[diff_cols].values, matchups["label"].values, matchups["Season"].values

    res = loso_cv_brier(make_lr_pipeline(C=LR_C_MEN), X, y, s, return_oof=True)
    log(f"OOF LR Brier (uncalibrated): {res['mean_brier']:.5f}")
    men_cal = _calibrate(res["oof_preds"], y)

    log("\n=== WOMEN ===")
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned_features("women")
    tourney = kl.load_w_tourney_compact()
    tourney = tourney[tourney["Season"].isin(WOMEN_SEASONS_FULL)]
    matchups = build_tourney_matchups(tourney, w_feat, w_features, seeds_df=w_seeds)
    diff_cols = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]
    X, y, s = matchups[diff_cols].values, matchups["label"].values, matchups["Season"].values

    res = loso_cv_brier(make_lr_pipeline(C=LR_C_WOMEN), X, y, s, return_oof=True)
    log(f"OOF LR Brier (uncalibrated): {res['mean_brier']:.5f}")
    women_cal = _calibrate(res["oof_preds"], y)

    with open(CV_REPORTS / "calibration.json", "w") as f:
        json.dump({"men": men_cal, "women": women_cal}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
