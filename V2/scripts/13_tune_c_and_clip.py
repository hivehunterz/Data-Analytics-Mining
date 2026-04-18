"""
Stage 13: Hyperparameter tuning — LR C and clip range.

Our CV Brier (0.1875 men / 0.1373 women) looked good but we overconfidently
predicted Duke 87% vs UConn, Wisconsin 85% vs High Point, etc. on the real
2026 bracket (scored 0.1635 men's).

Tighter clip + different C might calibrate confidence better.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from config import (PROCESSED, CV_REPORTS, TRUTH_DIR, SUBMISSIONS,
                     MEN_SEASONS_FULL, WOMEN_SEASONS_FULL, MEN_ID_CUTOFF)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, brier


def _load_pruned_features(gender: str) -> list:
    path = CV_REPORTS / f"{gender}_full_pruned.json"
    with open(path) as f:
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


def _tune_gender(label: str, feat, seeds, tourney_loader, seasons, features):
    tourney = tourney_loader()
    tourney = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    s = matchups["Season"].values

    # C sweep
    log(f"\n=== {label}: C parameter sweep ===")
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_c, best_brier = None, 1e9
    for c in c_values:
        res = loso_cv_brier(make_lr_pipeline(C=c), X, y, s, return_oof=False)
        log(f"  C={c:>6.2f}: Brier={res['mean_brier']:.5f}")
        if res['mean_brier'] < best_brier:
            best_c, best_brier = c, res['mean_brier']
    log(f"  -> best C: {best_c} with Brier={best_brier:.5f}")

    # Clip range sweep using OOF preds from best C
    log(f"\n=== {label}: clip range sweep (using best C={best_c}) ===")
    res = loso_cv_brier(make_lr_pipeline(C=best_c), X, y, s, return_oof=True)
    oof = res["oof_preds"]
    best_clip, best_clip_brier = (0.03, 0.97), best_brier
    for lo in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        hi = 1 - lo
        clipped = np.clip(oof, lo, hi)
        b = brier(y, clipped)
        log(f"  clip=[{lo:.2f}, {hi:.2f}]: Brier={b:.5f}")
        if b < best_clip_brier:
            best_clip, best_clip_brier = (lo, hi), b
    log(f"  -> best clip: {best_clip} with Brier={best_clip_brier:.5f}")

    return {"best_c": best_c, "best_brier": best_brier,
            "best_clip": best_clip, "best_clip_brier": best_clip_brier}


def main():
    log("=== MEN ===")
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned_features("men")
    m_result = _tune_gender("Men", m_feat, m_seeds, kl.load_m_tourney_compact,
                              MEN_SEASONS_FULL, m_features)

    log("\n=== WOMEN ===")
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned_features("women")
    w_result = _tune_gender("Women", w_feat, w_seeds, kl.load_w_tourney_compact,
                              WOMEN_SEASONS_FULL, w_features)

    with open(CV_REPORTS / "tuned_params.json", "w") as f:
        json.dump({"men": m_result, "women": w_result}, f, indent=2, default=str)

    print("\nSUMMARY:")
    print(f"  Men:   best C={m_result['best_c']}, clip={m_result['best_clip']}, "
          f"Brier={m_result['best_clip_brier']:.5f}")
    print(f"  Women: best C={w_result['best_c']}, clip={w_result['best_clip']}, "
          f"Brier={w_result['best_clip_brier']:.5f}")


if __name__ == "__main__":
    main()
