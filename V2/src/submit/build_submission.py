"""
Produce a Kaggle-format submission CSV.

Trains final LR models on all training seasons (2003-2025 men, 2010-2025
women), predicts every row in SampleSubmissionStage2.csv, clips to
[0.03, 0.97], writes submission CSV.

Uses the pruned feature lists from scripts/10_full_feature_cv.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (PROCESSED, SUBMISSIONS, CV_REPORTS,
                     MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN, HOLDOUT_SEASON,
                     CLIP_LO, CLIP_HI, MEN_ID_CUTOFF)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.postprocess.clip import clip_predictions
from src.utils import log


def _assemble_men_features():
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


def _assemble_women_features():
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


def _load_pruned_features(gender: str) -> list:
    path = CV_REPORTS / f"{gender}_full_pruned.json"
    with open(path) as f:
        payload = json.load(f)
    # Strip "diff_" prefix, drop "SeedNum" since it's added by build_matchups
    feats = [c.replace("diff_", "") for c in payload["final_features"]]
    feats = [f for f in feats if f != "SeedNum"]
    return feats


def _train_and_predict(feat, seeds, tourney_loader, train_seasons: list,
                       features: list, C: float, sub_m: pd.DataFrame):
    tourney = tourney_loader()
    tourney_in = tourney[tourney["Season"].isin(train_seasons)]
    train_matchups = build_tourney_matchups(tourney_in, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X_train = train_matchups[diff_cols].values
    y_train = train_matchups["label"].values

    model = make_lr_pipeline(C=C).fit(X_train, y_train)
    log(f"  trained on {len(X_train)} matchups over {len(train_seasons)} seasons")

    # Predict on Stage 2 submission pairs
    sub_features = build_submission_matchups(sub_m, feat, features, seeds_df=seeds)
    X_sub = sub_features[diff_cols].values
    probs = model.predict_proba(X_sub)[:, 1]
    return probs, sub_features


def main(tag: str = "v2_final"):
    log("loading SampleSubmissionStage2.csv...")
    sub = kl.load_sample_submission_stage2()
    log(f"  rows: {len(sub)}")

    # Split by gender
    sub_m = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()
    log(f"  men's pairs: {len(sub_m)}, women's pairs: {len(sub_w)}")

    # ── Men's ────────────────────────────────────────────────────────
    log("\n=== MEN: assembling features ===")
    m_feat, m_seeds = _assemble_men_features()
    m_features = _load_pruned_features("men")
    log(f"  using {len(m_features)} pruned features")
    m_probs, m_sub = _train_and_predict(
        m_feat, m_seeds, kl.load_m_tourney_compact,
        MEN_SEASONS_FULL, m_features, LR_C_MEN, sub_m,
    )
    m_preds = clip_predictions(m_probs, CLIP_LO, CLIP_HI)
    m_sub["Pred"] = m_preds

    # ── Women's ──────────────────────────────────────────────────────
    log("\n=== WOMEN: assembling features ===")
    w_feat, w_seeds = _assemble_women_features()
    w_features = _load_pruned_features("women")
    log(f"  using {len(w_features)} pruned features")
    w_probs, w_sub = _train_and_predict(
        w_feat, w_seeds, kl.load_w_tourney_compact,
        WOMEN_SEASONS_FULL, w_features, LR_C_WOMEN, sub_w,
    )
    w_preds = clip_predictions(w_probs, CLIP_LO, CLIP_HI)
    w_sub["Pred"] = w_preds

    # ── Assemble submission ──────────────────────────────────────────
    all_sub = pd.concat([m_sub, w_sub], ignore_index=True)
    # Safety: default missing to 0.5 (shouldn't happen with complete data)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / f"{tag}_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path} with {len(out)} rows")

    # Summary stats
    log(f"\nMen's predictions: mean={m_preds.mean():.3f}, "
        f"median={np.median(m_preds):.3f}, "
        f"% near {CLIP_LO}={(m_preds <= CLIP_LO + 1e-6).mean():.1%}, "
        f"% near {CLIP_HI}={(m_preds >= CLIP_HI - 1e-6).mean():.1%}")
    log(f"Women's predictions: mean={w_preds.mean():.3f}, "
        f"median={np.median(w_preds):.3f}, "
        f"% near {CLIP_LO}={(w_preds <= CLIP_LO + 1e-6).mean():.1%}, "
        f"% near {CLIP_HI}={(w_preds >= CLIP_HI - 1e-6).mean():.1%}")

    return out_path


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_final"
    main(tag)
