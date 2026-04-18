"""
Stage 20: Final submission with Barttorvik features (men) + 2015-2025 training.

Men's: LR C=0.03, features from men_barttorvik.json backward-elim result,
        training 2015-2025 only (where Barttorvik is complete).
Women's: unchanged from v2_tuned (no Barttorvik for women).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from config import (PROCESSED, CV_REPORTS, SUBMISSIONS,
                     MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     LR_C_WOMEN, MEN_ID_CUTOFF,
                     CLIP_LO, CLIP_HI, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.postprocess.clip import clip_predictions
from src.utils import log


# Best params from Stage 19 Option B
MEN_C = 0.03

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
    m_feat = m_feat.merge(
        m_bart[["Season", "TeamID"] + BARTT_COLS],
        on=["Season", "TeamID"], how="left",
    )
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


def _load_pruned_features(gender: str, json_name: str = None,
                           key: str = None) -> list:
    p = CV_REPORTS / (json_name or f"{gender}_full_pruned.json")
    with open(p) as f:
        payload = json.load(f)
    if key:
        payload = payload[key]
    # Support both key names ("final_features" from stage 10, "features" from stage 19)
    feat_list = payload.get("final_features") or payload.get("features")
    feats = [c.replace("diff_", "") for c in feat_list
             if c.replace("diff_", "") != "SeedNum"]
    return feats


def _predict(feat, seeds, tourney_loader, seasons, features, C, sub):
    tourney = tourney_loader()
    tourney_in = tourney[tourney["Season"].isin(seasons)]
    train_m = build_tourney_matchups(tourney_in, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values
    model = make_lr_pipeline(C=C).fit(X_train, y_train)
    log(f"  trained on {len(X_train)} matchups ({len(seasons)} seasons)")

    sub_m = build_submission_matchups(sub, feat, features, seeds_df=seeds)
    X_sub = sub_m[diff_cols].values
    probs = model.predict_proba(X_sub)[:, 1]
    sub_m["Pred"] = clip_predictions(probs, CLIP_LO, CLIP_HI)
    return sub_m


def main(tag: str = "v2_final_bartt"):
    log("loading submission template...")
    sub = kl.load_sample_submission_stage2()
    sub_m_all = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w_all = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    log("\n=== MEN (with Barttorvik, 2015-2025) ===")
    m_feat, m_seeds = _assemble_men_with_bartt()
    # Use pruned features from the 2015-2025 run (Option B)
    m_features = _load_pruned_features("men", "men_barttorvik.json", key="B_restricted")
    log(f"  using {len(m_features)} pruned features")
    pred_m = _predict(m_feat, m_seeds, kl.load_m_tourney_compact,
                      MEN_SEASONS_BARTT, m_features, MEN_C, sub_m_all)

    log("\n=== WOMEN (unchanged) ===")
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned_features("women")
    log(f"  using {len(w_features)} pruned features")
    pred_w = _predict(w_feat, w_seeds, kl.load_w_tourney_compact,
                      WOMEN_SEASONS_FULL, w_features, LR_C_WOMEN, sub_w_all)

    all_sub = pd.concat([pred_m, pred_w], ignore_index=True)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / f"{tag}_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path} with {len(out)} rows")

    log(f"Men's pred:   mean={pred_m['Pred'].mean():.3f}")
    log(f"Women's pred: mean={pred_w['Pred'].mean():.3f}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_final_bartt"
    main(tag)
