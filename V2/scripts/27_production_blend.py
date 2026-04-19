"""
Stage 27: Production submission with LR+XGB blend (men) on top of ALL post-processing.

Men:
  base = 0.70 * LR + 0.30 * XGB (tuned: depth=5, lr=0.01, reg_lambda=0.5)
  + injury Elo penalty
  + Tier 1 Vegas market blend (alpha_t1 = 0.20)
  + Tier 2 Bradley-Terry from championship futures (alpha_t2 = 0.75)
  + clip [0.03, 0.97]

Women:
  base = LR (pure — blend doesn't help per stage 26)
  + clip [0.03, 0.97]

Then score against 2026 truth.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from config import (PROCESSED, CV_REPORTS, SUBMISSIONS, TRUTH_DIR,
                     MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     LR_C_WOMEN, MEN_ID_CUTOFF, CLIP_LO, CLIP_HI,
                     HOLDOUT_SEASON, RANDOM_STATE)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.models.market_blend_v2 import apply_full_blend, load_injuries
from src.postprocess.clip import clip_predictions
from src.utils import log, brier


MEN_C = 0.03
MEN_XGB_PARAMS = dict(max_depth=5, learning_rate=0.01, reg_lambda=0.5, reg_alpha=0,
                     n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=5,
                     eval_metric="logloss", tree_method="hist",
                     random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
MEN_BLEND_ALPHA_LR = 0.70
BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


def xgb_pipe(**kw):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("xgb", XGBClassifier(**kw))])


def _load_pruned(gender, json_name=None, key=None):
    p = CV_REPORTS / (json_name or f"{gender}_full_pruned.json")
    with open(p) as f:
        payload = json.load(f)
    if key:
        payload = payload[key]
    feat_list = payload.get("final_features") or payload.get("features")
    return [c.replace("diff_", "") for c in feat_list if c.replace("diff_", "") != "SeedNum"]


def _assemble_men():
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_bart = pd.read_parquet(PROCESSED / "m_barttorvik.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()

    # Injury Elo penalty
    injury_penalty = load_injuries()
    mask_2026 = m_elo["Season"] == HOLDOUT_SEASON
    m_elo = m_elo.copy()
    pen_map = dict(zip(injury_penalty["TeamID"], injury_penalty["elo_penalty"]))
    adjustments = m_elo.loc[mask_2026, "TeamID"].map(pen_map).fillna(0)
    m_elo.loc[mask_2026, "Elo"] = m_elo.loc[mask_2026, "Elo"] - adjustments.values
    log(f"injuries: applied Elo penalty to {(adjustments!=0).sum()} teams")

    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    m_feat = m_feat.merge(m_bart[["Season", "TeamID"] + BARTT_COLS],
                            on=["Season", "TeamID"], how="left")
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


def _score(men_sub, women_sub):
    t = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    t["T1"] = t[["WTeamID", "LTeamID"]].min(axis=1)
    t["T2"] = t[["WTeamID", "LTeamID"]].max(axis=1)
    t["y"]  = (t["WTeamID"] == t["T1"]).astype(int)
    m = t.merge(men_sub[["T1", "T2", "Pred"]], on=["T1", "T2"], how="inner")
    men_brier = brier(m["y"].values, m["Pred"].values)

    tw = pd.read_csv(TRUTH_DIR / "womens_2026_truth.csv")
    tw["T1"] = tw[["WTeamID", "LTeamID"]].min(axis=1)
    tw["T2"] = tw[["WTeamID", "LTeamID"]].max(axis=1)
    tw["y"]  = (tw["WTeamID"] == tw["T1"]).astype(int)
    mw = tw.merge(women_sub[["T1", "T2", "Pred"]], on=["T1", "T2"], how="inner")
    women_brier = brier(mw["y"].values, mw["Pred"].values)

    nm, nw = len(m), len(mw)
    combined = (men_brier * nm + women_brier * nw) / (nm + nw)
    return {"men": (men_brier, nm), "women": (women_brier, nw),
             "combined": (combined, nm + nw)}


def main():
    log("loading submission template...")
    sub = kl.load_sample_submission_stage2()
    sub_m_all = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w_all = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    # ── MEN: train LR + XGB, blend, apply markets ────────────────────
    log("\n=== MEN: LR + XGB blend ===")
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    log(f"  using {len(m_features)} pruned features")

    m_tourney = kl.load_m_tourney_compact()
    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_BARTT)]
    train_m = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    diff_cols = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]
    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values

    lr_model = make_lr_pipeline(C=MEN_C).fit(X_train, y_train)
    xgb_model = xgb_pipe(**MEN_XGB_PARAMS).fit(X_train, y_train)
    log(f"  trained LR + XGB on {len(X_train)} matchups")

    sub_m_features = build_submission_matchups(sub_m_all, m_feat, m_features, seeds_df=m_seeds)
    X_sub = sub_m_features[diff_cols].values
    p_lr = lr_model.predict_proba(X_sub)[:, 1]
    p_xgb = xgb_model.predict_proba(X_sub)[:, 1]
    p_blend = MEN_BLEND_ALPHA_LR * p_lr + (1 - MEN_BLEND_ALPHA_LR) * p_xgb
    sub_m_features["Pred"] = p_blend
    log(f"  LR+XGB blend complete (alpha_LR={MEN_BLEND_ALPHA_LR})")

    # ── WOMEN: LR only ───────────────────────────────────────────────
    log("\n=== WOMEN: LR only ===")
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned("women")
    log(f"  using {len(w_features)} pruned features")
    w_tourney = kl.load_w_tourney_compact()
    w_tour_in = w_tourney[w_tourney["Season"].isin(WOMEN_SEASONS_FULL)]
    train_w = build_tourney_matchups(w_tour_in, w_feat, w_features, seeds_df=w_seeds)
    w_diff = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]
    X_train_w = train_w[w_diff].values
    y_train_w = train_w["label"].values
    lr_model_w = make_lr_pipeline(C=LR_C_WOMEN).fit(X_train_w, y_train_w)
    sub_w_features = build_submission_matchups(sub_w_all, w_feat, w_features, seeds_df=w_seeds)
    X_sub_w = sub_w_features[w_diff].values
    sub_w_features["Pred"] = lr_model_w.predict_proba(X_sub_w)[:, 1]

    # Clip women's now
    sub_w_features["Pred"] = clip_predictions(sub_w_features["Pred"].values,
                                                CLIP_LO, CLIP_HI)

    # ── Score pre-market-blend baseline ──────────────────────────────
    pre_men = sub_m_features.copy()
    pre_men["Pred"] = clip_predictions(pre_men["Pred"].values, CLIP_LO, CLIP_HI)
    pre_scores = _score(pre_men, sub_w_features)
    log(f"\n  PRE-market-blend (LR+XGB blend men, pure LR women):")
    log(f"    men: {pre_scores['men'][0]:.5f}  ({pre_scores['men'][1]} games)")
    log(f"    women: {pre_scores['women'][0]:.5f}  ({pre_scores['women'][1]} games)")
    log(f"    combined: {pre_scores['combined'][0]:.5f}")

    # ── Apply market blend to men ────────────────────────────────────
    log(f"\n=== MEN: apply Tier 1 (Vegas) + Tier 2 (Bradley-Terry) ===")
    post_men = apply_full_blend(sub_m_features, alpha_t1=0.20, alpha_t2=0.75)
    post_men["Pred"] = clip_predictions(post_men["Pred"].values, CLIP_LO, CLIP_HI)

    # ── Final score + save submission ────────────────────────────────
    final_scores = _score(post_men, sub_w_features)
    print("\n" + "="*60)
    print("FINAL SCORES (LR+XGB blend + injuries + market)")
    print("="*60)
    print(f"  men:      Brier = {final_scores['men'][0]:.5f}  (n={final_scores['men'][1]})")
    print(f"  women:    Brier = {final_scores['women'][0]:.5f}  (n={final_scores['women'][1]})")
    print(f"  combined: Brier = {final_scores['combined'][0]:.5f}  (n={final_scores['combined'][1]})")
    print()
    print("  vs leaderboard (combined):")
    print(f"    1st: 0.10975  gap = {final_scores['combined'][0] - 0.10975:+.5f}")
    print(f"    2nd: 0.11499  gap = {final_scores['combined'][0] - 0.11499:+.5f}")
    print(f"    3rd: 0.11604  gap = {final_scores['combined'][0] - 0.11604:+.5f}")

    all_sub = pd.concat([post_men, sub_w_features], ignore_index=True)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / "v2_lr_xgb_blend_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
