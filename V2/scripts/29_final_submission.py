"""
Stage 29: FINAL production submission.

Pipeline (men): LR + XGB 70/30 blend + injury Elo penalty
              + Tier 1 Vegas (52 games) + Tier 2 Bradley-Terry
              + NO CLIPPING (sweep showed optimal for 2026).
Pipeline (women): pure LR + NO CLIPPING.

Produces v2_final_submission.csv and scores against 2026 truth.
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
                     LR_C_WOMEN, MEN_ID_CUTOFF, RANDOM_STATE, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.models.market_blend_v2 import apply_full_blend, load_injuries
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


def main():
    log("=== FINAL V2 SUBMISSION (no clipping) ===")
    sub = kl.load_sample_submission_stage2()
    sub_m = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    # ── Men ──
    log("\nMen: assembling features (with injury Elo penalty)...")
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_bart = pd.read_parquet(PROCESSED / "m_barttorvik.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()
    injury_penalty = load_injuries()
    mask = m_elo["Season"] == HOLDOUT_SEASON
    m_elo = m_elo.copy()
    pen_map = dict(zip(injury_penalty["TeamID"], injury_penalty["elo_penalty"]))
    adj = m_elo.loc[mask, "TeamID"].map(pen_map).fillna(0)
    m_elo.loc[mask, "Elo"] = m_elo.loc[mask, "Elo"] - adj.values

    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    m_feat = m_feat.merge(m_bart[["Season", "TeamID"] + BARTT_COLS],
                            on=["Season", "TeamID"], how="left")

    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    m_tourney = kl.load_m_tourney_compact()
    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_BARTT)]
    train_m = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    diff_cols = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]
    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values
    lr = make_lr_pipeline(C=MEN_C).fit(X_train, y_train)
    xgb = xgb_pipe(**MEN_XGB_PARAMS).fit(X_train, y_train)
    log(f"  trained LR + XGB (70/30 blend) on {len(X_train)} matchups")

    sub_m_feat = build_submission_matchups(sub_m, m_feat, m_features, seeds_df=m_seeds)
    X_sub = sub_m_feat[diff_cols].values
    sub_m_feat["Pred"] = MEN_BLEND_ALPHA_LR * lr.predict_proba(X_sub)[:, 1] + \
                        (1 - MEN_BLEND_ALPHA_LR) * xgb.predict_proba(X_sub)[:, 1]
    post_men = apply_full_blend(sub_m_feat, alpha_t1=0.20, alpha_t2=0.75)
    # NO CLIPPING

    # ── Women ──
    log("\nWomen: assembling features...")
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
    w_features = _load_pruned("women")
    w_tourney = kl.load_w_tourney_compact()
    w_tour_in = w_tourney[w_tourney["Season"].isin(WOMEN_SEASONS_FULL)]
    train_w = build_tourney_matchups(w_tour_in, w_feat, w_features, seeds_df=w_seeds)
    w_diff = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]
    X_tw = train_w[w_diff].values
    y_tw = train_w["label"].values
    lr_w = make_lr_pipeline(C=LR_C_WOMEN).fit(X_tw, y_tw)
    sub_w_feat = build_submission_matchups(sub_w, w_feat, w_features, seeds_df=w_seeds)
    X_sw = sub_w_feat[w_diff].values
    sub_w_feat["Pred"] = lr_w.predict_proba(X_sw)[:, 1]
    # NO CLIPPING

    # ── Score ──
    tm = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    tm["T1"] = tm[["WTeamID","LTeamID"]].min(axis=1)
    tm["T2"] = tm[["WTeamID","LTeamID"]].max(axis=1)
    tm["y"]  = (tm["WTeamID"] == tm["T1"]).astype(int)
    jm = tm.merge(post_men[["T1","T2","Pred"]], on=["T1","T2"], how="inner")
    men_b = brier(jm["y"].values, jm["Pred"].values)

    tw = pd.read_csv(TRUTH_DIR / "womens_2026_truth.csv")
    tw["T1"] = tw[["WTeamID","LTeamID"]].min(axis=1)
    tw["T2"] = tw[["WTeamID","LTeamID"]].max(axis=1)
    tw["y"]  = (tw["WTeamID"] == tw["T1"]).astype(int)
    jw = tw.merge(sub_w_feat[["T1","T2","Pred"]], on=["T1","T2"], how="inner")
    women_b = brier(jw["y"].values, jw["Pred"].values)

    nm, nw = len(jm), len(jw)
    combined = (men_b*nm + women_b*nw) / (nm + nw)

    print("\n" + "="*60)
    print("FINAL V2 SUBMISSION — no clipping, full market coverage")
    print("="*60)
    print(f"  Men   (n={nm}): Brier = {men_b:.5f}")
    print(f"  Women (n={nw}): Brier = {women_b:.5f}")
    print(f"  Combined (n={nm+nw}): Brier = {combined:.5f}")
    print()
    print("  vs leaderboard (combined):")
    print(f"    1st: 0.10975  gap = {combined - 0.10975:+.5f}")
    print(f"    2nd: 0.11499  gap = {combined - 0.11499:+.5f}")
    print(f"    3rd: 0.11604  gap = {combined - 0.11604:+.5f}")

    # Save
    all_sub = pd.concat([post_men, sub_w_feat], ignore_index=True)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / "v2_final_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
