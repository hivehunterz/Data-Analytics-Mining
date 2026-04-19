"""
Stage 23: Apply injury-adjusted Elo + tiered market blend + score.

Pipeline:
  1. Load men's team features INCLUDING 2026 Elo.
  2. Subtract per-team injury penalty from 2026 Elo for affected teams.
  3. Retrain LR on 2015-2025 (injuries don't affect training — we only
     have 2026 injury data).
  4. Predict Stage 2 matchups.
  5. Apply tiered market blend (Vegas Tier 1, Bradley-Terry Tier 2).
  6. Clip to [0.03, 0.97].
  7. Score against 2026 men's + women's truth.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from config import (PROCESSED, CV_REPORTS, SUBMISSIONS,
                     MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     LR_C_WOMEN, MEN_ID_CUTOFF, CLIP_LO, CLIP_HI,
                     HOLDOUT_SEASON, TRUTH_DIR)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.models.market_blend_v2 import apply_full_blend, load_injuries
from src.postprocess.clip import clip_predictions
from src.utils import log, brier


MEN_C = 0.03

BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


def _score_vs_truth(men_sub, women_sub):
    out = {}
    # Men
    t = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    t["T1"] = t[["WTeamID", "LTeamID"]].min(axis=1)
    t["T2"] = t[["WTeamID", "LTeamID"]].max(axis=1)
    t["y"]  = (t["WTeamID"] == t["T1"]).astype(int)
    m = t.merge(men_sub[["T1", "T2", "Pred"]], on=["T1", "T2"], how="inner")
    out["men"] = {"brier": brier(m["y"].values, m["Pred"].values), "n": len(m)}
    # Women
    tw = pd.read_csv(TRUTH_DIR / "womens_2026_truth.csv")
    tw["T1"] = tw[["WTeamID", "LTeamID"]].min(axis=1)
    tw["T2"] = tw[["WTeamID", "LTeamID"]].max(axis=1)
    tw["y"]  = (tw["WTeamID"] == tw["T1"]).astype(int)
    mw = tw.merge(women_sub[["T1", "T2", "Pred"]], on=["T1", "T2"], how="inner")
    out["women"] = {"brier": brier(mw["y"].values, mw["Pred"].values), "n": len(mw)}
    # Combined (weighted by n)
    nm, nw = out["men"]["n"], out["women"]["n"]
    out["combined"] = {
        "brier": (out["men"]["brier"] * nm + out["women"]["brier"] * nw) / (nm + nw),
        "n": nm + nw,
    }
    return out


def _assemble_men_with_bartt_and_injuries():
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_bart = pd.read_parquet(PROCESSED / "m_barttorvik.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()

    # Apply injury penalty to 2026 Elo
    injury_penalty = load_injuries()
    mask_2026 = m_elo["Season"] == HOLDOUT_SEASON
    m_elo = m_elo.copy()
    pen_map = dict(zip(injury_penalty["TeamID"], injury_penalty["elo_penalty"]))
    adjustments = m_elo.loc[mask_2026, "TeamID"].map(pen_map).fillna(0)
    m_elo.loc[mask_2026, "Elo"] = m_elo.loc[mask_2026, "Elo"] - adjustments.values
    adjusted = (adjustments != 0).sum()
    log(f"injuries: applied Elo penalty to {adjusted} 2026 teams")

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


def _load_pruned_features(gender: str, json_name: str = None, key: str = None):
    p = CV_REPORTS / (json_name or f"{gender}_full_pruned.json")
    with open(p) as f:
        payload = json.load(f)
    if key:
        payload = payload[key]
    feat_list = payload.get("final_features") or payload.get("features")
    return [c.replace("diff_", "") for c in feat_list if c.replace("diff_", "") != "SeedNum"]


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
    sub_m["Pred"] = model.predict_proba(X_sub)[:, 1]
    return sub_m


def main(tag: str = "v2_full_blend"):
    log("loading submission template...")
    sub = kl.load_sample_submission_stage2()
    sub_m_all = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w_all = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    log("\n=== MEN ===")
    m_feat, m_seeds = _assemble_men_with_bartt_and_injuries()
    m_features = _load_pruned_features("men", "men_barttorvik.json", key="B_restricted")
    log(f"  using {len(m_features)} pruned features")
    pred_m = _predict(m_feat, m_seeds, kl.load_m_tourney_compact,
                      MEN_SEASONS_BARTT, m_features, MEN_C, sub_m_all)
    log(f"  LR men predictions: {len(pred_m)}")

    log("\n=== WOMEN ===")
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned_features("women")
    log(f"  using {len(w_features)} pruned features")
    pred_w = _predict(w_feat, w_seeds, kl.load_w_tourney_compact,
                      WOMEN_SEASONS_FULL, w_features, LR_C_WOMEN, sub_w_all)

    log("\n=== BLEND (men's only) ===")
    # Score BEFORE blend to see injury-only effect
    pred_m_pre = pred_m.copy()
    pred_m_pre["Pred"] = clip_predictions(pred_m_pre["Pred"].values, CLIP_LO, CLIP_HI)
    pre_scores = _score_vs_truth(pred_m_pre, pred_w)
    log(f"  PRE-BLEND (LR + Barttorvik + injuries, no market):")
    log(f"    men: {pre_scores['men']['brier']:.5f}  ({pre_scores['men']['n']})")
    log(f"    women: {pre_scores['women']['brier']:.5f}  ({pre_scores['women']['n']})")
    log(f"    combined: {pre_scores['combined']['brier']:.5f}")

    # Apply blend
    pred_m_blended = apply_full_blend(pred_m, alpha_t1=0.20, alpha_t2=0.75)
    pred_m_blended["Pred"] = clip_predictions(pred_m_blended["Pred"].values, CLIP_LO, CLIP_HI)

    # Save
    all_sub = pd.concat([pred_m_blended, pred_w], ignore_index=True)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / f"{tag}_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"saved {out_path}")

    # Score final
    final_scores = _score_vs_truth(pred_m_blended, pred_w)
    print("\n=== FINAL (with injuries + full market blend) ===")
    print(f"  men:      Brier = {final_scores['men']['brier']:.5f}  (n={final_scores['men']['n']})")
    print(f"  women:    Brier = {final_scores['women']['brier']:.5f}  (n={final_scores['women']['n']})")
    print(f"  combined: Brier = {final_scores['combined']['brier']:.5f}  (n={final_scores['combined']['n']})")
    print()
    print("  vs leaderboard (combined):")
    print(f"    Harry   (1st): 0.10975  gap = {final_scores['combined']['brier'] - 0.10975:+.5f}")
    print(f"    Brendan (2nd): 0.11499  gap = {final_scores['combined']['brier'] - 0.11499:+.5f}")
    print(f"    Kevin   (3rd): 0.11604  gap = {final_scores['combined']['brier'] - 0.11604:+.5f}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_full_blend"
    main(tag)
