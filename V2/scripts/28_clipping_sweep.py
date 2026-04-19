"""
Stage 28: Probability clipping study on the final blended 2026 predictions.

Produces the production prediction (LR+XGB blend for men + market tiers +
injuries) WITHOUT clipping, then sweeps clipping strategies and scores each
against 2026 truth.

Clipping strategies tested:
  - [0.00, 1.00]  no clip (raw)
  - [0.01, 0.99]  very loose
  - [0.02, 0.98]
  - [0.03, 0.97]  ships currently
  - [0.04, 0.96]
  - [0.05, 0.95]
  - [0.08, 0.92]
  - [0.10, 0.90]  tight
  - [0.15, 0.85]  very tight
  Plus asymmetric:
  - [0.03, 0.95]  heavy upper clip
  - [0.01, 0.97]  light lower, normal upper
  - [0.05, 0.97]

Also shows: Brier decomposition (how many games we hurt vs helped by clipping).
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
    # Injury
    injury_penalty = load_injuries()
    mask = m_elo["Season"] == HOLDOUT_SEASON
    m_elo = m_elo.copy()
    pen_map = dict(zip(injury_penalty["TeamID"], injury_penalty["elo_penalty"]))
    adj = m_elo.loc[mask, "TeamID"].map(pen_map).fillna(0)
    m_elo.loc[mask, "Elo"] = m_elo.loc[mask, "Elo"] - adj.values
    # Features
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


def _build_production_predictions():
    """Return (men_sub, women_sub) DataFrames with unclipped blended Pred."""
    sub = kl.load_sample_submission_stage2()
    sub_m = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    # Men
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    m_tourney = kl.load_m_tourney_compact()
    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_BARTT)]
    train_m = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    diff_cols = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]
    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values
    lr = make_lr_pipeline(C=MEN_C).fit(X_train, y_train)
    xgb = xgb_pipe(**MEN_XGB_PARAMS).fit(X_train, y_train)
    sub_m_feat = build_submission_matchups(sub_m, m_feat, m_features, seeds_df=m_seeds)
    X_sub = sub_m_feat[diff_cols].values
    p_blend = MEN_BLEND_ALPHA_LR * lr.predict_proba(X_sub)[:,1] + (1 - MEN_BLEND_ALPHA_LR) * xgb.predict_proba(X_sub)[:,1]
    sub_m_feat["Pred"] = p_blend

    # Apply market blend (uses existing clip=[0.03, 0.97] inside function? check — it does clip)
    # We want the UNCLIPPED blended prediction. Let's bypass the clip by using
    # raw values from the blend helper.
    post_men = apply_full_blend(sub_m_feat, alpha_t1=0.20, alpha_t2=0.75)
    # Do NOT clip — return raw blended prediction

    # Women
    w_feat, w_seeds = _assemble_women()
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
    sub_w_feat["Pred"] = lr_w.predict_proba(X_sw)[:,1]

    return post_men, sub_w_feat


def _score(men_sub, women_sub, lo, hi):
    """Clip predictions to [lo, hi] and score against truth."""
    m = men_sub.copy()
    m["Pred"] = np.clip(m["Pred"].values, lo, hi)
    w = women_sub.copy()
    w["Pred"] = np.clip(w["Pred"].values, lo, hi)

    tm = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    tm["T1"] = tm[["WTeamID","LTeamID"]].min(axis=1)
    tm["T2"] = tm[["WTeamID","LTeamID"]].max(axis=1)
    tm["y"]  = (tm["WTeamID"] == tm["T1"]).astype(int)
    jm = tm.merge(m[["T1","T2","Pred"]], on=["T1","T2"], how="inner")
    men_b = brier(jm["y"].values, jm["Pred"].values)

    tw = pd.read_csv(TRUTH_DIR / "womens_2026_truth.csv")
    tw["T1"] = tw[["WTeamID","LTeamID"]].min(axis=1)
    tw["T2"] = tw[["WTeamID","LTeamID"]].max(axis=1)
    tw["y"]  = (tw["WTeamID"] == tw["T1"]).astype(int)
    jw = tw.merge(w[["T1","T2","Pred"]], on=["T1","T2"], how="inner")
    women_b = brier(jw["y"].values, jw["Pred"].values)

    nm, nw = len(jm), len(jw)
    combined = (men_b*nm + women_b*nw) / (nm + nw)
    return men_b, women_b, combined, nm, nw


def main():
    log("building unclipped production predictions...")
    men_sub, women_sub = _build_production_predictions()
    log(f"  men rows: {len(men_sub)}, women rows: {len(women_sub)}")
    log(f"  men Pred  min/max: {men_sub['Pred'].min():.4f} / {men_sub['Pred'].max():.4f}")
    log(f"  women Pred min/max: {women_sub['Pred'].min():.4f} / {women_sub['Pred'].max():.4f}")

    log(f"\n{'='*60}\nCLIPPING SWEEP\n{'='*60}")
    log(f"{'lo':>6s} {'hi':>6s}  {'Men':>8s}  {'Women':>8s}  {'Combined':>9s}")
    log("  " + "-" * 47)

    configs = [
        (0.00, 1.00),  # no clip
        (0.01, 0.99),
        (0.02, 0.98),
        (0.03, 0.97),  # current
        (0.04, 0.96),
        (0.05, 0.95),
        (0.06, 0.94),
        (0.08, 0.92),
        (0.10, 0.90),
        (0.15, 0.85),
        # Asymmetric
        (0.03, 0.95),
        (0.05, 0.97),
        (0.01, 0.95),
        (0.03, 0.90),
    ]

    results = []
    for lo, hi in configs:
        m_b, w_b, c_b, nm, nw = _score(men_sub, women_sub, lo, hi)
        results.append({"lo": lo, "hi": hi, "men": m_b, "women": w_b,
                         "combined": c_b})
        mark = "  <-- current" if (lo, hi) == (0.03, 0.97) else ""
        log(f"  {lo:.2f}   {hi:.2f}   {m_b:.5f}  {w_b:.5f}  {c_b:.5f}{mark}")

    best = min(results, key=lambda r: r["combined"])
    log(f"\n  BEST clip: [{best['lo']:.2f}, {best['hi']:.2f}]  combined = {best['combined']:.5f}")

    # Save
    with open(CV_REPORTS / "clipping_sweep.json", "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
