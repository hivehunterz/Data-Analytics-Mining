"""
Stage 31: Margin regression + isotonic calibration.

Idea (borrowed from a strong Kaggle public notebook):
  1. Train XGBRegressor on PointDiff (T1_score - T2_score), not on win/loss.
     A 20-point win and a 2-point win become different training examples,
     so the model learns team strength from margin magnitude, not just
     the binary outcome.
  2. Leave-one-season-out CV produces OOF margin predictions for every
     historical tournament game.
  3. Fit IsotonicRegression on (OOF_margin, win_label) per gender. This
     is a monotonic non-parametric mapping margin -> P(T1 wins), Brier-optimal
     by construction.
  4. For 2026 submission: predict margins, then apply the gender's
     calibrator.

This is a DROP-IN replacement for the LR+XGB classifier blend. It does
NOT apply market blend or injury penalty — we want to isolate whether
the margin-regression approach alone is competitive with our full
pipeline (0.12287 local / 0.1213180 Kaggle).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from config import (PROCESSED, SUBMISSIONS, TRUTH_DIR,
                     MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     MEN_ID_CUTOFF, RANDOM_STATE, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.utils import log, brier


XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=700,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.6,
    colsample_bynode=0.8,
    min_child_weight=4,
    tree_method="hist",
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


def _load_pruned(gender, json_name=None, key=None):
    import json
    from config import CV_REPORTS
    p = CV_REPORTS / (json_name or f"{gender}_full_pruned.json")
    with open(p) as f:
        payload = json.load(f)
    if key:
        payload = payload[key]
    feat_list = payload.get("final_features") or payload.get("features")
    return [c.replace("diff_", "") for c in feat_list if c.replace("diff_", "") != "SeedNum"]


def impute_fit_predict(X_train, y_train, X_test):
    imp = SimpleImputer(strategy="median").fit(X_train)
    Xt = imp.transform(X_train)
    Xs = imp.transform(X_test)
    model = XGBRegressor(**XGB_PARAMS).fit(Xt, y_train)
    return model.predict(Xs), model, imp


def loso_margins(train_df, diff_cols):
    """Leave-one-season-out: for each season s, train on all others,
    predict margin for season s. Returns array of OOF margins aligned
    to train_df rows, plus the final model trained on ALL seasons."""
    seasons = sorted(train_df["Season"].unique())
    oof = np.full(len(train_df), np.nan)
    for s in seasons:
        tr = train_df["Season"] != s
        te = train_df["Season"] == s
        Xtr = train_df.loc[tr, diff_cols].values
        ytr = train_df.loc[tr, "margin"].values
        Xte = train_df.loc[te, diff_cols].values
        preds, _, _ = impute_fit_predict(Xtr, ytr, Xte)
        oof[te.values] = preds
        log(f"  LOSO {s}: n_val={te.sum()}, MAE={np.mean(np.abs(preds - train_df.loc[te, 'margin'].values)):.2f}")
    # Fit final model on full history for 2026 submission
    imp = SimpleImputer(strategy="median").fit(train_df[diff_cols].values)
    Xall = imp.transform(train_df[diff_cols].values)
    yall = train_df["margin"].values
    final = XGBRegressor(**XGB_PARAMS).fit(Xall, yall)
    return oof, final, imp


def main():
    log("=== STAGE 31: MARGIN REGRESSION + ISOTONIC ===")
    sub = kl.load_sample_submission_stage2()
    sub_m = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    # ── Men ──
    log("\nMen: assembling features...")
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
    m_feat = m_feat.merge(m_bart[["Season", "TeamID"] + BARTT_COLS],
                            on=["Season", "TeamID"], how="left")

    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    m_tourney = kl.load_m_tourney_compact()
    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_BARTT)]
    train_m = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    diff_cols_m = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]

    log(f"Men: {len(train_m)} training rows, {len(diff_cols_m)} features")
    log("Men: running LOSO margin regression...")
    oof_m, final_m, imp_m = loso_margins(train_m, diff_cols_m)

    # Fit isotonic on (OOF margin, win label)
    iso_m = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_m.fit(oof_m, train_m["label"].values)
    oof_probs_m = iso_m.predict(oof_m)
    log(f"Men OOF Brier (isotonic on margin): {brier(train_m['label'].values, oof_probs_m):.5f}")

    # Predict 2026
    sub_m_feat = build_submission_matchups(sub_m, m_feat, m_features, seeds_df=m_seeds)
    X_sub_m = imp_m.transform(sub_m_feat[diff_cols_m].values)
    margin_pred_m = final_m.predict(X_sub_m)
    sub_m_feat["Pred"] = iso_m.predict(margin_pred_m)

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
    diff_cols_w = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]

    log(f"Women: {len(train_w)} training rows, {len(diff_cols_w)} features")
    log("Women: running LOSO margin regression...")
    oof_w, final_w, imp_w = loso_margins(train_w, diff_cols_w)

    iso_w = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_w.fit(oof_w, train_w["label"].values)
    oof_probs_w = iso_w.predict(oof_w)
    log(f"Women OOF Brier (isotonic on margin): {brier(train_w['label'].values, oof_probs_w):.5f}")

    sub_w_feat = build_submission_matchups(sub_w, w_feat, w_features, seeds_df=w_seeds)
    X_sub_w = imp_w.transform(sub_w_feat[diff_cols_w].values)
    margin_pred_w = final_w.predict(X_sub_w)
    sub_w_feat["Pred"] = iso_w.predict(margin_pred_w)

    # ── Score vs 2026 truth ──
    tm = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    tm["T1"] = tm[["WTeamID","LTeamID"]].min(axis=1)
    tm["T2"] = tm[["WTeamID","LTeamID"]].max(axis=1)
    tm["y"]  = (tm["WTeamID"] == tm["T1"]).astype(int)
    jm = tm.merge(sub_m_feat[["T1","T2","Pred"]], on=["T1","T2"], how="inner")
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
    print("STAGE 31 — margin regression + per-gender isotonic")
    print("="*60)
    print(f"  Men   (n={nm}): Brier = {men_b:.5f}")
    print(f"  Women (n={nw}): Brier = {women_b:.5f}")
    print(f"  Combined (n={nm+nw}): Brier = {combined:.5f}")
    print()
    print(f"  vs our current full pipeline: 0.12287 (local) / 0.12132 (Kaggle)")
    print(f"  delta: {combined - 0.12287:+.5f}")
    print()
    print("  vs leaderboard:")
    print(f"    1st: 0.10975  gap = {combined - 0.10975:+.5f}")
    print(f"    3rd: 0.11604  gap = {combined - 0.11604:+.5f}")

    all_sub = pd.concat([sub_m_feat, sub_w_feat], ignore_index=True)
    all_sub["Pred"] = all_sub["Pred"].fillna(0.5)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / "v2_margin_regression.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
