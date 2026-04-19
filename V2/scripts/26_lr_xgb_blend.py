"""
Stage 26: Minimal LR + best-XGB blend sweep for both genders.
Uses the best hyperparameters found in stage 25:
  Men:   XGB(depth=5, lr=0.01, reg_lambda=0.5, reg_alpha=0)
  Women: XGB(depth=5, lr=0.01, reg_lambda=0.5, reg_alpha=0) — same as a first guess;
         stage 25 women's run didn't complete so we try a few.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     LR_C_WOMEN, RANDOM_STATE, CLIP_LO, CLIP_HI)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, brier


BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


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


def _prep(feat, seeds, tourney_loader, seasons, features):
    tourney = tourney_loader()
    tourney = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    return matchups[diff_cols].values, matchups["label"].values, matchups["Season"].values


def xgb_pipe(**kw):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(eval_metric="logloss", tree_method="hist",
                                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, **kw)),
    ])


def _sweep(label, X, y, s, lr_C, xgb_kwargs):
    log(f"\n=== {label} ===")
    lr_res = loso_cv_brier(make_lr_pipeline(C=lr_C), X, y, s, return_oof=True)
    log(f"  LR  : {lr_res['mean_brier']:.5f}")

    xgb_res = loso_cv_brier(xgb_pipe(**xgb_kwargs), X, y, s, return_oof=True)
    log(f"  XGB : {xgb_res['mean_brier']:.5f}  params={xgb_kwargs}")

    best_alpha, best_b = 1.0, brier(y, np.clip(lr_res["oof_preds"], CLIP_LO, CLIP_HI))
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = alpha * lr_res["oof_preds"] + (1 - alpha) * xgb_res["oof_preds"]
        blend = np.clip(blend, CLIP_LO, CLIP_HI)
        b = brier(y, blend)
        marker = "  <-- BEST" if b == best_b else ""
        log(f"    alpha_LR={alpha:.2f}: Brier={b:.5f}")
        if b < best_b:
            best_alpha, best_b = alpha, b
    log(f"  best blend: alpha_LR={best_alpha}, Brier={best_b:.5f}")
    return {"lr": lr_res['mean_brier'], "xgb": xgb_res['mean_brier'],
            "best_alpha_lr": best_alpha, "best_blend_brier": best_b,
            "lr_oof": lr_res["oof_preds"].tolist(),
            "xgb_oof": xgb_res["oof_preds"].tolist()}


def main():
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    mX, my, ms = _prep(m_feat, m_seeds, kl.load_m_tourney_compact, MEN_SEASONS_BARTT, m_features)
    # Best XGB from stage 25 men's grid
    m_xgb_kw = dict(max_depth=5, learning_rate=0.01, reg_lambda=0.5, reg_alpha=0,
                     n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=5)
    m_out = _sweep("Men", mX, my, ms, lr_C=0.03, xgb_kwargs=m_xgb_kw)

    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned("women")
    wX, wy, ws = _prep(w_feat, w_seeds, kl.load_w_tourney_compact, WOMEN_SEASONS_FULL, w_features)
    # For women: try same shallow+regularized config
    w_xgb_kw = dict(max_depth=3, learning_rate=0.01, reg_lambda=1.0, reg_alpha=0,
                     n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=5)
    w_out = _sweep("Women", wX, wy, ws, lr_C=LR_C_WOMEN, xgb_kwargs=w_xgb_kw)

    # Save (without the big OOF lists, just scalar results)
    save = {
        "men":   {k: v for k, v in m_out.items() if k.endswith("brier") or k == "best_alpha_lr" or k == "lr" or k == "xgb"},
        "women": {k: v for k, v in w_out.items() if k.endswith("brier") or k == "best_alpha_lr" or k == "lr" or k == "xgb"},
    }
    with open(CV_REPORTS / "lr_xgb_blend.json", "w") as f:
        json.dump(save, f, indent=2, default=str)


if __name__ == "__main__":
    main()
