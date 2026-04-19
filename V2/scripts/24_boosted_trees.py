"""
Stage 24: Serious exploration of boosted trees on the FULL feature set.

Earlier comparison (stage 6) used only the pre-Barttorvik, pre-interactions
feature set. Now we have 38 features for men (inc. Barttorvik) and 31 for
women. Maybe boosted trees can exploit non-linear interactions that LR
cannot.

Tests:
  1. XGBoost with a shallow config (depth=2, lr=0.05, n=500).
  2. XGBoost RandomizedSearchCV over depth / lr / n_estimators / subsample.
  3. LightGBM RandomizedSearchCV.
  4. CatBoost RandomizedSearchCV.
  5. Best boosted model vs LR in LOSO CV.
  6. Best boosted model as a BLEND with LR (50/50, 70/30 LR-heavy, etc).

Results saved to outputs/cv_reports/boosted_trees.json.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_BARTT,
                     WOMEN_SEASONS_FULL, LR_C_MEN, LR_C_WOMEN, RANDOM_STATE)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.validation.loso_cv import loso_cv_brier
from src.utils import log, brier


BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]

# ── Data loading helpers ──────────────────────────────────────────────
def _load_pruned_features(gender: str, json_name: str = None, key: str = None):
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


def _prepare_Xy(feat, seeds, tourney_loader, seasons, features):
    tourney = tourney_loader()
    tourney = tourney[tourney["Season"].isin(seasons)]
    matchups = build_tourney_matchups(tourney, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X = matchups[diff_cols].values
    y = matchups["label"].values
    s = matchups["Season"].values
    return X, y, s, diff_cols


# ── Model factories ──────────────────────────────────────────────────
def xgb_pipeline(**kw):
    params = dict(
        eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    params.update(kw)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(**params)),
    ])


def lgbm_pipeline(**kw):
    params = dict(
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, objective="binary",
    )
    params.update(kw)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lgbm", LGBMClassifier(**params)),
    ])


def catboost_pipeline(**kw):
    params = dict(
        random_state=RANDOM_STATE, verbose=False, thread_count=-1,
        loss_function="Logloss",
    )
    params.update(kw)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("cb", CatBoostClassifier(**params)),
    ])


# ── Random search ─────────────────────────────────────────────────────
def _random_search_xgb(X, y, groups, n_iter=20):
    param_dist = {
        "xgb__max_depth": randint(2, 7),
        "xgb__n_estimators": randint(100, 1000),
        "xgb__learning_rate": uniform(0.005, 0.1),
        "xgb__subsample": uniform(0.6, 0.4),
        "xgb__colsample_bytree": uniform(0.5, 0.5),
        "xgb__min_child_weight": randint(1, 10),
        "xgb__reg_alpha": uniform(0, 0.5),
        "xgb__reg_lambda": uniform(0.5, 2.5),
    }
    gkf = GroupKFold(n_splits=min(10, len(np.unique(groups))))
    scorer = make_scorer(brier_score_loss, greater_is_better=False,
                         needs_proba=True)
    search = RandomizedSearchCV(
        xgb_pipeline(), param_dist, n_iter=n_iter, cv=gkf,
        scoring=scorer, random_state=RANDOM_STATE, n_jobs=-1,
        refit=False,
    )
    search.fit(X, y, groups=groups)
    return search


def _random_search_lgbm(X, y, groups, n_iter=20):
    param_dist = {
        "lgbm__max_depth": randint(2, 8),
        "lgbm__n_estimators": randint(100, 1000),
        "lgbm__learning_rate": uniform(0.005, 0.1),
        "lgbm__num_leaves": randint(8, 64),
        "lgbm__subsample": uniform(0.6, 0.4),
        "lgbm__colsample_bytree": uniform(0.5, 0.5),
        "lgbm__reg_alpha": uniform(0, 0.5),
        "lgbm__reg_lambda": uniform(0.5, 2.5),
        "lgbm__min_child_samples": randint(5, 50),
    }
    gkf = GroupKFold(n_splits=min(10, len(np.unique(groups))))
    scorer = make_scorer(brier_score_loss, greater_is_better=False,
                         needs_proba=True)
    search = RandomizedSearchCV(
        lgbm_pipeline(), param_dist, n_iter=n_iter, cv=gkf,
        scoring=scorer, random_state=RANDOM_STATE, n_jobs=-1,
        refit=False,
    )
    search.fit(X, y, groups=groups)
    return search


def _random_search_cb(X, y, groups, n_iter=15):
    param_dist = {
        "cb__depth": randint(3, 8),
        "cb__iterations": randint(100, 800),
        "cb__learning_rate": uniform(0.01, 0.15),
        "cb__l2_leaf_reg": uniform(1, 5),
        "cb__bagging_temperature": uniform(0, 1),
    }
    gkf = GroupKFold(n_splits=min(10, len(np.unique(groups))))
    scorer = make_scorer(brier_score_loss, greater_is_better=False,
                         needs_proba=True)
    search = RandomizedSearchCV(
        catboost_pipeline(), param_dist, n_iter=n_iter, cv=gkf,
        scoring=scorer, random_state=RANDOM_STATE, n_jobs=2,   # catboost parallel internally
        refit=False,
    )
    search.fit(X, y, groups=groups)
    return search


def _loso_eval(pipe, X, y, seasons, label):
    res = loso_cv_brier(pipe, X, y, seasons, return_oof=True)
    log(f"  {label}: Brier = {res['mean_brier']:.5f}  "
        f"logloss = {res['mean_logloss']:.5f}")
    return res


def _blend_with_lr(lr_oof, other_oof, y, label):
    log(f"\n  {label} — blend sweep with LR OOF:")
    best_a, best_b = 1.0, brier(y, np.clip(lr_oof, 0.03, 0.97))
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = alpha * lr_oof + (1 - alpha) * other_oof
        blend = np.clip(blend, 0.03, 0.97)
        b = brier(y, blend)
        log(f"    alpha={alpha:.2f}  (α = LR weight): Brier = {b:.5f}")
        if b < best_b:
            best_a, best_b = alpha, b
    log(f"    -> best alpha = {best_a}, Brier = {best_b:.5f}")
    return best_a, best_b


def _xgb_depth_reg_grid(X, y, seasons):
    """
    Focused 2-D grid of XGBoost depth x regularization.
    Prints a readable table of LOSO Brier across depth and reg_lambda.
    """
    log("\n  [2a] XGB depth x reg_lambda grid")
    depths = [2, 3, 4, 5, 6, 8]
    reg_lambdas = [0.1, 0.5, 1.0, 3.0, 10.0]
    # Fix other params at reasonable defaults
    fixed = dict(
        n_estimators=500, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1,
    )
    header = f"    depth \\ lam  " + "  ".join(f"{l:>7}" for l in reg_lambdas)
    log(header)
    grid_result = {}
    for d in depths:
        row = [f"    depth={d:<2d}   "]
        grid_result[d] = {}
        for lam in reg_lambdas:
            pipe = xgb_pipeline(max_depth=d, reg_lambda=lam, **fixed)
            r = loso_cv_brier(pipe, X, y, seasons, return_oof=False)
            row.append(f"{r['mean_brier']:.5f}")
            grid_result[d][lam] = r["mean_brier"]
        log("  ".join(row))
    return grid_result


def _run_gender(label: str, feat, seeds, tourney_loader, seasons,
                features, lr_C: float, n_iter_xgb=20, n_iter_lgbm=20,
                n_iter_cb=12):
    log(f"\n=== {label} ===")
    X, y, s, diff_cols = _prepare_Xy(feat, seeds, tourney_loader, seasons, features)
    log(f"  data: {X.shape[0]} rows, {X.shape[1]} features, "
        f"{len(np.unique(s))} seasons")

    # 1. LR baseline
    log("\n  [1] LR baseline")
    lr_res = _loso_eval(make_lr_pipeline(C=lr_C), X, y, s, "LR")

    # 2a. XGB depth x reg_lambda focused grid
    depth_reg_grid = _xgb_depth_reg_grid(X, y, s)

    # 2. XGBoost RandomizedSearchCV (wide sweep)
    log(f"\n  [2] XGBoost RandomizedSearchCV ({n_iter_xgb} iter)")
    t0 = time.time()
    xgb_search = _random_search_xgb(X, y, s, n_iter=n_iter_xgb)
    log(f"    search done in {time.time()-t0:.1f}s")
    log(f"    best params: {xgb_search.best_params_}")
    log(f"    best CV Brier: {-xgb_search.best_score_:.5f}")
    xgb_best = xgb_search.best_estimator_ if xgb_search.refit else \
               xgb_pipeline(**{k.replace("xgb__", ""): v
                               for k, v in xgb_search.best_params_.items()})
    xgb_res = _loso_eval(xgb_best, X, y, s, "XGB (tuned)")

    # 3. LightGBM
    log(f"\n  [3] LightGBM RandomizedSearchCV ({n_iter_lgbm} iter)")
    t0 = time.time()
    lgbm_search = _random_search_lgbm(X, y, s, n_iter=n_iter_lgbm)
    log(f"    search done in {time.time()-t0:.1f}s")
    log(f"    best params: {lgbm_search.best_params_}")
    log(f"    best CV Brier: {-lgbm_search.best_score_:.5f}")
    lgbm_best = lgbm_pipeline(**{k.replace("lgbm__", ""): v
                                  for k, v in lgbm_search.best_params_.items()})
    lgbm_res = _loso_eval(lgbm_best, X, y, s, "LGBM (tuned)")

    # 4. CatBoost
    log(f"\n  [4] CatBoost RandomizedSearchCV ({n_iter_cb} iter)")
    t0 = time.time()
    cb_search = _random_search_cb(X, y, s, n_iter=n_iter_cb)
    log(f"    search done in {time.time()-t0:.1f}s")
    log(f"    best params: {cb_search.best_params_}")
    log(f"    best CV Brier: {-cb_search.best_score_:.5f}")
    cb_best = catboost_pipeline(**{k.replace("cb__", ""): v
                                     for k, v in cb_search.best_params_.items()})
    cb_res = _loso_eval(cb_best, X, y, s, "CatBoost (tuned)")

    # 5. Blends
    log(f"\n  [5] Blend analyses")
    xgb_alpha, xgb_brier = _blend_with_lr(lr_res["oof_preds"], xgb_res["oof_preds"],
                                            y, "LR + XGB")
    lgbm_alpha, lgbm_brier = _blend_with_lr(lr_res["oof_preds"], lgbm_res["oof_preds"],
                                              y, "LR + LGBM")
    cb_alpha, cb_brier = _blend_with_lr(lr_res["oof_preds"], cb_res["oof_preds"],
                                          y, "LR + CatBoost")

    return {
        "gender": label,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "lr":   {"brier": lr_res["mean_brier"]},
        "xgb_depth_reg_grid": depth_reg_grid,
        "xgb":  {"brier": xgb_res["mean_brier"],
                 "best_params": xgb_search.best_params_,
                 "blend_alpha": xgb_alpha, "blend_brier": xgb_brier},
        "lgbm": {"brier": lgbm_res["mean_brier"],
                 "best_params": lgbm_search.best_params_,
                 "blend_alpha": lgbm_alpha, "blend_brier": lgbm_brier},
        "cb":   {"brier": cb_res["mean_brier"],
                 "best_params": cb_search.best_params_,
                 "blend_alpha": cb_alpha, "blend_brier": cb_brier},
    }


def main():
    # Men's
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned_features("men", "men_barttorvik.json", key="B_restricted")
    m_out = _run_gender("Men", m_feat, m_seeds, kl.load_m_tourney_compact,
                         MEN_SEASONS_BARTT, m_features, lr_C=0.03)

    # Women's
    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned_features("women")
    w_out = _run_gender("Women", w_feat, w_seeds, kl.load_w_tourney_compact,
                         WOMEN_SEASONS_FULL, w_features, lr_C=LR_C_WOMEN)

    # Save
    result = {"men": m_out, "women": w_out}
    with open(CV_REPORTS / "boosted_trees.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for g in ["men", "women"]:
        r = result[g]
        print(f"\n{g.upper()}:")
        print(f"  LR alone:         {r['lr']['brier']:.5f}")
        print(f"  XGB (tuned):      {r['xgb']['brier']:.5f}  "
              f"-- blend alpha={r['xgb']['blend_alpha']}, "
              f"blend Brier={r['xgb']['blend_brier']:.5f}")
        print(f"  LGBM (tuned):     {r['lgbm']['brier']:.5f}  "
              f"-- blend alpha={r['lgbm']['blend_alpha']}, "
              f"blend Brier={r['lgbm']['blend_brier']:.5f}")
        print(f"  CatBoost (tuned): {r['cb']['brier']:.5f}  "
              f"-- blend alpha={r['cb']['blend_alpha']}, "
              f"blend Brier={r['cb']['blend_brier']:.5f}")


if __name__ == "__main__":
    main()
