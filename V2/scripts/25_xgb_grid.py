"""
Stage 25: Manual grid search for XGBoost (and LightGBM / CatBoost).

Uses our working loso_cv_brier (no make_scorer issues). Explores:
  - max_depth: 2, 3, 4, 5, 6
  - learning_rate: 0.01, 0.03, 0.05, 0.1
  - n_estimators: 200, 500, 1000
  - reg_lambda: 0.5, 1, 3, 10
  - subsample: 0.7, 0.85
  - colsample_bytree: 0.6, 0.8

That's a huge grid; we do staged: first depth x lr, then best + reg sweep,
then best + subsample/colsample.

Also tests LR+XGB blend at every stage.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (PROCESSED, CV_REPORTS, MEN_SEASONS_BARTT,
                     WOMEN_SEASONS_FULL, LR_C_WOMEN, RANDOM_STATE, CLIP_LO, CLIP_HI)
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
        ("xgb", XGBClassifier(
            eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, **kw)),
    ])


def lgbm_pipe(**kw):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lgbm", LGBMClassifier(
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
            objective="binary", **kw)),
    ])


def _eval(pipe, X, y, s, label):
    res = loso_cv_brier(pipe, X, y, s, return_oof=True)
    return res


def _run_gender(label, feat, seeds, tourney_loader, seasons, features, lr_C):
    log(f"\n{'='*60}\n  {label}\n{'='*60}")
    X, y, s = _prep(feat, seeds, tourney_loader, seasons, features)
    log(f"  rows: {X.shape[0]}, features: {X.shape[1]}, seasons: {len(np.unique(s))}")

    # LR baseline
    lr_res = _eval(make_lr_pipeline(C=lr_C), X, y, s, "LR")
    log(f"  LR baseline: {lr_res['mean_brier']:.5f}")

    # XGB grid: depth x lr x reg_lambda
    log(f"\n  === XGB GRID ===")
    log(f"  fixed: n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=5")
    log(f"  {'depth':>5s} {'lr':>6s} {'reg_lam':>8s} {'reg_alpha':>9s}  {'Brier':>8s}")
    results = []
    grid = [(d, lr_e, rlam, ralp)
            for d in [2, 3, 4, 5, 6]
            for lr_e in [0.01, 0.03, 0.05, 0.1]
            for rlam in [0.5, 1.0, 3.0, 10.0]
            for ralp in [0.0, 0.1]]
    log(f"  total combinations: {len(grid)}")
    best_xgb, best_brier, best_params = None, 1e9, None
    t0 = time.time()
    for (d, lr_e, rlam, ralp) in grid:
        pipe = xgb_pipe(
            max_depth=d, learning_rate=lr_e, reg_lambda=rlam, reg_alpha=ralp,
            n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        )
        r = _eval(pipe, X, y, s, f"d{d}lr{lr_e}rlam{rlam}ra{ralp}")
        results.append({
            "depth": d, "lr": lr_e, "reg_lambda": rlam, "reg_alpha": ralp,
            "brier": r["mean_brier"],
        })
        if r["mean_brier"] < best_brier:
            best_brier = r["mean_brier"]
            best_params = dict(max_depth=d, learning_rate=lr_e,
                                 reg_lambda=rlam, reg_alpha=ralp,
                                 n_estimators=500, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=5)
            best_xgb = pipe
            log(f"  * NEW BEST: d{d} lr{lr_e} rlam{rlam} ra{ralp}: {r['mean_brier']:.5f}")
    log(f"  grid done in {time.time()-t0:.0f}s. Best XGB Brier: {best_brier:.5f}")
    log(f"  Best params: {best_params}")

    # Print top-10
    results_sorted = sorted(results, key=lambda r: r["brier"])[:10]
    log(f"\n  top 10 XGB configs:")
    for r in results_sorted:
        log(f"    d{r['depth']} lr{r['lr']:.2f} rlam{r['reg_lambda']:.1f} ra{r['reg_alpha']:.1f}: {r['brier']:.5f}")

    # LightGBM short grid (focused)
    log(f"\n  === LIGHTGBM GRID ===")
    log(f"  {'depth':>5s} {'lr':>6s} {'leaves':>7s} {'reg_lam':>8s}  {'Brier':>8s}")
    lgbm_grid = [(d, lr_e, leaves, rlam)
                  for d in [3, 5, 7]
                  for lr_e in [0.01, 0.05]
                  for leaves in [15, 31]
                  for rlam in [0.5, 1.0, 5.0]]
    best_lgbm_brier = 1e9
    best_lgbm_params = None
    best_lgbm = None
    for (d, lr_e, leaves, rlam) in lgbm_grid:
        pipe = lgbm_pipe(
            max_depth=d, learning_rate=lr_e, num_leaves=leaves,
            n_estimators=500, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=rlam, reg_alpha=0.0, min_child_samples=10,
        )
        r = _eval(pipe, X, y, s, f"d{d}lr{lr_e}leaves{leaves}rlam{rlam}")
        if r["mean_brier"] < best_lgbm_brier:
            best_lgbm_brier = r["mean_brier"]
            best_lgbm_params = dict(max_depth=d, learning_rate=lr_e,
                                     num_leaves=leaves, reg_lambda=rlam)
            best_lgbm = pipe
    log(f"  Best LGBM Brier: {best_lgbm_brier:.5f}  params: {best_lgbm_params}")

    # Blend LR + best XGB
    log(f"\n  === LR + XGB BLEND SWEEP ===")
    # Re-run best XGB to get OOF
    best_xgb_res = _eval(xgb_pipe(**best_params), X, y, s, "best XGB")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blend = alpha * lr_res["oof_preds"] + (1 - alpha) * best_xgb_res["oof_preds"]
        blend = np.clip(blend, CLIP_LO, CLIP_HI)
        b = brier(y, blend)
        log(f"  alpha={alpha:.2f} (alpha=LR weight): Brier={b:.5f}")

    return {
        "lr_brier": lr_res["mean_brier"],
        "best_xgb_brier": best_brier,
        "best_xgb_params": best_params,
        "best_lgbm_brier": best_lgbm_brier,
        "best_lgbm_params": best_lgbm_params,
        "xgb_top10": results_sorted,
    }


def main():
    m_feat, m_seeds = _assemble_men()
    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    m_out = _run_gender("Men", m_feat, m_seeds, kl.load_m_tourney_compact,
                         MEN_SEASONS_BARTT, m_features, lr_C=0.03)

    w_feat, w_seeds = _assemble_women()
    w_features = _load_pruned("women")
    w_out = _run_gender("Women", w_feat, w_seeds, kl.load_w_tourney_compact,
                         WOMEN_SEASONS_FULL, w_features, lr_C=LR_C_WOMEN)

    with open(CV_REPORTS / "xgb_grid.json", "w") as f:
        json.dump({"men": m_out, "women": w_out}, f, indent=2, default=str)

    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for g, r in [("Men", m_out), ("Women", w_out)]:
        print(f"\n{g}:")
        print(f"  LR  : {r['lr_brier']:.5f}")
        print(f"  XGB : {r['best_xgb_brier']:.5f}  params={r['best_xgb_params']}")
        print(f"  LGBM: {r['best_lgbm_brier']:.5f}  params={r['best_lgbm_params']}")


if __name__ == "__main__":
    main()
