"""
Stage 5: Deep Run Prediction — Player-Only vs Combined.
Models: Random Forest, XGBoost, LightGBM, CatBoost + RandomizedSearchCV.
Reports ROC-AUC, Brier score, log-loss, and calibration.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False

from config import (PLAYER_ONLY_COLS, TEAM_COLS, COMBINED_COLS,
                    BPM_ONLY_COLS, RANDOM_STATE, CV_FOLDS,
                    HOLDOUT_YEAR, OUTPUT_DIR)
from utils import log_step, save_fig


def _build_search_spaces():
    """Hyperparameter search spaces for RandomizedSearchCV."""
    spaces = {}

    spaces["RF"] = {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "n_estimators": [200, 300, 500],
            "max_depth": [8, 10, 12, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.3],
        },
    }

    if HAS_XGB:
        spaces["XGBoost"] = {
            "model": XGBClassifier(
                eval_metric="logloss", use_label_encoder=False,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1, 2, 5],
            },
        }

    if HAS_LGBM:
        spaces["LightGBM"] = {
            "model": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [15, 31, 63, 127],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [0, 1, 5],
            },
        }

    if HAS_CB:
        spaces["CatBoost"] = {
            "model": CatBoostClassifier(
                random_state=RANDOM_STATE, verbose=0, thread_count=-1),
            "params": {
                "iterations": [200, 300, 500],
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "l2_leaf_reg": [1, 3, 5, 7],
                "bagging_temperature": [0, 0.5, 1],
            },
        }

    return spaces


def run_classification(df):
    """Stage 5: Cross-validated deep-run prediction with tuned models."""
    df_train = df[df["season_year"] < HOLDOUT_YEAR].copy()
    y_train  = df_train["deep_run"].values

    # Fill NaN in feature columns
    for col in COMBINED_COLS:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            df_train[col] = df_train[col].fillna(med)

    log_step(f"Training on 2011-{HOLDOUT_YEAR-1}: {len(df_train)} rows")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    spaces = _build_search_spaces()

    # ── Phase 1: Player-only (RF baseline, no tuning needed) ──────────
    print(f"\n  === PLAYER-ONLY MODELS (37 features) ===")
    print(f"  {'Model':<20s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s}")
    print("  " + "-" * 48)

    rf_player = RandomForestClassifier(n_estimators=300, max_depth=12,
                                        random_state=RANDOM_STATE, n_jobs=-1)
    X_po = df_train[PLAYER_ONLY_COLS].values
    probs_po = cross_val_predict(rf_player, X_po, y_train, cv=cv, method="predict_proba")[:,1]
    auc_po = roc_auc_score(y_train, probs_po)
    brier_po = brier_score_loss(y_train, probs_po)
    ll_po = log_loss(y_train, probs_po)
    print(f"  {'RF (player-only)':<20s} {auc_po:>8.4f} {brier_po:>8.4f} {ll_po:>9.4f}")

    # ── Phase 2: Combined with hyperparameter tuning ──────────────────
    print(f"\n  === COMBINED MODELS (41 features) — with RandomizedSearchCV ===")
    print(f"  {'Model':<20s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s} {'Best Params'}")
    print("  " + "-" * 90)

    X_comb = df_train[COMBINED_COLS].values
    best_models = {}
    results_table = []

    for name, spec in spaces.items():
        print(f"\n    [*] Tuning {name} (30 iterations x {CV_FOLDS}-fold CV)...", flush=True)
        search = RandomizedSearchCV(
            spec["model"], spec["params"], n_iter=30, cv=cv,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
            refit=True, error_score="raise",
        )
        search.fit(X_comb, y_train)
        print(f"    [*] {name} tuning done. Running cross-val predict...", flush=True)
        best_m = search.best_estimator_
        probs = cross_val_predict(best_m, X_comb, y_train, cv=cv, method="predict_proba")[:,1]
        auc = roc_auc_score(y_train, probs)
        brier = brier_score_loss(y_train, probs)
        ll = log_loss(y_train, probs)
        short_params = {k: v for k, v in search.best_params_.items()
                        if k in ("n_estimators","max_depth","learning_rate","num_leaves","depth","iterations")}
        print(f"  {name:<20s} {auc:>8.4f} {brier:>8.4f} {ll:>9.4f} {short_params}")
        best_models[name] = best_m
        results_table.append({"model": name, "auc": auc, "brier": brier, "log_loss": ll,
                              "best_params": search.best_params_})

    # ── Phase 3: Win Shares Ablation ──────────────────────────────────
    print(f"\n  === WIN SHARES ABLATION (BPM-only vs full) ===", flush=True)
    # Pick the best available model for ablation
    ablation_name = "XGBoost" if "XGBoost" in best_models else "RF"
    ab_base = best_models[ablation_name].__class__(**best_models[ablation_name].get_params())

    # Full features
    print(f"    [*] Running ablation with {ablation_name} (full features)...", flush=True)
    probs_full = cross_val_predict(ab_base, X_comb, y_train, cv=cv, method="predict_proba")[:,1]
    auc_full = roc_auc_score(y_train, probs_full)

    # Without Win Shares
    print(f"    [*] Running ablation without Win Shares...", flush=True)
    bpm_cols = BPM_ONLY_COLS + TEAM_COLS
    X_bpm = df_train[bpm_cols].values
    ab_base2 = best_models[ablation_name].__class__(**best_models[ablation_name].get_params())
    probs_bpm = cross_val_predict(ab_base2, X_bpm, y_train, cv=cv, method="predict_proba")[:,1]
    auc_bpm = roc_auc_score(y_train, probs_bpm)

    print(f"  With WS features:     AUC = {auc_full:.4f}  ({len(COMBINED_COLS)} features)")
    print(f"  Without WS features:  AUC = {auc_bpm:.4f}  ({len(bpm_cols)} features)")
    print(f"  Delta:                {auc_full - auc_bpm:+.4f}")

    # ── Feature importance from best RF ───────────────────────────────
    rf_final = best_models.get("RF", list(best_models.values())[0])
    if not hasattr(rf_final, "feature_importances_"):
        rf_final = RandomForestClassifier(n_estimators=300, max_depth=12,
                                          random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_comb, y_train)
    importances = pd.Series(rf_final.feature_importances_, index=COMBINED_COLS).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    top20 = importances.head(20).sort_values()
    colors_fi = ["#e74c3c" if f in TEAM_COLS else "#3498db" for f in top20.index]
    ax.barh(range(len(top20)), top20.values, color=colors_fi)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index)
    ax.set_title("Top 20 Features (Red=Team, Blue=Player)", fontsize=13)
    ax.set_xlabel("Feature Importance (Gini)")
    save_fig("stage5_feature_importance")

    # Player-only importance
    rf_player.fit(X_po, y_train)
    imp_player = pd.Series(rf_player.feature_importances_, index=PLAYER_ONLY_COLS).sort_values(ascending=False)
    print(f"\n  Top 15 PLAYER-ONLY features:")
    for feat, imp in imp_player.head(15).items():
        print(f"    {feat:<30s} {imp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    imp_player.head(15).sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("viridis", 15))
    ax.set_title("Top 15 Player-Only Features", fontsize=13)
    ax.set_xlabel("Feature Importance (Gini)")
    save_fig("stage5_player_only_importance")

    # ── Calibration plot for Stage 5 ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0,1],[0,1],"--",color="gray",alpha=0.7,label="Perfect")
    for name, model in best_models.items():
        probs_cv = cross_val_predict(model, X_comb, y_train, cv=cv, method="predict_proba")[:,1]
        fraction_pos, mean_pred = calibration_curve(y_train, probs_cv, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, fraction_pos, "o-", label=f"{name} (Brier={brier_score_loss(y_train, probs_cv):.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Deep Run Classifier — Calibration (5-fold CV)")
    ax.legend(loc="lower right")
    save_fig("stage5_calibration")

    return best_models, results_table
