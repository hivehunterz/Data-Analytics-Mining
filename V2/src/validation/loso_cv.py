"""
Leave-One-Season-Out cross-validation harness.

Uses GroupKFold(n_splits=n_seasons) grouped by Season. This is the CV
strategy used by Harry and Kevin. Scoring: neg_brier_score.

Also provides a helper to produce out-of-fold predictions for calibration.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.base import clone


def loso_cv_brier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    return_oof: bool = True,
):
    """
    Leave-one-season-out CV evaluating Brier score.

    Parameters
    ----------
    model : sklearn estimator with fit / predict_proba
    X : feature matrix (n_samples, n_features)
    y : binary labels (n_samples,)
    seasons : season identifier per row (n_samples,)

    Returns
    -------
    dict with:
        - fold_scores : per-season Brier
        - mean_brier, std_brier
        - mean_logloss
        - oof_preds (if return_oof)
    """
    unique_seasons = np.unique(seasons)
    n_splits = len(unique_seasons)
    if n_splits < 2:
        raise ValueError(f"Need >=2 seasons for LOSO CV, got {n_splits}")
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores, fold_logloss = [], []
    oof = np.full_like(y, np.nan, dtype=float)
    per_season = {}

    for train_idx, test_idx in gkf.split(X, y, groups=seasons):
        est = clone(model)
        est.fit(X[train_idx], y[train_idx])
        preds = est.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = preds
        season_held = seasons[test_idx][0]
        b = brier_score_loss(y[test_idx], preds)
        ll = log_loss(y[test_idx], np.clip(preds, 1e-6, 1 - 1e-6))
        fold_scores.append(b)
        fold_logloss.append(ll)
        per_season[int(season_held)] = {
            "brier": float(b), "log_loss": float(ll), "n": int(len(test_idx))
        }

    result = {
        "mean_brier": float(np.mean(fold_scores)),
        "std_brier": float(np.std(fold_scores)),
        "mean_logloss": float(np.mean(fold_logloss)),
        "per_season": per_season,
    }
    if return_oof:
        result["oof_preds"] = oof
    return result
