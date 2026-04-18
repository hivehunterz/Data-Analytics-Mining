"""
Backward elimination feature pruning.

Kevin's biggest single-round CV gain came from this: drop one feature at a
time, keep if LOSO mean Brier drops by at least `threshold` (default 0.0005).
Stop when no remaining feature yields an improvement.

Algorithm:
    current_features = all
    loop:
        baseline_brier = cv(current_features)
        best_drop = None
        for f in current_features:
            trial_brier = cv(current_features - {f})
            if baseline_brier - trial_brier >= threshold:
                best_drop = (f, trial_brier)
        if best_drop: drop it; else: stop.
"""
import numpy as np
import pandas as pd

from sklearn.base import clone
from src.validation.loso_cv import loso_cv_brier
from src.utils import log


def backward_elimination(
    model_factory,
    df_X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    threshold: float = 0.0005,
    max_iters: int = 20,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    model_factory : callable -> sklearn estimator
        Called fresh each evaluation (so state is not shared).
    df_X : pd.DataFrame
        Feature matrix. Columns are feature names.
    y : np.ndarray
    seasons : np.ndarray
    threshold : float
        Minimum Brier improvement to accept a drop.
    max_iters : int
        Hard cap on iterations.

    Returns
    -------
    dict with:
        - final_features : list of kept features
        - history : list of (iter, dropped_feature, new_brier)
        - final_brier : float
    """
    current = list(df_X.columns)
    X_full = df_X.values
    colidx = {c: i for i, c in enumerate(current)}

    def _cv(feat_list):
        idx = [colidx[c] for c in feat_list]
        X = X_full[:, idx]
        model = model_factory()
        r = loso_cv_brier(model, X, y, seasons, return_oof=False)
        return r["mean_brier"]

    history = []
    baseline = _cv(current)
    if verbose:
        log(f"baseline Brier with {len(current)} features: {baseline:.5f}")

    for it in range(max_iters):
        best_feature, best_brier = None, baseline
        for f in current:
            trial_feats = [c for c in current if c != f]
            if not trial_feats:
                continue
            b = _cv(trial_feats)
            if baseline - b >= threshold and b < best_brier:
                best_feature, best_brier = f, b
        if best_feature is None:
            if verbose:
                log(f"no further drops improve Brier. Stopping at {len(current)} features.")
            break
        current.remove(best_feature)
        if verbose:
            log(f"  iter {it+1}: drop {best_feature!r}  "
                f"Brier {baseline:.5f} -> {best_brier:.5f}  "
                f"({len(current)} features left)")
        history.append({
            "iter": it + 1, "dropped": best_feature,
            "new_brier": best_brier, "n_features": len(current),
        })
        baseline = best_brier

    return {
        "final_features": current,
        "history": history,
        "final_brier": baseline,
    }
