"""
Shallow XGBoost classifier — Harry's 1st-place configuration.

Harry's hyperparameters: depth=2, lr=0.003, n_estimators=4000 (with early
stopping in his setup; we use 1000 here since we don't have an eval set
by default in the LOSO loop).

Trees are fed RAW features (no scaling) — scaling only helps LR. This is
a Brendan finding: separate pipelines for LR (scaled) vs XGB (raw).
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


def make_xgb_pipeline(
    max_depth: int = 2,
    learning_rate: float = 0.05,
    n_estimators: int = 500,
    subsample: float = 0.7,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Harry-style shallow XGBoost. Defaults closer to Harry's config but with
    faster lr/n_estimators combo so LOSO CV completes in reasonable time.

    For the final submission pass, we can crank up n_estimators=4000 and
    lr=0.003 (Harry's exact settings).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )),
    ])
