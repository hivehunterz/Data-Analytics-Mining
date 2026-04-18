"""
Logistic Regression training with StandardScaler + median imputation.

Kevin's architecture, verbatim:
    SimpleImputer(strategy="median")
    -> StandardScaler()
    -> LogisticRegression(solver="lbfgs", max_iter=1000, C=<gender-specific>)
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def make_lr_pipeline(C: float, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            solver="lbfgs", max_iter=1000, C=C, random_state=random_state,
        )),
    ])
