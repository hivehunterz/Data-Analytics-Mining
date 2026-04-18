"""
LR + XGBoost probability blend (Brendan 2nd-place pattern).

Each base learner is trained independently on its own feature pipeline
(LR on scaled features, XGBoost on raw features). The blend is a simple
weighted average of predicted probabilities, clipped to [lo, hi].

    blended = w_lr * lr_preds + w_xgb * xgb_preds
    blended = clip(blended, lo, hi)

Brendan used w_lr=0.40, w_xgb=0.60, clip [0.02, 0.98].
We parameterize so Kevin's finding (blend hurts) is testable too.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class LRxXGBBlend(BaseEstimator, ClassifierMixin):
    """
    Composite estimator that fits two pipelines and averages their probs.
    Compatible with sklearn CV / GroupKFold machinery.
    """
    def __init__(self, lr_pipeline, xgb_pipeline,
                 w_lr: float = 0.40, w_xgb: float = 0.60,
                 clip_lo: float = 0.02, clip_hi: float = 0.98):
        self.lr_pipeline  = lr_pipeline
        self.xgb_pipeline = xgb_pipeline
        self.w_lr  = w_lr
        self.w_xgb = w_xgb
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def fit(self, X, y):
        self._lr  = clone(self.lr_pipeline).fit(X, y)
        self._xgb = clone(self.xgb_pipeline).fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p_lr  = self._lr.predict_proba(X)[:, 1]
        p_xgb = self._xgb.predict_proba(X)[:, 1]
        blended = self.w_lr * p_lr + self.w_xgb * p_xgb
        blended = np.clip(blended, self.clip_lo, self.clip_hi)
        # Two-class output for sklearn compatibility
        return np.column_stack([1 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
