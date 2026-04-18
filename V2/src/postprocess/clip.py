"""
Brier-aware prediction clipping.

User choice: [0.03, 0.97]. Penalizes confident-and-wrong predictions less
than an uncapped model would.
"""
import numpy as np


def clip_predictions(probs: np.ndarray, lo: float = 0.03, hi: float = 0.97) -> np.ndarray:
    return np.clip(probs, lo, hi)
