"""
Shared utilities: logging, path helpers, Brier metric.
"""
import json
import time
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def log(msg: str) -> None:
    """Timestamped stdout log line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_cv_report(tag: str, payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"saved CV report -> {path}")


def brier(y_true: Iterable, y_pred: Iterable) -> float:
    """Standard Brier score; lower is better. Matches the Kaggle metric."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_pred)))


def safe_log_loss(y_true: Iterable, y_pred: Iterable, eps: float = 1e-15) -> float:
    y = np.asarray(y_pred).clip(eps, 1 - eps)
    return float(log_loss(np.asarray(y_true), y))


def is_men(team_id: int, cutoff: int = 3000) -> bool:
    return team_id < cutoff


def df_from_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the upstream stage first.")
    return pd.read_parquet(path)
