"""
Shared utilities: logging, plotting, metrics.
"""
import sys, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss

from config import OUTPUT_DIR

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DualWriter:
    """Write to both terminal and log file, flushing after every write."""
    def __init__(self, filepath):
        self.terminal = sys.__stdout__          # always the real terminal
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()                   # flush immediately so you see output live
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def log_stage(n, total, title):
    print(f"\n{'='*70}")
    print(f"  STAGE {n}/{total}: {title}")
    print(f"{'='*70}")
    print(f"  [{time.strftime('%H:%M:%S')}]")


def log_step(msg):
    print(f"    >> {msg}")


def save_fig(name):
    path = f"{OUTPUT_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_step(f"[saved] {path}")


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
    return 0.0 if pooled == 0 else (g1.mean() - g2.mean()) / pooled


def probability_metrics(y_true, y_prob):
    """Return Brier score and log-loss."""
    return {
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss":    log_loss(y_true, y_prob),
    }
