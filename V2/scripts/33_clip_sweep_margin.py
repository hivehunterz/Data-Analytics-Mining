"""
Stage 33: Sweep probability clipping on the Stage 32 margin+market submission.

The isotonic step already clips at [0.01, 0.99]. The question here is
whether pulling in the tails further (e.g. [0.03, 0.97], [0.05, 0.95])
improves 2026 Brier, as it did in the classic Kaggle tutorial, or hurts,
as happened in Stage 29 (chalky 2026).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import SUBMISSIONS, TRUTH_DIR
from src.utils import brier


def main():
    sub = pd.read_csv(SUBMISSIONS / "v2_margin_plus_market.csv")
    sub["Season"] = sub["ID"].str.split("_").str[0].astype(int)
    sub["T1"] = sub["ID"].str.split("_").str[1].astype(int)
    sub["T2"] = sub["ID"].str.split("_").str[2].astype(int)

    tm = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    tm["T1"] = tm[["WTeamID","LTeamID"]].min(axis=1)
    tm["T2"] = tm[["WTeamID","LTeamID"]].max(axis=1)
    tm["y"]  = (tm["WTeamID"] == tm["T1"]).astype(int)
    jm = tm.merge(sub[["T1","T2","Pred"]], on=["T1","T2"], how="inner")

    tw = pd.read_csv(TRUTH_DIR / "womens_2026_truth.csv")
    tw["T1"] = tw[["WTeamID","LTeamID"]].min(axis=1)
    tw["T2"] = tw[["WTeamID","LTeamID"]].max(axis=1)
    tw["y"]  = (tw["WTeamID"] == tw["T1"]).astype(int)
    jw = tw.merge(sub[["T1","T2","Pred"]], on=["T1","T2"], how="inner")

    # Prediction distribution
    print("Prediction distribution (scored games):")
    allp = np.concatenate([jm["Pred"].values, jw["Pred"].values])
    for q in [0.0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0]:
        print(f"  q{int(q*100):>3d}: {np.quantile(allp, q):.4f}")
    print(f"  n_very_low (<0.03):  {(allp < 0.03).sum()}")
    print(f"  n_very_high (>0.97): {(allp > 0.97).sum()}")
    print(f"  n_low (<0.05):       {(allp < 0.05).sum()}")
    print(f"  n_high (>0.95):      {(allp > 0.95).sum()}")
    print()

    clip_values = [
        (0.00, 1.00),   # no clipping
        (0.01, 0.99),   # isotonic built-in (baseline)
        (0.02, 0.98),
        (0.03, 0.97),
        (0.04, 0.96),
        (0.05, 0.95),
        (0.07, 0.93),
        (0.10, 0.90),
        (0.15, 0.85),
    ]

    print(f"  {'clip_lo':>8s} {'clip_hi':>8s} {'Men':>8s} {'Women':>8s} {'Combined':>10s}")
    print("  " + "-" * 48)
    best = (None, np.inf)
    for lo, hi in clip_values:
        pm = np.clip(jm["Pred"].values, lo, hi)
        pw = np.clip(jw["Pred"].values, lo, hi)
        mb = brier(jm["y"].values, pm)
        wb = brier(jw["y"].values, pw)
        cb = (mb*len(jm) + wb*len(jw)) / (len(jm) + len(jw))
        marker = "  <-- BEST" if cb < best[1] else ""
        if cb < best[1]:
            best = ((lo, hi), cb)
        print(f"  {lo:>8.2f} {hi:>8.2f} {mb:>8.5f} {wb:>8.5f} {cb:>10.5f}{marker}")

    print()
    print(f"  Best clip: {best[0]} -> combined Brier {best[1]:.5f}")
    print(f"  vs unclipped (no-clip baseline): {brier(jm['y'].values, jm['Pred'].values)*len(jm)/(len(jm)+len(jw)) + brier(jw['y'].values, jw['Pred'].values)*len(jw)/(len(jm)+len(jw)):.5f}")


if __name__ == "__main__":
    main()
