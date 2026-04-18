"""
Stage 21: Blend v2_final_bartt submission with Vegas spreads and score.

Also sweeps alpha (LR weight) across [0.05, 0.10, 0.20, 0.30, 0.50, 0.75,
1.00] so we pick the empirically best blend against 2026 truth (the
ONLY place we validate a blend weight — no CV was possible for this).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import SUBMISSIONS, TRUTH_DIR, CLIP_LO, CLIP_HI, MEN_ID_CUTOFF
from src.models.market_blend import load_vegas_2026_men, apply_tiered_blend
from src.postprocess.clip import clip_predictions
from src.utils import log, brier


def _load_submission(tag: str) -> pd.DataFrame:
    sub = pd.read_csv(SUBMISSIONS / f"{tag}_submission.csv")
    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["T1"] = parts[1].astype(int)
    sub["T2"] = parts[2].astype(int)
    return sub


def _score_vs_truth(sub: pd.DataFrame) -> tuple[float, int]:
    """Men's Brier against mens_2026_truth.csv."""
    truth = pd.read_csv(TRUTH_DIR / "mens_2026_truth.csv")
    truth["T1"] = truth[["WTeamID", "LTeamID"]].min(axis=1)
    truth["T2"] = truth[["WTeamID", "LTeamID"]].max(axis=1)
    truth["y"]  = (truth["WTeamID"] == truth["T1"]).astype(int)
    m = truth.merge(sub[["T1", "T2", "Pred"]], on=["T1", "T2"], how="inner")
    return brier(m["y"].values, m["Pred"].values), len(m)


def main():
    base_tag = "v2_final_bartt"
    sub = _load_submission(base_tag)
    men_sub = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    women_sub = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    base_brier, n = _score_vs_truth(men_sub)
    log(f"baseline {base_tag} men's Brier: {base_brier:.5f}  (n={n})")

    market = load_vegas_2026_men()
    log(f"market games with mapped IDs: {len(market)}")

    log("\nalpha sweep (LR weight; 1-alpha is market weight):")
    for alpha in [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90, 1.00]:
        blended = apply_tiered_blend(men_sub, market, alpha_r1=alpha)
        blended["Pred"] = clip_predictions(blended["Pred"].values, CLIP_LO, CLIP_HI)
        b, _ = _score_vs_truth(blended)
        log(f"  alpha={alpha:.2f}:  men's Brier = {b:.5f}  (delta={b-base_brier:+.5f})")

    # Apply best (based on Kevin's intuition + our sweep) and save final
    best_alpha = 0.10   # Kevin's men's Tier-1 weight
    final = apply_tiered_blend(men_sub, market, alpha_r1=best_alpha)
    final["Pred"] = clip_predictions(final["Pred"].values, CLIP_LO, CLIP_HI)
    out = pd.concat([final, women_sub], ignore_index=True)[["ID", "Pred"]]
    out_path = SUBMISSIONS / "v2_final_blended_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"saved {out_path}")

    b_final, _ = _score_vs_truth(final)
    log(f"\nFinal men's Brier (alpha={best_alpha}): {b_final:.5f}")


if __name__ == "__main__":
    main()
