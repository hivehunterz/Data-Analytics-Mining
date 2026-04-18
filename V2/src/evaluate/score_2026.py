"""
Local Brier scoring against 2026 ground truth.

Reads:
  - outputs/submissions/{tag}_submission.csv  (132,133 rows, ID, Pred)
  - data_raw/ground_truth/mens_2026_truth.csv  (Round, WTeamID, LTeamID)
  - data_raw/ground_truth/womens_2026_truth.csv (optional)

Outputs per-gender Brier + combined Brier.
"""
import pandas as pd
import numpy as np

from config import TRUTH_DIR, SUBMISSIONS, MEN_ID_CUTOFF
from src.utils import log, brier


def score_submission(submission_path):
    submission = pd.read_csv(submission_path)
    parts = submission["ID"].str.split("_", expand=True)
    submission["Season"] = parts[0].astype(int)
    submission["T1"]     = parts[1].astype(int)
    submission["T2"]     = parts[2].astype(int)

    # Load truths
    m_truth_path = TRUTH_DIR / "mens_2026_truth.csv"
    w_truth_path = TRUTH_DIR / "womens_2026_truth.csv"
    results = {}

    if m_truth_path.exists():
        m_truth = pd.read_csv(m_truth_path)
        m_brier = _score_gender(submission, m_truth, is_men=True)
        results["men"] = m_brier
    if w_truth_path.exists():
        w_truth = pd.read_csv(w_truth_path)
        w_brier = _score_gender(submission, w_truth, is_men=False)
        results["women"] = w_brier

    if "men" in results and "women" in results:
        # Combined: weighted mean by n_games
        m_n = results["men"]["n_games"]
        w_n = results["women"]["n_games"]
        combined = (results["men"]["brier"] * m_n + results["women"]["brier"] * w_n) / (m_n + w_n)
        results["combined"] = {"brier": combined, "n_games": m_n + w_n}

    return results


def _score_gender(submission: pd.DataFrame, truth: pd.DataFrame, is_men: bool):
    # Filter submission to this gender
    if is_men:
        sub = submission[submission["T1"] < MEN_ID_CUTOFF]
    else:
        sub = submission[submission["T1"] >= MEN_ID_CUTOFF]

    # Canonicalize truth: T1 = min(W, L), T2 = max(W, L), y = 1 if T1 won
    t = truth.copy()
    t["T1"] = t[["WTeamID", "LTeamID"]].min(axis=1)
    t["T2"] = t[["WTeamID", "LTeamID"]].max(axis=1)
    t["y"]  = (t["WTeamID"] == t["T1"]).astype(int)

    merged = t.merge(sub, on=["T1", "T2"], how="inner")
    if len(merged) == 0:
        return {"brier": None, "n_games": 0, "note": "no matching rows"}
    b = brier(merged["y"].values, merged["Pred"].values)
    return {"brier": float(b), "n_games": int(len(merged))}


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    path = SUBMISSIONS / f"{tag}_submission.csv"
    if not path.exists():
        raise SystemExit(f"No submission at {path}")
    results = score_submission(path)
    for k, v in results.items():
        if v.get("brier") is not None:
            log(f"{k}: Brier = {v['brier']:.5f}  (n={v['n_games']})")
        else:
            log(f"{k}: {v}")
