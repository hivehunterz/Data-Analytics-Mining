"""
Stage 12: Score the v2_final submission against actual 2026 ground truth.
Men's truth was reconstructed from V1/data/mm2026_updated.csv.
Women's truth requires manual entry (see data_raw/ground_truth/womens_2026_truth.csv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SUBMISSIONS
from src.evaluate.score_2026 import score_submission
from src.utils import log


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_final"
    path = SUBMISSIONS / f"{tag}_submission.csv"
    if not path.exists():
        raise SystemExit(f"No submission at {path}")

    log(f"scoring {path.name}...")
    results = score_submission(path)
    print()
    for k, v in results.items():
        if v.get("brier") is not None:
            print(f"  {k:10s}: Brier = {v['brier']:.5f}  (n={v['n_games']})")
        else:
            print(f"  {k:10s}: {v}")

    # Context
    print(f"\n  vs. leaderboard:")
    print(f"    Harry   (1st): 0.10975")
    print(f"    Brendan (2nd): 0.11499")
    print(f"    Kevin   (3rd): 0.11604")


if __name__ == "__main__":
    main()
