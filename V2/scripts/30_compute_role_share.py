"""
Stage 30: Principled role_share computation for injury Elo penalty.

FORMULA:
    role_share = mpg/200 + 0.75 * ppg/team_ppg
    clip to [0.05, 0.45]

RATIONALE:
  - A team plays 5 positions * 40 minutes = 200 team-minutes per game,
    so mpg/200 is the share of court-minutes the player occupies.
  - ppg/team_ppg is the share of team scoring. Weighted 0.75 because
    scoring is the dominant observable but rebounding / defence /
    playmaking matter too.
  - Floor 0.05: every listed player contributes SOMETHING; even a 12-min
    bench player has some role.
  - Cap 0.45: no single player is more than ~45% of a team. Anthony
    Davis-level contributions (>45%) are rare and would saturate the
    penalty in the 100-point Elo map anyway.

EXCLUSION RULES applied when curating injury_report_v3:
  (a) Multi-month absences already baked into team Elo -> drop
      Players on the team for <5 games of the last 20 regular-season games.
      (Examples: Kentucky Lowe, Quaintance; Gonzaga Huff)
  (b) Confirmed healthy at tipoff via pre-tournament beat reporting -> drop
      (Examples: Kansas Peterson, Purdue Kaufman-Renn)
  (c) In-tournament injury (occurred after R1 tipoff) -> excluded
      (Example: Iowa State's Joshua Jefferson, ankle vs Tennessee State)

This script:
  1. Verifies each row's role_share matches the formula.
  2. Computes an independent role_share from the (mpg, ppg, team_ppg) cols.
  3. Flags any discrepancies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import INJURIES_DIR
from src.utils import log


def compute_role_share(mpg: float, ppg: float, team_ppg: float,
                        floor: float = 0.05, cap: float = 0.45) -> float:
    """Principled role_share formula."""
    minutes_share = mpg / 200.0
    scoring_share = ppg / max(team_ppg, 1.0)
    raw = minutes_share + 0.75 * scoring_share
    return float(np.clip(raw, floor, cap))


def main():
    path = INJURIES_DIR / "injury_report_v3.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["player"])   # drop any empty rows
    log(f"loaded {len(df)} injury rows from {path.name}")

    print(f"\n  {'Player':<22} {'Team':<18} {'MPG':>5s} {'PPG':>5s} "
          f"{'T.PPG':>6s} {'stored':>7s} {'formula':>8s} {'diff':>6s}")
    print("  " + "-" * 80)

    mismatch = []
    for _, row in df.iterrows():
        stored = row["role_share"]
        formula = compute_role_share(row["mpg"], row["ppg"], row["team_ppg"])
        diff = abs(stored - formula)
        print(f"  {row['player']:<22} {row['team']:<18} "
              f"{row['mpg']:>5.1f} {row['ppg']:>5.1f} {row['team_ppg']:>6.1f}  "
              f"{stored:>7.3f} {formula:>8.3f} {diff:>+6.3f}")
        if diff > 0.01:
            mismatch.append((row["player"], stored, formula))

    if mismatch:
        log(f"WARNING: {len(mismatch)} rows differ from formula by > 0.01")
        for m in mismatch:
            log(f"  {m[0]}: stored={m[1]:.3f}, formula={m[2]:.3f}")
    else:
        log("All rows match the formula within 0.01.")

    # Verify Elo penalties
    status_w = {"OUT_FOR_SEASON": 1.0, "OUT": 0.75, "GAME_TIME": 0.50}
    df["sw"] = df["status"].map(status_w)
    df["penalty"] = df["sw"] * df["role_share"] * 100

    print(f"\n  Elo penalty breakdown per team:")
    team_pen = df.groupby("team")["penalty"].sum().sort_values(ascending=False)
    for team, pen in team_pen.items():
        n = (df["team"] == team).sum()
        print(f"    {team:<20s} {pen:>6.1f} Elo (from {n} player(s))")


if __name__ == "__main__":
    main()
