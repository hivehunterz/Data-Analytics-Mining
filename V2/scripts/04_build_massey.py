"""
Stage 4: Build Massey ordinal aggregation for men (women have no Massey).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import PROCESSED, MEN_SEASONS_FULL
from src.loaders import kaggle_loader as kl
from src.features.massey_agg import find_eligible_systems, aggregate_massey
from src.utils import log


def main():
    log("loading MMasseyOrdinals.csv (large file)...")
    m = kl.load_m_massey()
    log(f"  rows: {len(m):,}, seasons: {m.Season.min()}-{m.Season.max()}")

    target_seasons = MEN_SEASONS_FULL + [2026]
    # Kevin's dynamic selection used ~28 systems. Our 95% threshold gave 10.
    # Loosen to 80% to match Kevin's count — he found more systems beats fewer.
    log(f"selecting systems with >=80% coverage across {len(target_seasons)} seasons...")
    eligible = find_eligible_systems(m, target_seasons, coverage_threshold=0.80)
    log(f"  eligible systems: {len(eligible)}")
    print(f"  examples: {eligible[:10]}")

    log("aggregating (mean / median / min of last-2-week ranks)...")
    agg = aggregate_massey(m, seasons=target_seasons, systems=eligible)
    log(f"  (Season, TeamID) rows: {len(agg)}")

    out = PROCESSED / "m_massey_agg.parquet"
    agg.to_parquet(out, index=False)
    log(f"saved {out}")

    # Sanity: top-10 2026 men by Massey mean (lowest = best)
    print("\n  Men's 2026 top 10 by Massey consensus:")
    print(agg[agg.Season == 2026].nsmallest(10, "MasseyMean").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
