"""
Stage 18: Load all Barttorvik seasons, attach Kaggle TeamIDs,
save as parquet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED
from src.loaders.barttorvik_loader import load_all, attach_teamid
from src.utils import log


def main():
    seasons = list(range(2008, 2027))

    log("loading all Barttorvik seasons (men)...")
    bart = load_all(seasons)
    log(f"rows: {len(bart)}, cols: {list(bart.columns)}")

    log("attaching Kaggle TeamIDs via spellings + fuzzy match...")
    # Barttorvik only covers men's D1; all teams should map to men's TeamIDs
    mapped = attach_teamid(bart, is_men=True)
    log(f"after mapping: {len(mapped)} rows")

    out = PROCESSED / "m_barttorvik.parquet"
    mapped.to_parquet(out, index=False)
    log(f"saved {out}")

    # Sanity check
    print("\n  2026 top 5 by barthag:")
    print(mapped[mapped.Season == 2026].nlargest(5, "barthag")[
        ["TeamID", "team_bart", "adjoe", "adjde", "barthag", "WAB"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
