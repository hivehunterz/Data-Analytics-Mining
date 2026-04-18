"""
Stage 2: Compute MOV-weighted Elo ratings with 0.75 season carry-over
for men and women. Output: parquet with (Season, TeamID, Elo, EloSlope).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED
from src.loaders import kaggle_loader as kl
from src.features.elo import compute_elo_ratings, compute_elo_slope
from src.utils import log


def _run(label: str, compact_loader, out_path: Path):
    log(f"{label}: loading compact results...")
    compact = compact_loader()
    log(f"  rows: {len(compact)}, seasons: {compact['Season'].min()}-{compact['Season'].max()}")

    log(f"{label}: computing end-of-season Elo...")
    elo = compute_elo_ratings(compact)
    log(f"  (Season, TeamID) rows: {len(elo)}")

    log(f"{label}: computing Elo slope...")
    slope = compute_elo_slope(compact)
    merged = elo.merge(slope, on=["Season", "TeamID"], how="left")
    merged["EloSlope"] = merged["EloSlope"].fillna(0.0)

    merged.to_parquet(out_path, index=False)
    log(f"saved {out_path}")

    # Sanity: top-5 by 2026 Elo
    last = merged[merged.Season == 2026].nlargest(5, "Elo")
    print(f"\n  {label} 2026 top 5 Elo:")
    print(last.round(2).to_string(index=False))
    return merged


def main():
    _run("Men's",   kl.load_m_regular_compact, PROCESSED / "m_elo.parquet")
    _run("Women's", kl.load_w_regular_compact, PROCESSED / "w_elo.parquet")


if __name__ == "__main__":
    main()
