"""
Stage 7: Compute Colley, SRS, and GLM Quality ratings per (Season, TeamID).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import PROCESSED
from src.loaders import kaggle_loader as kl
from src.features.ratings_custom import compute_colley, compute_srs, compute_glm_quality
from src.utils import log


def _run(label: str, compact_loader, out_path: Path):
    log(f"{label}: loading compact results...")
    c = compact_loader()
    log(f"  rows: {len(c):,}, seasons: {c.Season.min()}-{c.Season.max()}")

    log(f"{label}: computing Colley...")
    colley = compute_colley(c)
    log(f"  rows: {len(colley)}")

    log(f"{label}: computing SRS...")
    srs = compute_srs(c)
    log(f"  rows: {len(srs)}")

    log(f"{label}: computing GLM Quality (slower)...")
    glm = compute_glm_quality(c)
    log(f"  rows: {len(glm)}")

    out = (
        colley
        .merge(srs, on=["Season", "TeamID"], how="outer")
        .merge(glm, on=["Season", "TeamID"], how="outer")
    )
    out.to_parquet(out_path, index=False)
    log(f"saved {out_path}")

    # Sanity: 2026 top 5 by each rating
    last = out[out.Season == 2026]
    print(f"\n  {label} 2026 top 5 Colley:")
    print(last.nlargest(5, "ColleyRating")[["TeamID", "ColleyRating", "SRS", "GLMQuality"]].round(3).to_string(index=False))
    print(f"\n  {label} 2026 top 5 SRS:")
    print(last.nlargest(5, "SRS")[["TeamID", "ColleyRating", "SRS", "GLMQuality"]].round(3).to_string(index=False))


def main():
    _run("Men's",   kl.load_m_regular_compact, PROCESSED / "m_ratings_custom.parquet")
    _run("Women's", kl.load_w_regular_compact, PROCESSED / "w_ratings_custom.parquet")


if __name__ == "__main__":
    main()
