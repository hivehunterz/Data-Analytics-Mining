"""
Stage 1: Build per-team-season feature tables from detailed regular season
results for men (2003+) and women (2010+). Output: parquet files in
data_processed/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROCESSED
from src.loaders import kaggle_loader as kl
from src.features.efficiency import build_team_season
from src.utils import log


def main():
    log("loading men's detailed results...")
    m_det = kl.load_m_regular_detailed()
    log(f"  rows: {len(m_det)}, seasons: {m_det['Season'].min()}-{m_det['Season'].max()}")

    log("aggregating men's team-seasons...")
    m_ts = build_team_season(m_det)
    log(f"  men team-seasons: {len(m_ts)}")
    out = PROCESSED / "m_team_season.parquet"
    m_ts.to_parquet(out, index=False)
    log(f"saved {out}")

    log("loading women's detailed results...")
    w_det = kl.load_w_regular_detailed()
    log(f"  rows: {len(w_det)}, seasons: {w_det['Season'].min()}-{w_det['Season'].max()}")

    log("aggregating women's team-seasons...")
    w_ts = build_team_season(w_det)
    log(f"  women team-seasons: {len(w_ts)}")
    out = PROCESSED / "w_team_season.parquet"
    w_ts.to_parquet(out, index=False)
    log(f"saved {out}")

    # Quick sanity print
    print("\n  Men's 2026 sample (top 5 by NetEff):")
    print(m_ts[m_ts.Season == 2026].nlargest(5, "NetEff")[
        ["TeamID", "NetEff", "OffEff", "DefEff", "WinPct", "EFGPct", "Tempo"]
    ].round(3).to_string(index=False))

    print("\n  Women's 2026 sample (top 5 by NetEff):")
    print(w_ts[w_ts.Season == 2026].nlargest(5, "NetEff")[
        ["TeamID", "NetEff", "OffEff", "DefEff", "WinPct", "EFGPct", "Tempo"]
    ].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
