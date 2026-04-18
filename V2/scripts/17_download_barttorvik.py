"""
Stage 17: Download Barttorvik team_results CSVs for all relevant seasons.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loaders.barttorvik_loader import download_all
from src.utils import log


def main():
    seasons = list(range(2008, 2027))
    log(f"downloading Barttorvik for seasons {seasons[0]}-{seasons[-1]}...")
    paths = download_all(seasons, force=False)
    log(f"have {len(paths)} season CSVs on disk")


if __name__ == "__main__":
    main()
