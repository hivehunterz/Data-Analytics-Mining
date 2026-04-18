"""
Download and load Barttorvik team-season tables.

URL pattern: http://barttorvik.com/{season}_team_results.csv
Coverage: 2008-2026 (earlier years exist but fields may differ).

Kevin's top features from this source:
  adjoe, adjde, barthag, WAB, sos, adjt (tempo)
"""
from pathlib import Path
import time
import urllib.request
import urllib.error

import pandas as pd
from rapidfuzz import fuzz, process

from config import BARTTORVIK
from src.loaders import kaggle_loader as kl
from src.utils import log


URL_TEMPLATE = "http://barttorvik.com/{season}_team_results.csv"


def download_all(seasons: list[int], force: bool = False) -> list[Path]:
    """Download CSVs for each season that's missing (or all if force=True)."""
    BARTTORVIK.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seasons:
        p = BARTTORVIK / f"{s}_team_results.csv"
        if p.exists() and not force:
            paths.append(p)
            continue
        url = URL_TEMPLATE.format(season=s)
        try:
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, p)
            # Be polite
            time.sleep(0.5)
            paths.append(p)
        except urllib.error.HTTPError as e:
            log(f"  {s}: HTTP {e.code} (skipping)")
        except Exception as e:
            log(f"  {s}: {type(e).__name__} — {e}")
    return paths


# Canonical columns we'll keep (ignore the rest).
KEEP_COLS = {
    "rank":    "BartRank",
    "team":    "team_bart",
    "conf":    "bart_conf",
    "adjoe":   "adjoe",
    "adjde":   "adjde",
    "barthag": "barthag",
    "sos":     "bart_sos",
    "ncsos":   "bart_ncsos",
    "WAB":     "WAB",
    "adjt":    "adjt",
    "FUN":     "bart_FUN",
    "proj. W": "bart_proj_W",
    "proj. L": "bart_proj_L",
}


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using KEEP_COLS map (case-insensitive) and coerce numerics."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for want, new_name in KEEP_COLS.items():
        wkey = want.lower()
        if wkey in lower:
            rename[lower[wkey]] = new_name
    df = df.rename(columns=rename)
    cols = [c for c in KEEP_COLS.values() if c in df.columns]
    out = df[cols].copy()
    # Coerce numeric columns (handles mixed-type strings like '1-' in older CSVs)
    numeric_cols = ["BartRank", "adjoe", "adjde", "barthag", "bart_sos",
                    "bart_ncsos", "WAB", "adjt", "bart_FUN",
                    "bart_proj_W", "bart_proj_L"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_season(season: int) -> pd.DataFrame:
    p = BARTTORVIK / f"{season}_team_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run download_all first.")
    df = pd.read_csv(p)
    df = _normalize_cols(df)
    df["Season"] = season
    return df


def load_all(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            frames.append(load_season(s))
        except FileNotFoundError:
            log(f"  {s}: not downloaded, skip")
    return pd.concat(frames, ignore_index=True)


# ── Team name -> Kaggle TeamID mapping ────────────────────────────────
def build_team_mapping(is_men: bool = True) -> dict:
    """
    Map Barttorvik's `team` strings to Kaggle TeamIDs via MTeamSpellings /
    WTeamSpellings with fuzzy fallback. Returns {bart_name: TeamID}.
    """
    if is_men:
        teams = kl.load_m_teams()[["TeamID", "TeamName"]]
        spellings = kl.load_m_team_spellings()
    else:
        teams = kl.load_w_teams()[["TeamID", "TeamName"]]
        spellings = kl.load_w_team_spellings()

    exact = {}
    for _, r in teams.iterrows():
        exact[r.TeamName.lower().strip()] = r.TeamID
    for _, r in spellings.iterrows():
        key = str(r.TeamNameSpelling).lower().strip()
        if key not in exact:
            exact[key] = r.TeamID

    return exact


def fuzzy_map(names: list, lookup: dict, score_cutoff: int = 80) -> dict:
    """
    Map unmapped Barttorvik names via rapidfuzz. Returns {bart_name: TeamID}.
    """
    mapping = {}
    unmatched = []
    keys = list(lookup.keys())
    for n in set(names):
        if not isinstance(n, str):
            continue
        k = n.lower().strip()
        if k in lookup:
            mapping[n] = lookup[k]
            continue
        # Common cleanups
        cleaned = k.replace(" st.", " state").replace(" st ", " state ")
        if cleaned in lookup:
            mapping[n] = lookup[cleaned]
            continue
        match = process.extractOne(k, keys, scorer=fuzz.WRatio,
                                    score_cutoff=score_cutoff)
        if match:
            mapping[n] = lookup[match[0]]
        else:
            unmatched.append(n)
    if unmatched:
        log(f"  unmatched Barttorvik teams ({len(unmatched)}): "
            f"{unmatched[:10]}...")
    return mapping


def attach_teamid(bart_df: pd.DataFrame, is_men: bool = True) -> pd.DataFrame:
    """Return bart_df with TeamID column added."""
    bart_df = bart_df.dropna(subset=["team_bart"]).copy()
    lookup = build_team_mapping(is_men=is_men)
    mapping = fuzzy_map(bart_df["team_bart"].tolist(), lookup)
    bart_df["TeamID"] = bart_df["team_bart"].map(mapping)
    missing = bart_df["TeamID"].isna().sum()
    if missing:
        log(f"  {missing} teams unmapped (will be dropped)")
    return bart_df.dropna(subset=["TeamID"]).astype({"TeamID": int})
