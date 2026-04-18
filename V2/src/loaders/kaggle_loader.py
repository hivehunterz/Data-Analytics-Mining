"""
Load raw Kaggle competition CSVs from `new dataset/`.

Returns pandas DataFrames without any transformation. Downstream features
modules are responsible for derived columns.
"""
from pathlib import Path
import pandas as pd

from config import KAGGLE_DIR


def _read(name: str) -> pd.DataFrame:
    path = KAGGLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing Kaggle file: {path}")
    return pd.read_csv(path)


# ── Men's ─────────────────────────────────────────────────────────────
def load_m_regular_compact() -> pd.DataFrame:
    return _read("MRegularSeasonCompactResults.csv")


def load_m_regular_detailed() -> pd.DataFrame:
    return _read("MRegularSeasonDetailedResults.csv")


def load_m_tourney_compact() -> pd.DataFrame:
    return _read("MNCAATourneyCompactResults.csv")


def load_m_tourney_detailed() -> pd.DataFrame:
    return _read("MNCAATourneyDetailedResults.csv")


def load_m_seeds() -> pd.DataFrame:
    df = _read("MNCAATourneySeeds.csv")
    # parse numeric seed e.g. "W04" -> 4, "Y16a" -> 16
    df["SeedNum"] = df["Seed"].str.extract(r"(\d+)").astype(int)
    return df


def load_m_massey() -> pd.DataFrame:
    return _read("MMasseyOrdinals.csv")


def load_m_teams() -> pd.DataFrame:
    return _read("MTeams.csv")


def load_m_team_spellings() -> pd.DataFrame:
    # Non-ASCII text; Kaggle uses Latin-1 encoding
    path = KAGGLE_DIR / "MTeamSpellings.csv"
    return pd.read_csv(path, encoding="latin-1")


def load_m_coaches() -> pd.DataFrame:
    return _read("MTeamCoaches.csv")


def load_m_conferences() -> pd.DataFrame:
    return _read("MTeamConferences.csv")


def load_m_secondary_compact() -> pd.DataFrame:
    return _read("MSecondaryTourneyCompactResults.csv")


# ── Women's ───────────────────────────────────────────────────────────
def load_w_regular_compact() -> pd.DataFrame:
    return _read("WRegularSeasonCompactResults.csv")


def load_w_regular_detailed() -> pd.DataFrame:
    return _read("WRegularSeasonDetailedResults.csv")


def load_w_tourney_compact() -> pd.DataFrame:
    return _read("WNCAATourneyCompactResults.csv")


def load_w_tourney_detailed() -> pd.DataFrame:
    return _read("WNCAATourneyDetailedResults.csv")


def load_w_seeds() -> pd.DataFrame:
    df = _read("WNCAATourneySeeds.csv")
    df["SeedNum"] = df["Seed"].str.extract(r"(\d+)").astype(int)
    return df


def load_w_teams() -> pd.DataFrame:
    return _read("WTeams.csv")


def load_w_team_spellings() -> pd.DataFrame:
    path = KAGGLE_DIR / "WTeamSpellings.csv"
    return pd.read_csv(path, encoding="latin-1")


def load_w_conferences() -> pd.DataFrame:
    return _read("WTeamConferences.csv")


# ── Shared ────────────────────────────────────────────────────────────
def load_sample_submission_stage2() -> pd.DataFrame:
    df = _read("SampleSubmissionStage2.csv")
    parts = df["ID"].str.split("_", expand=True)
    df["Season"] = parts[0].astype(int)
    df["T1"]     = parts[1].astype(int)   # lower TeamID by convention
    df["T2"]     = parts[2].astype(int)
    return df


def load_conferences() -> pd.DataFrame:
    return _read("Conferences.csv")


def load_m_seasons() -> pd.DataFrame:
    return _read("MSeasons.csv")


def load_w_seasons() -> pd.DataFrame:
    return _read("WSeasons.csv")
