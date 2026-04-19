"""
Stage 22: Build womens_2026_truth.csv from manually-entered 2026 women's
tournament bracket results (sourced from Wikipedia + NCAA.com).

Produces data_raw/ground_truth/womens_2026_truth.csv with columns:
    Round, WTeamID, LTeamID, Slot (optional)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rapidfuzz import fuzz, process

from config import TRUTH_DIR
from src.loaders import kaggle_loader as kl
from src.utils import log


# Source: Wikipedia 2026 NCAA D1 Women's Basketball Tournament + NCAA.com
# Format: (winner_name, loser_name, round)
# Rounds: 0=R68 (First Four), 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=NCG

GAMES = [
    # First Four (Round 0) — these don't score in Kaggle Stage 2 (play-in),
    # but we include them for completeness.
    ("Missouri State", "Stephen F. Austin", 0),
    ("Nebraska", "Richmond", 0),
    ("Southern", "Samford", 0),
    ("Virginia", "Arizona State", 0),

    # Round of 64 (32 games)
    # Fort Worth #1 Regional
    ("Connecticut", "UTSA", 1),
    ("Syracuse", "Iowa State", 1),
    ("Maryland", "Murray State", 1),
    ("North Carolina", "Western Illinois", 1),
    ("Notre Dame", "Fairfield", 1),
    ("Ohio State", "Howard", 1),
    ("Illinois", "Colorado", 1),
    ("Vanderbilt", "High Point", 1),
    # Sacramento #4 Regional
    ("South Carolina", "Southern", 1),
    ("USC", "Clemson", 1),
    ("Michigan State", "Colorado State", 1),
    ("Oklahoma", "Idaho", 1),
    ("Washington", "South Dakota State", 1),
    ("TCU", "UC San Diego", 1),
    ("Virginia", "Georgia", 1),
    ("Iowa", "Fairleigh Dickinson", 1),
    # Sacramento #2 Regional
    ("UCLA", "California Baptist", 1),
    ("Oklahoma State", "Princeton", 1),
    ("Ole Miss", "Gonzaga", 1),
    ("Minnesota", "Green Bay", 1),
    ("Baylor", "Nebraska", 1),
    ("Duke", "Charleston", 1),
    ("Villanova", "Texas Tech", 1),
    ("LSU", "Jacksonville", 1),
    # Fort Worth #3 Regional
    ("Texas", "Missouri State", 1),
    ("Oregon", "Virginia Tech", 1),
    ("Kentucky", "James Madison", 1),
    ("West Virginia", "Miami (OH)", 1),
    ("Alabama", "Rhode Island", 1),
    ("Louisville", "Vermont", 1),
    ("NC State", "Tennessee", 1),
    ("Michigan", "Holy Cross", 1),

    # Round of 32 (16 games)
    ("Connecticut", "Syracuse", 2),
    ("North Carolina", "Maryland", 2),
    ("Notre Dame", "Ohio State", 2),
    ("Vanderbilt", "Illinois", 2),
    ("South Carolina", "USC", 2),
    ("Oklahoma", "Michigan State", 2),
    ("TCU", "Washington", 2),
    ("Virginia", "Iowa", 2),
    ("UCLA", "Oklahoma State", 2),
    ("Minnesota", "Ole Miss", 2),
    ("Duke", "Baylor", 2),
    ("LSU", "Villanova", 2),
    ("Texas", "Oregon", 2),
    ("Kentucky", "West Virginia", 2),
    ("Louisville", "Alabama", 2),
    ("Michigan", "NC State", 2),

    # Sweet 16 (8 games)
    ("Connecticut", "North Carolina", 3),
    ("Notre Dame", "Vanderbilt", 3),
    ("South Carolina", "Oklahoma", 3),
    ("TCU", "Virginia", 3),
    ("UCLA", "Minnesota", 3),
    ("Duke", "LSU", 3),
    ("Texas", "Kentucky", 3),
    ("Michigan", "Louisville", 3),

    # Elite 8 (4 games)
    ("Connecticut", "Notre Dame", 4),
    ("South Carolina", "TCU", 4),
    ("UCLA", "Duke", 4),
    ("Texas", "Michigan", 4),

    # Final Four (2 games)
    ("South Carolina", "Connecticut", 5),
    ("UCLA", "Texas", 5),

    # Championship (1 game)
    ("UCLA", "South Carolina", 6),
]


def build_lookup():
    """Team name -> WTeamID via WTeams + WTeamSpellings + fuzzy."""
    teams = kl.load_w_teams()[["TeamID", "TeamName"]]
    spellings = kl.load_w_team_spellings()
    lookup = {}
    for _, r in teams.iterrows():
        lookup[r.TeamName.lower().strip()] = r.TeamID
    for _, r in spellings.iterrows():
        k = str(r.TeamNameSpelling).lower().strip()
        lookup.setdefault(k, r.TeamID)
    return lookup


def resolve(name: str, lookup: dict) -> int | None:
    k = name.lower().strip()
    if k in lookup:
        return lookup[k]
    # Hand-overrides for common mismatches
    overrides = {
        "connecticut": "connecticut",
        "uconn": "connecticut",
        "ole miss": "mississippi",
        "nc state": "north carolina state",
        "usc": "southern california",
        "miami (oh)": "miami oh",
        "utsa": "texas-san antonio",
        "uc san diego": "uc-san diego",
        "tcu": "texas christian",
        "holy cross": "holy cross",
    }
    if k in overrides:
        candidate = overrides[k]
        if candidate in lookup:
            return lookup[candidate]
    # Fuzzy fallback
    match = process.extractOne(k, lookup.keys(),
                                scorer=fuzz.WRatio, score_cutoff=80)
    return lookup[match[0]] if match else None


def main():
    lookup = build_lookup()
    rows = []
    unmapped = []
    for winner_name, loser_name, round_ in GAMES:
        w = resolve(winner_name, lookup)
        l = resolve(loser_name, lookup)
        if w is None:
            unmapped.append(winner_name)
            continue
        if l is None:
            unmapped.append(loser_name)
            continue
        rows.append({
            "Season": 2026, "Round": round_,
            "WTeamID": w, "LTeamID": l,
        })
    if unmapped:
        log(f"UNMAPPED: {sorted(set(unmapped))}")

    df = pd.DataFrame(rows)
    out = TRUTH_DIR / "womens_2026_truth.csv"
    df.to_csv(out, index=False)
    log(f"saved {out}  ({len(df)} games)")
    print(f"\nGames by round:")
    print(df.groupby("Round").size().to_string())

    # Sanity check: champion is UCLA
    champ = df[df.Round == 6].iloc[0]
    teams = kl.load_w_teams().set_index("TeamID")["TeamName"].to_dict()
    print(f"\nChampionship: {teams[champ.WTeamID]} def. {teams[champ.LTeamID]}")


if __name__ == "__main__":
    main()
