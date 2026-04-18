"""
Build 2026 ground-truth game outcomes for local Brier evaluation.

Men's: derive from V1/data/mm2026_updated.csv (921 patched rows, one per
player-season). Group by team, take one row -> (team, seed, games_won,
tournament_result). Map team -> TeamID via MTeams + MTeamSpellings. Then
reconstruct bracket using MNCAATourneySlots.

Women's: manual spreadsheet entry at data_raw/ground_truth/womens_2026_truth.csv
(columns: Round, WTeamID, LTeamID). This module simply reads/validates it.
"""
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from config import V1_DATA_DIR, TRUTH_DIR, HOLDOUT_SEASON
from src.loaders import kaggle_loader as kl
from src.utils import log


RESULT_TO_ROUND = {
    "R68": 0, "R64": 1, "R32": 2, "S16": 3,
    "E8": 4, "F4": 5, "NCG": 6, "Champion": 7,
}
# round index 1..6 corresponds to "games won to reach NEXT round"
# games_won = round number of their LAST win. Champion = 6 wins.


def _build_team_to_teamid_map() -> dict:
    """
    Build a dict from V1 team string -> Kaggle TeamID using MTeams + MTeamSpellings.
    Uses rapidfuzz as a fallback for unmapped strings.
    """
    teams = kl.load_m_teams()[["TeamID", "TeamName"]]
    spellings = kl.load_m_team_spellings()   # cols: TeamNameSpelling, TeamID

    # exact lower-case matches first
    name_to_id = {}
    for _, r in teams.iterrows():
        name_to_id[r.TeamName.lower().strip()] = r.TeamID
    for _, r in spellings.iterrows():
        name_to_id[str(r.TeamNameSpelling).lower().strip()] = r.TeamID

    return name_to_id


def _fuzzy_team_lookup(name: str, name_to_id: dict) -> int | None:
    """Fuzzy fallback when an exact name isn't found."""
    key = name.lower().strip()
    if key in name_to_id:
        return name_to_id[key]
    # Remove parentheses / punctuation
    cleaned = re.sub(r"\([^)]*\)", "", key).strip()
    if cleaned in name_to_id:
        return name_to_id[cleaned]
    # Rapidfuzz last-resort
    match = process.extractOne(key, name_to_id.keys(),
                               scorer=fuzz.WRatio, score_cutoff=85)
    if match is not None:
        return name_to_id[match[0]]
    return None


def load_mens_2026_tournament_table() -> pd.DataFrame:
    """
    From V1/data/mm2026_updated.csv, produce one row per team with:
    (TeamID, SeedStr, SeedNum, GamesWon, Result)
    """
    src = V1_DATA_DIR / "mm2026_updated.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing V1 source: {src}. Restore V1 archive first.")

    df = pd.read_csv(src)
    df = df[df["season_year"] == HOLDOUT_SEASON].copy()

    # One row per team
    team_rows = (
        df.groupby("team")
        .first()[["tournament_seed", "games_won", "tournament_result"]]
        .reset_index()
    )

    # Map team name -> TeamID
    name_map = _build_team_to_teamid_map()
    unmapped = []
    team_ids = []
    for name in team_rows["team"]:
        tid = _fuzzy_team_lookup(str(name), name_map)
        if tid is None:
            unmapped.append(name)
        team_ids.append(tid)
    team_rows["TeamID"] = team_ids

    if unmapped:
        log(f"WARNING: {len(unmapped)} teams unmapped from V1: {unmapped}")

    team_rows = team_rows.dropna(subset=["TeamID"])
    team_rows["TeamID"] = team_rows["TeamID"].astype(int)
    team_rows["SeedNum"] = team_rows["tournament_seed"].astype(int)
    team_rows["GamesWon"] = team_rows["games_won"].astype(int)
    team_rows["Result"] = team_rows["tournament_result"]
    # ResultNumeric = round of last game played: R68=0, R64=1, R32=2, ...
    team_rows["ResultNumeric"] = team_rows["Result"].map(RESULT_TO_ROUND).fillna(-1).astype(int)
    return team_rows[["TeamID", "team", "SeedNum", "GamesWon", "Result", "ResultNumeric"]]


def _reconstruct_men_bracket(team_table: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild round-by-round (WTeamID, LTeamID, Round) using 2026 seeds +
    MNCAATourneySlots structure.

    Approach: walk the bracket. At each slot, take the two feeding teams;
    the one with more games_won (and whose result number >= this round)
    is the winner.
    """
    slots = kl._read("MNCAATourneySlots.csv")
    slots = slots[slots["Season"] == HOLDOUT_SEASON].copy()
    seeds = kl.load_m_seeds()
    seeds = seeds[seeds["Season"] == HOLDOUT_SEASON][["Seed", "TeamID"]]

    # Quick lookups. Use ResultNumeric (round of last game played) as the
    # primary signal — it disambiguates play-in winners from R64 winners.
    result_num = dict(zip(team_table["TeamID"], team_table["ResultNumeric"]))
    seed_num   = dict(zip(team_table["TeamID"], team_table["SeedNum"]))

    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))

    # Map slot -> TeamID winner
    slot_winner: dict = {}
    games = []

    # Process slots by round order: slot names starting with R1, R2, R3, R4, R5, R6
    # plus R68 play-in slots like "W11" (First Four)
    def slot_to_round(slot: str) -> int:
        # "R1W1" -> round 1; "R6CH" -> round 6
        m = re.match(r"R([1-6])", slot)
        if m:
            return int(m.group(1))
        # First Four slots begin with the region letter + seed number
        return 0

    # Resolve strong/weak seeds: start with play-in rounds then R1..R6
    # Process in round order so dependencies resolve
    slots_sorted = slots.copy()
    slots_sorted["round_order"] = slots_sorted["Slot"].apply(slot_to_round)
    slots_sorted = slots_sorted.sort_values("round_order")

    def resolve(seed_or_slot: str):
        """Return TeamID given a seed code (like 'W01') or a prior slot name."""
        if seed_or_slot in seed_to_team:
            return seed_to_team[seed_or_slot]
        return slot_winner.get(seed_or_slot)

    for _, row in slots_sorted.iterrows():
        t1 = resolve(row["StrongSeed"])
        t2 = resolve(row["WeakSeed"])
        round_ = row["round_order"]
        if t1 is None or t2 is None:
            continue
        # Winner of this slot's round R = team whose ResultNumeric > R
        # (they advanced past this round). Loser has ResultNumeric == R.
        r1, r2 = result_num.get(t1, -1), result_num.get(t2, -1)
        if r1 > round_ and r2 <= round_:
            winner, loser = t1, t2
        elif r2 > round_ and r1 <= round_:
            winner, loser = t2, t1
        elif r1 > round_ and r2 > round_:
            # Both advanced — bracket tie means they both won this slot.
            # Impossible under single-elim unless data corruption; fall
            # back to higher ResultNumeric (deeper run).
            winner, loser = (t1, t2) if r1 >= r2 else (t2, t1)
        else:
            # Neither advanced past this round (both r <= round_). Can't
            # happen in a real bracket unless one of them didn't actually
            # play this game. Skip.
            continue
        slot_winner[row["Slot"]] = winner
        if round_ > 0:
            games.append({
                "Season": HOLDOUT_SEASON,
                "Round": round_,
                "WTeamID": winner,
                "LTeamID": loser,
                "Slot": row["Slot"],
            })

    return pd.DataFrame(games)


def build_mens_2026_truth() -> pd.DataFrame:
    """End-to-end: produce men's 2026 ground-truth bracket and save."""
    tbl = load_mens_2026_tournament_table()
    log(f"men's 2026 teams found: {len(tbl)}")
    bracket = _reconstruct_men_bracket(tbl)
    log(f"men's 2026 games reconstructed: {len(bracket)}")
    out = TRUTH_DIR / "mens_2026_truth.csv"
    bracket.to_csv(out, index=False)
    log(f"saved {out}")
    return bracket


def load_womens_2026_truth() -> pd.DataFrame:
    """Read manually-entered women's truth. Raises if missing."""
    path = TRUTH_DIR / "womens_2026_truth.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Enter 2026 women's tournament games manually "
            "with columns: Round, WTeamID, LTeamID."
        )
    df = pd.read_csv(path)
    required = {"Round", "WTeamID", "LTeamID"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must have columns {required}")
    log(f"women's 2026 truth loaded: {len(df)} games")
    return df


if __name__ == "__main__":
    build_mens_2026_truth()
