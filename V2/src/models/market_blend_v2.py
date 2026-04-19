"""
Extended market blend: Vegas spreads (Tier 1) + Bradley-Terry from
championship futures (Tier 2) + injury adjustments.

For any matchup:
  - If we have a Vegas spread for that pair, use Φ(spread/11) as p_market.
    Blend: p_final = alpha_t1 * p_lr + (1 - alpha_t1) * p_market.
  - Else if both teams have championship futures, use Bradley-Terry
    conversion: p_market = p_A_implied / (p_A_implied + p_B_implied).
    Blend: p_final = alpha_t2 * p_lr + (1 - alpha_t2) * p_market.
  - Else: keep LR prediction.

Injury adjustments: for each team with a listed injury, discount the team's
Elo rating (affects LR via the diff_Elo feature) by a weighted factor:
    status_weights = {
        "OUT_FOR_SEASON": 1.0,
        "OUT":            0.75,
        "GAME_TIME":      0.50,
    }
    elo_adjustment = -status_weights[status] * role_share * 100
(Why 100: rough conversion — a full star out is ~100 Elo points.)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from rapidfuzz import fuzz, process

from config import MARKETS_DIR, INJURIES_DIR
from src.loaders import kaggle_loader as kl
from src.utils import log


CBB_MARGIN_SIGMA = 11.0


def spread_to_prob(spread: float) -> float:
    return float(norm.cdf(spread / CBB_MARGIN_SIGMA))


def build_team_lookup(is_men: bool = True) -> dict:
    if is_men:
        teams = kl.load_m_teams()[["TeamID", "TeamName"]]
        spellings = kl.load_m_team_spellings()
    else:
        teams = kl.load_w_teams()[["TeamID", "TeamName"]]
        spellings = kl.load_w_team_spellings()
    lookup = {}
    for _, r in teams.iterrows():
        lookup[r.TeamName.lower().strip()] = r.TeamID
    for _, r in spellings.iterrows():
        k = str(r.TeamNameSpelling).lower().strip()
        lookup.setdefault(k, r.TeamID)
    return lookup


def resolve_team(name: str, lookup: dict) -> int | None:
    k = name.lower().strip()
    if k in lookup:
        return lookup[k]
    match = process.extractOne(k, lookup.keys(),
                                scorer=fuzz.WRatio, score_cutoff=80)
    return lookup[match[0]] if match else None


def load_vegas_spreads() -> pd.DataFrame:
    """Men's Vegas spreads → (T1, T2, market_prob_T1)."""
    df = pd.read_csv(MARKETS_DIR / "vegas_2026_r64_men.csv")
    # Filter out First Four rows (Kaggle Stage 2 might include them, but we
    # have separate injury info for R64+ — keep them all and let the join
    # decide which matter).
    lookup = build_team_lookup(is_men=True)
    df["fav_id"] = df["favorite"].apply(lambda n: resolve_team(n, lookup))
    df["und_id"] = df["underdog"].apply(lambda n: resolve_team(n, lookup))
    df = df.dropna(subset=["fav_id", "und_id"]).astype({"fav_id": int, "und_id": int})
    df["p_fav"] = df["spread"].apply(spread_to_prob)
    df["T1"] = df[["fav_id", "und_id"]].min(axis=1)
    df["T2"] = df[["fav_id", "und_id"]].max(axis=1)
    df["market_prob_T1"] = np.where(df["fav_id"] == df["T1"], df["p_fav"], 1 - df["p_fav"])
    return df[["T1", "T2", "market_prob_T1"]].drop_duplicates(["T1", "T2"])


def load_championship_futures() -> pd.DataFrame:
    """Men's championship futures → (TeamID, implied_prob)."""
    df = pd.read_csv(MARKETS_DIR / "championship_futures_2026_men.csv")
    lookup = build_team_lookup(is_men=True)
    df["TeamID"] = df["team"].apply(lambda n: resolve_team(n, lookup))
    unmapped = df[df["TeamID"].isna()]
    if len(unmapped):
        log(f"futures: unmapped teams: {unmapped['team'].tolist()}")
    df = df.dropna(subset=["TeamID"]).astype({"TeamID": int})
    # Normalize implied probabilities so they sum to 1 (remove vig)
    df["implied_norm"] = df["implied_prob"] / df["implied_prob"].sum()
    return df[["TeamID", "implied_norm"]]


def load_injuries() -> pd.DataFrame:
    """
    Injury report → (TeamID, elo_penalty).

    Uses injury_report_v3.csv which has a principled role_share computed
    by the formula (see ROLE_SHARE_FORMULA below), and excludes:
      (a) players whose multi-month absences are already baked into the
          team's regular-season Elo (e.g. Kentucky's Lowe, Gonzaga's Huff)
      (b) players confirmed healthy at tipoff (Kansas's Peterson,
          Purdue's Kaufman-Renn)

    Formula (applied in scripts/30_compute_role_share.py):
        role_share = mpg/200 + 0.75 * ppg/team_ppg
        clipped to [0.05, 0.45]
    """
    df = pd.read_csv(INJURIES_DIR / "injury_report_v3.csv")
    status_w = {"OUT_FOR_SEASON": 1.0, "OUT": 0.75, "GAME_TIME": 0.50}
    df["status_weight"] = df["status"].map(status_w).fillna(0)
    df["elo_penalty_player"] = df["status_weight"] * df["role_share"] * 100

    lookup = build_team_lookup(is_men=True)
    df["TeamID"] = df["team"].apply(lambda n: resolve_team(n, lookup))
    df = df.dropna(subset=["TeamID"]).astype({"TeamID": int})
    # Sum penalties per team (multi-player injuries stack)
    team_penalty = df.groupby("TeamID")["elo_penalty_player"].sum().reset_index()
    team_penalty = team_penalty.rename(columns={"elo_penalty_player": "elo_penalty"})
    return team_penalty


def apply_full_blend(
    sub_df: pd.DataFrame,
    alpha_t1: float = 0.20,
    alpha_t2: float = 0.75,
) -> pd.DataFrame:
    """
    Apply tiered blend. sub_df has columns T1, T2, Pred.
    Returns updated with p_final.
    """
    out = sub_df.copy()
    vegas = load_vegas_spreads()
    futures = load_championship_futures()

    # Tier 1: Vegas
    vegas_lookup = dict(zip(zip(vegas["T1"], vegas["T2"]), vegas["market_prob_T1"]))
    # Tier 2: Bradley-Terry from championship futures
    futures_lookup = dict(zip(futures["TeamID"], futures["implied_norm"]))

    t1_count, t2_count = 0, 0
    for i, row in out.iterrows():
        t1, t2 = row["T1"], row["T2"]
        vegas_p = vegas_lookup.get((t1, t2))
        if vegas_p is not None:
            out.at[i, "Pred"] = alpha_t1 * row["Pred"] + (1 - alpha_t1) * vegas_p
            t1_count += 1
            continue
        p_t1 = futures_lookup.get(t1)
        p_t2 = futures_lookup.get(t2)
        if p_t1 is not None and p_t2 is not None and (p_t1 + p_t2) > 0:
            bt = p_t1 / (p_t1 + p_t2)
            out.at[i, "Pred"] = alpha_t2 * row["Pred"] + (1 - alpha_t2) * bt
            t2_count += 1
    log(f"market blend: Tier 1 (Vegas) applied to {t1_count} rows, "
        f"Tier 2 (Bradley-Terry) applied to {t2_count} rows")
    return out
