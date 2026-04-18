"""
Market-blend module — Kevin's tiered weight scheme applied to 2026 men's.

We have Vegas point spreads for first-round (R64) + round-of-32 (R32)
games from ESPN + CBS, scraped April 2026. Convert each spread to a
favorite win probability using the standard CBB assumption:

    margin ~ N(spread, 11^2)
    P(favorite wins) = Phi(spread / 11)

Kevin's tier-1 weights for men's R1 were 0.10 LR + 0.90 market. We'll
sweep this empirically on the 2026 holdout since we already have the
actual outcomes — this is the ONE place we validate a blend weight.

Returns a function that augments a submission DataFrame with a market-
blended Pred column where market data exists.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from rapidfuzz import fuzz, process

from config import MARKETS_DIR
from src.loaders import kaggle_loader as kl
from src.utils import log


CBB_MARGIN_SIGMA = 11.0   # empirical stdev of NCAA men's game margin


def spread_to_prob(spread: float, sigma: float = CBB_MARGIN_SIGMA) -> float:
    """P(favorite wins) given point spread (positive = favorite)."""
    return float(norm.cdf(spread / sigma))


def load_vegas_2026_men() -> pd.DataFrame:
    """
    Returns DataFrame with columns (T1, T2, market_prob_T1), where T1 < T2
    and market_prob_T1 is P(T1 wins) per Vegas.
    """
    path = MARKETS_DIR / "vegas_2026_r64_men.csv"
    df = pd.read_csv(path)

    # Map team names to Kaggle TeamIDs
    teams = kl.load_m_teams()[["TeamID", "TeamName"]]
    spellings = kl.load_m_team_spellings()
    lookup = {}
    for _, r in teams.iterrows():
        lookup[r.TeamName.lower().strip()] = r.TeamID
    for _, r in spellings.iterrows():
        k = str(r.TeamNameSpelling).lower().strip()
        lookup.setdefault(k, r.TeamID)

    keys = list(lookup.keys())

    def _resolve(name: str) -> int | None:
        k = name.lower().strip()
        if k in lookup:
            return lookup[k]
        m = process.extractOne(k, keys, scorer=fuzz.WRatio, score_cutoff=85)
        return lookup[m[0]] if m else None

    df["fav_id"] = df["favorite"].apply(_resolve)
    df["und_id"] = df["underdog"].apply(_resolve)
    unmapped = df[df["fav_id"].isna() | df["und_id"].isna()]
    if len(unmapped):
        log(f"market: unmapped team rows:\n{unmapped[['favorite','underdog']].to_string()}")
    df = df.dropna(subset=["fav_id", "und_id"]).astype({"fav_id": int, "und_id": int})

    # Compute P(favorite wins) from spread
    df["p_fav"] = df["spread"].apply(spread_to_prob)

    # Canonicalize to (T1, T2) with T1 < T2
    df["T1"] = df[["fav_id", "und_id"]].min(axis=1)
    df["T2"] = df[["fav_id", "und_id"]].max(axis=1)
    df["market_prob_T1"] = np.where(df["fav_id"] == df["T1"], df["p_fav"], 1 - df["p_fav"])
    return df[["T1", "T2", "market_prob_T1", "spread", "source"]]


def apply_tiered_blend(
    sub_df: pd.DataFrame,
    market_df: pd.DataFrame,
    alpha_r1: float = 0.10,
    alpha_deep: float = 0.75,
) -> pd.DataFrame:
    """
    Blend LR predictions with market probabilities using Kevin's tier scheme.

    sub_df : columns T1, T2, Pred   (one row per Stage-2 matchup pair)
    market_df : columns T1, T2, market_prob_T1

    Returns sub_df with Pred updated for rows that match a market game.
    alpha is LR weight: final = alpha * lr + (1-alpha) * market.

    For now, all scraped market games are R1 or R2, so we apply alpha_r1 = 0.10
    (Kevin's men's Tier 1 weight for R1-with-markets).
    """
    out = sub_df.copy()
    lookup = dict(zip(zip(market_df["T1"], market_df["T2"]),
                      market_df["market_prob_T1"]))
    matched = 0
    for i, row in out.iterrows():
        key = (row["T1"], row["T2"])
        if key in lookup:
            mkt = lookup[key]
            out.at[i, "Pred"] = alpha_r1 * row["Pred"] + (1 - alpha_r1) * mkt
            matched += 1
    log(f"market blend: applied to {matched} rows")
    return out
