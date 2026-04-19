"""
Stage 35: Generate prediction-diagnostic figures for the writeup.

Produces four PNGs saved to outputs/figures/:
  1. calibration.png  — OOF margin vs empirical win rate, overlaid with
                         per-gender isotonic fit.
  2. prob_distribution.png — histogram of 2026 predicted P(T1 wins)
                              for the full submission set, split m/w.
  3. elo_by_seed.png  — stripplot of men's Elo (2015-2025 seasons) by
                         tournament seed, showing that our MOV-weighted Elo
                         separates top seeds from bench teams.
  4. brier_progression.png — bar chart of men's 2026 Brier by pipeline
                              stage (A..P).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from config import (PROCESSED, SUBMISSIONS, TRUTH_DIR, OUTPUTS,
                     MEN_SEASONS_BARTT, WOMEN_SEASONS_FULL,
                     MEN_ID_CUTOFF, RANDOM_STATE, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.market_blend_v2 import load_injuries
from src.utils import log

FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=700,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.6,
    colsample_bynode=0.8,
    min_child_weight=4,
    tree_method="hist",
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)

BARTT_COLS = ["adjoe", "adjde", "barthag", "bart_sos", "bart_ncsos",
              "WAB", "adjt", "BartRank"]


def _load_pruned(gender, json_name=None, key=None):
    import json
    from config import CV_REPORTS
    p = CV_REPORTS / (json_name or f"{gender}_full_pruned.json")
    with open(p) as f:
        payload = json.load(f)
    if key:
        payload = payload[key]
    feat_list = payload.get("final_features") or payload.get("features")
    return [c.replace("diff_", "") for c in feat_list if c.replace("diff_", "") != "SeedNum"]


def loso_oof_margins(train_df, diff_cols):
    seasons = sorted(train_df["Season"].unique())
    oof = np.full(len(train_df), np.nan)
    for s in seasons:
        tr = train_df["Season"] != s
        te = train_df["Season"] == s
        imp = SimpleImputer(strategy="median").fit(train_df.loc[tr, diff_cols].values)
        Xtr = imp.transform(train_df.loc[tr, diff_cols].values)
        ytr = train_df.loc[tr, "margin"].values
        Xte = imp.transform(train_df.loc[te, diff_cols].values)
        model = XGBRegressor(**XGB_PARAMS).fit(Xtr, ytr)
        oof[te.values] = model.predict(Xte)
    return oof


def _build_men_feat():
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_bart = pd.read_parquet(PROCESSED / "m_barttorvik.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()
    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    m_feat = m_feat.merge(m_bart[["Season", "TeamID"] + BARTT_COLS],
                            on=["Season", "TeamID"], how="left")
    return m_feat, m_seeds, m_elo


def _build_women_feat():
    w_team = pd.read_parquet(PROCESSED / "w_team_season.parquet")
    w_elo  = pd.read_parquet(PROCESSED / "w_elo.parquet")
    w_cust = pd.read_parquet(PROCESSED / "w_ratings_custom.parquet")
    w_seeds = kl.load_w_seeds()
    w_confs = kl.load_w_conferences()
    w_secondary = kl._read("WSecondaryTourneyTeams.csv")
    w_reg = kl.load_w_regular_compact()
    w_opq = compute_opp_quality_pts_won(w_reg, w_seeds, w_secondary)
    w_team_hr = build_harry_rating(w_team, w_opq, w_confs, is_men=False)
    w_inter = add_interactions_women(w_team_hr, w_cust, w_elo, w_seeds)
    w_feat = w_inter.merge(w_elo[["Season", "TeamID", "EloSlope"]],
                            on=["Season", "TeamID"], how="left")
    return w_feat, w_seeds


def fig_calibration():
    """Figure 1: Margin -> empirical win rate, with isotonic overlay."""
    log("Generating calibration figure...")
    m_feat, m_seeds, _ = _build_men_feat()
    m_features = _load_pruned("men", "men_barttorvik.json", key="B_restricted")
    m_tourney = kl.load_m_tourney_compact()
    m_tour_in = m_tourney[m_tourney["Season"].isin(MEN_SEASONS_BARTT)]
    train_m = build_tourney_matchups(m_tour_in, m_feat, m_features, seeds_df=m_seeds)
    diff_cols_m = [f"diff_{c}" for c in m_features] + ["diff_SeedNum"]
    oof_m = loso_oof_margins(train_m, diff_cols_m)

    w_feat, w_seeds = _build_women_feat()
    w_features = _load_pruned("women")
    w_tourney = kl.load_w_tourney_compact()
    w_tour_in = w_tourney[w_tourney["Season"].isin(WOMEN_SEASONS_FULL)]
    train_w = build_tourney_matchups(w_tour_in, w_feat, w_features, seeds_df=w_seeds)
    diff_cols_w = [f"diff_{c}" for c in w_features] + ["diff_SeedNum"]
    oof_w = loso_oof_margins(train_w, diff_cols_w)

    iso_m = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_m.fit(oof_m, train_m["label"].values)
    iso_w = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_w.fit(oof_w, train_w["label"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, oof, labels, iso, title in [
        (axes[0], oof_m, train_m["label"].values, iso_m, "Men (742 matchups, 2015-2025)"),
        (axes[1], oof_w, train_w["label"].values, iso_w, "Women (961 matchups, 2010-2025)"),
    ]:
        bins = np.linspace(np.floor(oof.min()), np.ceil(oof.max()), 22)
        centers = (bins[:-1] + bins[1:]) / 2
        digits = np.digitize(oof, bins) - 1
        digits = np.clip(digits, 0, len(centers) - 1)
        emp = np.full(len(centers), np.nan)
        n_in = np.zeros(len(centers))
        for i in range(len(centers)):
            m = digits == i
            if m.sum() >= 5:
                emp[i] = labels[m].mean()
                n_in[i] = m.sum()
        ax.scatter(centers, emp, s=np.sqrt(n_in) * 6,
                   color="#1f77b4", alpha=0.75, label="empirical win rate")
        xs = np.linspace(oof.min(), oof.max(), 200)
        ax.plot(xs, iso.predict(xs), "r-", lw=2, label="isotonic fit")
        ax.axhline(0.5, color="gray", lw=0.6, ls="--")
        ax.axvline(0.0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("Predicted margin (points)")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Empirical P(T1 beats T2)")
    plt.tight_layout()
    out = FIG_DIR / "calibration.png"
    plt.savefig(out, dpi=140)
    plt.close()
    log(f"  saved {out}")


def fig_prob_distribution():
    """Figure 2: histogram of predicted P(T1 wins) on the 2026 submission."""
    log("Generating prediction-distribution figure...")
    sub = pd.read_csv(SUBMISSIONS / "v2_margin_plus_market.csv")
    sub["T1"] = sub["ID"].str.split("_").str[1].astype(int)
    men = sub[sub["T1"] < MEN_ID_CUTOFF]["Pred"].values
    women = sub[sub["T1"] >= MEN_ID_CUTOFF]["Pred"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(men, bins=bins, alpha=0.55, color="#1f77b4", label=f"Men  (n={len(men)})", edgecolor="white")
    ax.hist(women, bins=bins, alpha=0.55, color="#d62728", label=f"Women (n={len(women)})", edgecolor="white")
    ax.set_xlabel("Predicted P(T1 beats T2)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 2026 submission predictions (all Stage-2 matchups)")
    ax.axvline(0.5, color="gray", lw=0.8, ls="--")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "prob_distribution.png"
    plt.savefig(out, dpi=140)
    plt.close()
    log(f"  saved {out}")


def fig_elo_by_seed():
    """Figure 3: stripplot of men's Elo by tournament seed (2015-2025)."""
    log("Generating Elo-by-seed figure...")
    m_elo = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_seeds = kl.load_m_seeds()
    j = m_seeds.merge(m_elo[["Season", "TeamID", "Elo"]], on=["Season", "TeamID"], how="left")
    j = j[j["Season"].isin(MEN_SEASONS_BARTT)].dropna(subset=["Elo"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    seeds = sorted(j["SeedNum"].unique())
    for s in seeds:
        vals = j[j["SeedNum"] == s]["Elo"].values
        jitter = np.random.RandomState(s).uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(np.full(len(vals), s) + jitter, vals,
                   s=12, alpha=0.55, color="#1f77b4")
    med = j.groupby("SeedNum")["Elo"].median().reindex(seeds)
    ax.plot(seeds, med.values, "ro-", lw=1.5, markersize=5, label="Median Elo")
    ax.set_xlabel("Tournament seed")
    ax.set_ylabel("MOV-weighted Elo (end-of-season)")
    ax.set_title("Men's Elo rating by tournament seed (2015-2025 seasons)")
    ax.set_xticks(seeds)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "elo_by_seed.png"
    plt.savefig(out, dpi=140)
    plt.close()
    log(f"  saved {out}")


def fig_brier_progression():
    """Figure 4: bar chart of cumulative men's 2026 Brier by pipeline stage."""
    log("Generating Brier-progression figure...")
    stages = [
        ("A. seed-only", None),
        ("F. +Four Factors, Elo, Massey,\n  custom ratings, interactions", 0.1635),
        ("G. +tune C", 0.1560),
        ("H. +Barttorvik", 0.1539),
        ("I. +Vegas Tier 1", 0.1515),
        ("J. +injury Elo penalty", 0.1529),
        ("K. +Bradley-Terry Tier 2", 0.1472),
        ("L. +LR/XGB 70/30 blend", 0.1460),
        ("M. +10 more market games", 0.1451),
        ("N. -clipping", 0.14451),
        ("O. curated injuries", 0.14437),
        ("P. margin regression\n  + isotonic", 0.14311),
    ]
    labels = [s[0] for s in stages if s[1] is not None]
    values = [s[1] for s in stages if s[1] is not None]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#888"] * len(values)
    colors[-1] = "#d62728"  # highlight final stage
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("2026 men's Brier")
    ax.set_title("Men's 2026 Brier by pipeline stage (lower is better)")
    ax.axhline(values[-1], color="red", lw=0.8, ls="--", alpha=0.4)
    ax.set_ylim(0.14, max(values) + 0.005)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}",
                ha="center", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "brier_progression.png"
    plt.savefig(out, dpi=140)
    plt.close()
    log(f"  saved {out}")


def main():
    np.random.seed(RANDOM_STATE)
    fig_calibration()
    fig_prob_distribution()
    fig_elo_by_seed()
    fig_brier_progression()
    log("All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
