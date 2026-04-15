"""
Stage 6: Temporal trends, conference analysis, seed reliability.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import OUTPUT_DIR
from utils import save_fig


def run_temporal_analysis(df):
    """Stage 6: Temporal trends, conference chi-squared, seed correlation."""
    yearly = df.groupby("season_year").agg(
        avg_pts=("pts_per_game","mean"), avg_3pct=("three_p_pct","mean"),
        avg_3par=("three_par","mean"),   avg_ts=("ts_pct","mean"),
        avg_per=("per","mean"),          avg_pace=("usg_pct","mean"),
        n=("player_name","count")).reset_index()

    trends = [("avg_pts","Avg PPG"),("avg_3pct","Avg 3P%"),
              ("avg_3par","3-Pt Attempt Rate"),("avg_ts","True Shooting %"),
              ("avg_per","Avg PER"),("avg_pace","Avg Usage %")]

    print(f"\n  {'Metric':<25s} {'Slope/yr':>10s} {'r':>8s} {'p':>10s} {'Sig':>5s}")
    print("  " + "-" * 62)
    for col, title in trends:
        slope, intercept, r, p, se = stats.linregress(yearly["season_year"], yearly[col])
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        direction = "UP" if slope > 0 else "DOWN"
        print(f"  {title:<25s} {slope:>+10.5f} {r:>8.4f} {p:>10.4f} {sig:>5s} {direction}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title), color in zip(axes.ravel(), trends,
            ["#e74c3c","#3498db","#2ecc71","#9b59b6","#e67e22","#1abc9c"]):
        ax.plot(yearly["season_year"], yearly[col], "o-", color=color, linewidth=2)
        slope, intercept, r, p, _ = stats.linregress(yearly["season_year"], yearly[col])
        ax.plot(yearly["season_year"], intercept + slope*yearly["season_year"], "--", color="gray", alpha=0.7)
        ax.set_title(f"{title}\nr={r:.3f}, p={p:.3f}" + (" *" if p<0.05 else " (ns)"), fontsize=11)
        ax.set_xlabel("Season")
    plt.suptitle("College Basketball Trends (2011-2026)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig("stage6_temporal_trends")

    # Conference analysis
    conf_data = df[df["conference"] != "Unknown"]
    conf_success = conf_data.groupby("conference").agg(
        avg_games_won=("games_won","mean"), deep_run_rate=("deep_run","mean"),
        avg_seed=("tournament_seed","mean"), player_count=("player_name","count"),
        avg_srs=("team_srs","mean")).sort_values("avg_games_won", ascending=False)
    conf_success = conf_success[conf_success["player_count"] >= 50]

    conf_players = conf_data[conf_data["conference"].isin(conf_success.index)]
    contingency = pd.crosstab(conf_players["conference"], conf_players["deep_run"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    print(f"\n  Top conferences by tournament success:")
    print(conf_success.head(12).round(3).to_string())
    print(f"\n  Chi-squared (conference vs deep run): chi2={chi2:.1f}, p={chi_p:.2e}, dof={dof}")

    fig, ax = plt.subplots(figsize=(12, 6))
    tc = conf_success.head(15)
    x = range(len(tc)); w = 0.35
    ax.bar([i-w/2 for i in x], tc["avg_games_won"], w, label="Avg Games Won", color="#3498db")
    ax.bar([i+w/2 for i in x], tc["deep_run_rate"]*5, w, label="Deep Run Rate (x5)", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tc.index, rotation=45, ha="right")
    ax.set_title("Tournament Performance by Conference", fontsize=13)
    ax.legend()
    plt.tight_layout()
    save_fig("stage6_conference_analysis")

    # Seed reliability
    seed_corr, seed_p = stats.spearmanr(df["tournament_seed"], df["games_won"])
    print(f"\n  Spearman (seed vs games_won): rho={seed_corr:.4f}, p={seed_p:.2e}")
