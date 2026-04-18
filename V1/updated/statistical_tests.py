"""
Stage 4: Statistical significance testing — Deep Run vs Early Exit.
Welch's t-tests with Bonferroni correction, Cohen's d effect sizes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import OUTPUT_DIR, HOLDOUT_YEAR
from utils import cohens_d, save_fig


TEST_COLS = [
    "pts_per_game","trb_per_game","ast_per_game","stl_per_game",
    "blk_per_game","fg_pct","three_p_pct","ts_pct","efg_pct",
    "per","ws","bpm","obpm","dbpm","dws","ows",
    "usg_pct","tov_per_game","ftr","three_par",
    "scoring_efficiency","defensive_impact","playmaking_index",
    "rebounding_rate","win_shares_per_game",
]


def run_statistical_tests(df):
    """Run Welch's t-tests on 2011-2025 data. Returns list of result dicts."""
    df_analysis = df[df["season_year"] < HOLDOUT_YEAR]
    deep  = df_analysis[df_analysis["deep_run"] == 1]
    early = df_analysis[df_analysis["deep_run"] == 0]
    print(f"    >> Using 2011-{HOLDOUT_YEAR-1} only: {len(df_analysis)} players")
    print(f"    >> Deep Run: {len(deep)}  |  Early Exit: {len(early)}")

    n_tests = len(TEST_COLS)
    bonferroni_alpha = 0.05 / n_tests
    print(f"\n  Bonferroni-corrected alpha = {bonferroni_alpha:.4f} ({n_tests} tests)\n")
    print(f"  {'Stat':<25s} {'Deep':>8s} {'Early':>8s} {'Diff':>7s} {'Cohen d':>8s} {'p-value':>10s} {'Sig':>4s}")
    print("  " + "-" * 78)

    all_results = []
    for col in TEST_COLS:
        d_v = deep[col].dropna()
        e_v = early[col].dropna()
        d = cohens_d(d_v, e_v)
        _, p = stats.ttest_ind(d_v, e_v, equal_var=False)
        sig = "***" if p < bonferroni_alpha/10 else ("**" if p < bonferroni_alpha else ("*" if p < 0.05 else ""))
        print(f"  {col:<25s} {d_v.mean():>8.3f} {e_v.mean():>8.3f} {d_v.mean()-e_v.mean():>+7.3f} "
              f"{d:>+8.3f} {p:>10.2e} {sig:>4s}")
        all_results.append({
            "feature": col, "deep_mean": d_v.mean(), "early_mean": e_v.mean(),
            "cohens_d": d, "p_value": p,
            "significant_bonferroni": p < bonferroni_alpha,
        })

    significant = [(r["feature"], r["cohens_d"], r["p_value"])
                    for r in all_results if r["significant_bonferroni"] and abs(r["cohens_d"]) >= 0.2]
    large  = [(f,d,p) for f,d,p in significant if abs(d) >= 0.5]
    medium = [(f,d,p) for f,d,p in significant if 0.2 <= abs(d) < 0.5]
    print(f"\n  Large effect (|d|>=0.5): {len(large)}")
    for f,d,p in sorted(large, key=lambda x: abs(x[1]), reverse=True):
        print(f"    {f:<25s} d={d:+.3f}  p={p:.2e}")
    print(f"  Medium effect: {len(medium)}")
    for f,d,p in sorted(medium, key=lambda x: abs(x[1]), reverse=True):
        print(f"    {f:<25s} d={d:+.3f}  p={p:.2e}")

    # ── Plots ─────────────────────────────────────────────────────────
    rdf = pd.DataFrame(all_results).sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#e74c3c" if abs(d)>=0.5 else "#f39c12" if abs(d)>=0.2 else "#95a5a6"
              for d in rdf["cohens_d"]]
    ax.barh(rdf["feature"], rdf["cohens_d"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.2, color="green", linestyle="--", alpha=0.5, label="Small (0.2)")
    ax.axvline(-0.2, color="green", linestyle="--", alpha=0.5)
    ax.axvline( 0.5, color="red",   linestyle="--", alpha=0.5, label="Medium (0.5)")
    ax.axvline(-0.5, color="red",   linestyle="--", alpha=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Deep Run vs Early Exit: Effect Sizes", fontsize=13)
    ax.legend(loc="lower right")
    save_fig("stage4_effect_sizes")

    top_feats = sorted(significant, key=lambda x: abs(x[1]), reverse=True)[:6]
    if top_feats:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, (col, d_val, p_val) in enumerate(top_feats):
            ax = axes.ravel()[i]
            for label, subset, color in [("Early Exit", early, "#e74c3c"), ("Deep Run", deep, "#2ecc71")]:
                ax.hist(subset[col].dropna(), bins=30, alpha=0.5, label=label, color=color, density=True)
            ax.set_title(f"{col}\n(d={d_val:+.3f}, p={p_val:.1e})", fontsize=11)
            ax.legend(fontsize=9)
        plt.suptitle("Most Significant Differences: Deep Run vs Early Exit", fontsize=13, y=1.02)
        plt.tight_layout()
        save_fig("stage4_distributions")

    pd.DataFrame(all_results).to_csv(f"{OUTPUT_DIR}/statistical_tests.csv", index=False)
    return all_results
