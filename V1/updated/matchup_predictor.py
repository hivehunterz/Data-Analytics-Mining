"""
Stages 8-9: Team profile aggregation + Head-to-head matchup prediction.
Models: RF, XGBoost, LightGBM, CatBoost, Decision Tree + Stacking Ensemble.
Includes seed-only baseline comparison.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False

from config import OUTPUT_DIR, RANDOM_STATE, CV_FOLDS, HOLDOUT_YEAR
from utils import log_step, save_fig


def aggregate_team(team_df):
    """Aggregate player-level stats to a single team profile."""
    team_df = team_df.sort_values("minutes_per_game", ascending=False)
    result = {}
    for col in ["pts_per_game","trb_per_game","ast_per_game","stl_per_game",
                "blk_per_game","tov_per_game","fg_pct","three_p_pct","ft_pct",
                "efg_pct","ts_pct","per","bpm","obpm","dbpm","usg_pct","three_par","ftr"]:
        result[f"team_{col}"] = np.average(team_df[col], weights=team_df["minutes_per_game"])
    result["team_total_ws"]        = team_df["ws"].sum()
    result["team_n_players"]       = len(team_df)
    result["team_starter_avg_bpm"] = team_df.head(5)["bpm"].mean()
    result["team_bench_avg_bpm"]   = team_df.iloc[5:]["bpm"].mean() if len(team_df) > 5 else 0
    result["team_depth"]           = result["team_bench_avg_bpm"] - result["team_starter_avg_bpm"]
    result["team_top_scorer_ppg"]  = team_df["pts_per_game"].max()
    result["team_scoring_balance"] = team_df["pts_per_game"].std()
    result["team_max_bpm"]         = team_df["bpm"].max()
    result["team_max_dbpm"]        = team_df["dbpm"].max()
    result["team_impact_score_avg"]= team_df["impact_score"].mean()
    result["team_has_exceptional"] = int(team_df["is_exceptional"].any())
    result["tournament_seed"]      = team_df["tournament_seed"].iloc[0]
    result["games_won"]            = team_df["games_won"].iloc[0]
    result["tournament_result"]    = team_df["tournament_result"].iloc[0]
    result["team_srs"]             = team_df["team_srs"].iloc[0]
    return pd.Series(result)


def build_team_profiles(df):
    """Stage 8: Aggregate all player stats to team level."""
    n_groups = df.groupby(["season_year","team"]).ngroups
    print(f"  Aggregating {n_groups} team-seasons...")
    team_list = []
    for idx, ((year, team), team_df) in enumerate(df.groupby(["season_year","team"])):
        row = aggregate_team(team_df)
        row["season_year"] = year
        row["team"] = team
        team_list.append(row)
        if (idx + 1) % 200 == 0 or (idx + 1) == n_groups:
            print(f"    [{idx+1}/{n_groups}] aggregated...", flush=True)
    teams = pd.DataFrame(team_list)
    print(f"  Created {len(teams)} team profiles with {len([c for c in teams.columns if c.startswith('team_')])} features")
    return teams


def generate_matchup_pairs(teams, matchup_features, label=""):
    """Generate symmetric pairwise matchup instances from team profiles."""
    rows = []
    season_groups = list(teams.groupby("season_year"))
    for si, (season, st) in enumerate(season_groups):
        tl = st.reset_index(drop=True)
        n = len(tl)
        for i in range(n):
            for j in range(i+1, n):
                t1, t2 = tl.iloc[i], tl.iloc[j]
                if t1["games_won"] == t2["games_won"]:
                    continue
                w, l = (t1, t2) if t1["games_won"] > t2["games_won"] else (t2, t1)
                row  = {f"diff_{f}": w[f]-l[f] for f in matchup_features}
                row.update({"label":1, "winner_team":w["team"], "loser_team":l["team"],
                            "winner_seed":w["tournament_seed"], "loser_seed":l["tournament_seed"],
                            "winner_wins":w["games_won"], "loser_wins":l["games_won"], "season":season})
                rowf = {f"diff_{f}": l[f]-w[f] for f in matchup_features}
                rowf.update({"label":0, "winner_team":l["team"], "loser_team":w["team"],
                             "winner_seed":l["tournament_seed"], "loser_seed":w["tournament_seed"],
                             "winner_wins":l["games_won"], "loser_wins":w["games_won"], "season":season})
                rows.append(row)
                rows.append(rowf)
        print(f"    {label} [{si+1}/{len(season_groups)}] {int(season)}: {n} teams -> {len(rows)//2} pairs so far", flush=True)
    out = pd.DataFrame(rows)
    print(f"  {label}: {len(out)} samples ({len(out)//2} unique pairs)")
    return out


def run_matchup_predictor(df):
    """Stages 8-9: Build team profiles, generate matchups, train models with tuning + stacking."""
    teams = build_team_profiles(df)

    matchup_features = [c for c in teams.columns
                        if c.startswith("team_")
                        and c not in ["team_srs","tournament_seed","team_has_exceptional",
                                      "team_n_players","team_total_ws"]
                        and teams[c].dtype in [np.float64, np.int64, float, int]]
    log_step(f"Matchup features: {len(matchup_features)}")

    # Generate training matchups (2011-2025)
    teams_pre = teams[teams["season_year"] < HOLDOUT_YEAR]
    print(f"\n  Generating matchup pairs (2011-{HOLDOUT_YEAR-1})...")
    matchups = generate_matchup_pairs(teams_pre, matchup_features, "Train")

    diff_cols = [c for c in matchups.columns if c.startswith("diff_")]
    matchups[diff_cols] = matchups[diff_cols].fillna(matchups[diff_cols].median())
    X_match = matchups[diff_cols].values
    y_match = matchups["label"].values

    cv5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Seed-only baseline ────────────────────────────────────────────
    winner_rows_train = matchups[matchups["label"]==1]
    seed_baseline_acc = (winner_rows_train["winner_seed"] <= winner_rows_train["loser_seed"]).mean()
    print(f"\n  SEED-ONLY BASELINE (pick lower seed): {seed_baseline_acc:.1%}")

    # ── Build models with hyperparameter tuning ───────────────────────
    model_specs = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [8, 10, 12, None],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", 0.3],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "max_depth": [4, 5, 6, 7],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [2, 5, 10],
            },
        },
    }
    if HAS_XGB:
        model_specs["XGBoost"] = {
            "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                                   random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
        }
    if HAS_LGBM:
        model_specs["LightGBM"] = {
            "model": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "subsample": [0.7, 0.8, 1.0],
            },
        }
    if HAS_CB:
        model_specs["CatBoost"] = {
            "model": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            "params": {
                "iterations": [200, 300, 500],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5],
            },
        }

    print(f"\n  {'Model':<20s} {'AUC':>8s} {'Acc':>8s} {'Brier':>8s} {'LogLoss':>9s}")
    print("  " + "-" * 57)

    best_models = {}
    for name, spec in model_specs.items():
        print(f"\n    [*] Tuning {name} (20 iterations x {CV_FOLDS}-fold CV)...", flush=True)
        search = RandomizedSearchCV(
            spec["model"], spec["params"], n_iter=20, cv=cv5,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_match, y_match)
        print(f"    [*] {name} tuning done. Running cross-val evaluation...", flush=True)
        best_m = search.best_estimator_
        auc = cross_val_score(best_m, X_match, y_match, cv=cv5, scoring="roc_auc").mean()
        acc = cross_val_score(best_m, X_match, y_match, cv=cv5, scoring="accuracy").mean()
        from sklearn.model_selection import cross_val_predict
        probs_cv = cross_val_predict(best_m, X_match, y_match, cv=cv5, method="predict_proba")[:,1]
        brier = brier_score_loss(y_match, probs_cv)
        ll = log_loss(y_match, probs_cv)
        print(f"  {name:<20s} {auc:>8.4f} {acc:>8.4f} {brier:>8.4f} {ll:>9.4f}")
        best_models[name] = best_m

    # ── Stacking Ensemble ─────────────────────────────────────────────
    estimators = []
    if "Random Forest" in best_models:
        estimators.append(("rf", best_models["Random Forest"]))
    if "XGBoost" in best_models:
        estimators.append(("xgb", best_models["XGBoost"]))
    if "LightGBM" in best_models:
        estimators.append(("lgbm", best_models["LightGBM"]))
    if "CatBoost" in best_models:
        estimators.append(("cb", best_models["CatBoost"]))

    if len(estimators) >= 2:
        print(f"\n  Training Stacking Ensemble ({len(estimators)} base learners -> LogisticRegression)...")
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=cv5, passthrough=False, n_jobs=-1,
        )
        auc_s = cross_val_score(stack, X_match, y_match, cv=cv5, scoring="roc_auc").mean()
        acc_s = cross_val_score(stack, X_match, y_match, cv=cv5, scoring="accuracy").mean()
        from sklearn.model_selection import cross_val_predict
        probs_stack = cross_val_predict(stack, X_match, y_match, cv=cv5, method="predict_proba")[:,1]
        brier_s = brier_score_loss(y_match, probs_stack)
        ll_s = log_loss(y_match, probs_stack)
        print(f"  {'Stacking Ensemble':<20s} {auc_s:>8.4f} {acc_s:>8.4f} {brier_s:>8.4f} {ll_s:>9.4f}")
        stack.fit(X_match, y_match)
        best_models["Stacking"] = stack

    # ── Win Shares Ablation (matchup level) ───────────────────────────
    ws_feat_names = [c for c in diff_cols if any(w in c for w in ["_ws","_ows","_dws","_total_ws"])]
    non_ws_cols = [c for c in diff_cols if c not in ws_feat_names]
    if ws_feat_names and "XGBoost" in best_models:
        X_no_ws = matchups[non_ws_cols].fillna(0).values
        ablation_model = best_models["XGBoost"].__class__(**best_models["XGBoost"].get_params())
        auc_full = cross_val_score(best_models["XGBoost"], X_match, y_match, cv=cv5, scoring="roc_auc").mean()
        auc_no_ws = cross_val_score(ablation_model, X_no_ws, y_match, cv=cv5, scoring="roc_auc").mean()
        print(f"\n  WIN SHARES ABLATION (matchup model):")
        print(f"    With WS features:    AUC = {auc_full:.4f}")
        print(f"    Without WS features: AUC = {auc_no_ws:.4f}")
        print(f"    Delta:               {auc_full - auc_no_ws:+.4f}")

    # Fit all final models on full training data
    for name, model in best_models.items():
        if name != "Stacking":
            model.fit(X_match, y_match)

    # Feature importance from best RF
    if "Random Forest" in best_models:
        imp_match = pd.Series(best_models["Random Forest"].feature_importances_,
                              index=diff_cols).sort_values(ascending=False)
        print(f"\n  Top 15 matchup features (RF importance):")
        for feat, imp in imp_match.head(15).items():
            print(f"    {feat.replace('diff_team_',''):<35s} {imp:.4f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        imp_match.head(15).sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("viridis", 15))
        ax.set_title("Top 15 Features for Predicting Matchup Winner", fontsize=13)
        ax.set_xlabel("Feature Importance (Gini)")
        ax.set_yticklabels([l.get_text().replace("diff_team_","") for l in ax.get_yticklabels()])
        save_fig("stage9_matchup_feature_importance")

    # Decision tree rules
    if "Decision Tree" in best_models:
        dt = best_models["Decision Tree"]
        tree_text = export_text(dt, feature_names=[c.replace("diff_team_","") for c in diff_cols], max_depth=5)
        with open(f"{OUTPUT_DIR}/decision_tree_rules.txt", "w") as f:
            f.write("Decision Tree Rules\nPositive diff = Team A is higher\nLabel 1 = Team A wins\n\n")
            f.write(tree_text)
        log_step("[saved] decision_tree_rules.txt")

    teams.to_csv(f"{OUTPUT_DIR}/team_profiles.csv", index=False)
    log_step("[saved] team_profiles.csv")

    return teams, best_models, matchup_features, diff_cols
