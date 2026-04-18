"""
Stage 10: True holdout evaluation — train on 2011-2025, test on 2026.
Reports AUC, accuracy, Brier score, log-loss, seed baseline, calibration.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                              roc_curve, brier_score_loss, log_loss)
from sklearn.calibration import calibration_curve
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
from matchup_predictor import generate_matchup_pairs


def run_holdout_evaluation(teams, trained_models, matchup_features, diff_cols):
    """Stage 10: Evaluate on 2026 holdout with all models + stacking + baselines."""
    cv5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    train_teams = teams[teams["season_year"] < HOLDOUT_YEAR]
    test_teams  = teams[teams["season_year"] == HOLDOUT_YEAR]

    print(f"  Building train matchups (2011-{HOLDOUT_YEAR-1})...")
    train_m = generate_matchup_pairs(train_teams, matchup_features, "Train")
    print(f"  Building test matchups ({HOLDOUT_YEAR})...")
    test_m  = generate_matchup_pairs(test_teams, matchup_features, "Test")

    train_m[diff_cols] = train_m[diff_cols].fillna(train_m[diff_cols].median())
    test_m[diff_cols]  = test_m[diff_cols].fillna(train_m[diff_cols].median())

    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values
    X_test  = test_m[diff_cols].values
    y_test  = test_m["label"].values

    # ── Seed-only baseline on 2026 ────────────────────────────────────
    winner_test = test_m[test_m["label"]==1].copy()
    seed_baseline_acc = (winner_test["winner_seed"] <= winner_test["loser_seed"]).mean()
    print(f"\n  SEED-ONLY BASELINE (2026): {seed_baseline_acc:.1%}")

    # ── Train & evaluate all models ───────────────────────────────────
    holdout_models = {}
    # Use same model types from Stage 9 but retrain on full training set
    holdout_models["Random Forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    if HAS_XGB:
        # Use best params from trained_models if available
        if "XGBoost" in trained_models:
            params = trained_models["XGBoost"].get_params()
            holdout_models["XGBoost"] = XGBClassifier(**{k:v for k,v in params.items()
                                                          if k not in ("n_jobs",)})
            holdout_models["XGBoost"].set_params(n_jobs=-1)
        else:
            holdout_models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", use_label_encoder=False,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if HAS_LGBM:
        if "LightGBM" in trained_models:
            params = trained_models["LightGBM"].get_params()
            holdout_models["LightGBM"] = LGBMClassifier(**{k:v for k,v in params.items()
                                                            if k not in ("n_jobs",)})
            holdout_models["LightGBM"].set_params(n_jobs=-1)
        else:
            holdout_models["LightGBM"] = LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    if HAS_CB:
        if "CatBoost" in trained_models:
            params = trained_models["CatBoost"].get_params()
            holdout_models["CatBoost"] = CatBoostClassifier(**{k:v for k,v in params.items()})
            holdout_models["CatBoost"].set_params(verbose=0)
        else:
            holdout_models["CatBoost"] = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=0)

    print(f"\n  {'Model':<20s} {'CV AUC':>8s} {'Test AUC':>9s} {'Test Acc':>9s} {'Brier':>8s} {'LogLoss':>9s}")
    print("  " + "-" * 67)

    best_model, best_auc, best_probs, best_preds, best_name = None, 0, None, None, ""
    all_holdout_results = []

    for name, model in holdout_models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv5, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        yp = model.predict_proba(X_test)[:,1]
        ya = model.predict(X_test)
        t_auc = roc_auc_score(y_test, yp)
        t_acc = accuracy_score(y_test, ya)
        brier = brier_score_loss(y_test, yp)
        ll = log_loss(y_test, yp)
        print(f"  {name:<20s} {cv_auc:>8.4f} {t_auc:>9.4f} {t_acc:>9.4f} {brier:>8.4f} {ll:>9.4f}")
        all_holdout_results.append({"model":name, "cv_auc":cv_auc, "test_auc":t_auc,
                                     "test_acc":t_acc, "brier":brier, "log_loss":ll})
        if t_auc > best_auc:
            best_auc, best_model, best_probs, best_preds, best_name = t_auc, model, yp, ya, name

    # ── Stacking Ensemble ─────────────────────────────────────────────
    estimators = [(n, m) for n, m in holdout_models.items() if n != "Decision Tree"]
    if len(estimators) >= 2:
        stack = StackingClassifier(
            estimators=[(n.lower().replace(" ","_"), m.__class__(**m.get_params())) for n, m in estimators],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=cv5, passthrough=False, n_jobs=-1,
        )
        stack.fit(X_train, y_train)
        yp_s = stack.predict_proba(X_test)[:,1]
        ya_s = stack.predict(X_test)
        s_auc = roc_auc_score(y_test, yp_s)
        s_acc = accuracy_score(y_test, ya_s)
        s_brier = brier_score_loss(y_test, yp_s)
        s_ll = log_loss(y_test, yp_s)
        print(f"  {'Stacking Ensemble':<20s} {'--':>8s} {s_auc:>9.4f} {s_acc:>9.4f} {s_brier:>8.4f} {s_ll:>9.4f}")
        all_holdout_results.append({"model":"Stacking", "cv_auc":None, "test_auc":s_auc,
                                     "test_acc":s_acc, "brier":s_brier, "log_loss":s_ll})
        if s_auc > best_auc:
            best_auc, best_model, best_probs, best_preds, best_name = s_auc, stack, yp_s, ya_s, "Stacking"

    # ── Per-pair analysis ─────────────────────────────────────────────
    test_m = test_m.copy()
    test_m["prob_winner"] = best_probs
    winner_rows = test_m[test_m["label"]==1].copy()
    winner_rows["correct"] = (winner_rows["prob_winner"] > 0.5).astype(int)
    winner_rows["seed_diff"] = winner_rows["loser_seed"] - winner_rows["winner_seed"]
    winner_rows["is_upset"] = winner_rows["seed_diff"] < 0

    chalk_rows = winner_rows[~winner_rows["is_upset"]]
    upset_rows = winner_rows[winner_rows["is_upset"]]

    print(f"\n  Best model: {best_name} (Test AUC = {best_auc:.4f})")
    print(f"  Unique 2026 matchup pairs: {len(winner_rows)}")
    print(f"  Correctly predicted: {winner_rows['correct'].sum()} ({winner_rows['correct'].mean():.1%})")
    print(f"  Chalk accuracy:  {chalk_rows['correct'].mean():.1%} ({chalk_rows['correct'].sum()}/{len(chalk_rows)})")
    print(f"  Upset accuracy:  {upset_rows['correct'].mean():.1%} ({upset_rows['correct'].sum()}/{len(upset_rows)})")
    print(f"  Seed-only baseline: {seed_baseline_acc:.1%}")
    print(f"  Model improvement over seed baseline: {winner_rows['correct'].mean() - seed_baseline_acc:+.1%}")

    # Key matchups
    key_matchups = [("Michigan","Connecticut"),("Arizona","Michigan"),("Illinois","Connecticut"),
                    ("Iowa","Illinois"),("Duke","Connecticut"),("Florida","Iowa"),("Wisconsin","High Point")]
    teams_2026_idx = teams[teams["season_year"]==HOLDOUT_YEAR].set_index("team")
    print(f"\n  Key 2026 matchup predictions:")
    print(f"  {'Team A':<22s} {'Team B':<22s} {'Actual':<12s} {'Predicted':>10s} {'Prob A':>8s} {'Correct':>8s}")
    print("  " + "-"*85)
    for ta, tb in key_matchups:
        if ta not in teams_2026_idx.index or tb not in teams_2026_idx.index:
            continue
        a, b = teams_2026_idx.loc[ta], teams_2026_idx.loc[tb]
        if a["games_won"] == b["games_won"]:
            continue
        actual = ta if a["games_won"] > b["games_won"] else tb
        diff = np.array([a[f]-b[f] for f in matchup_features]).reshape(1,-1)
        diff = np.where(np.isnan(diff), 0, diff)
        prob_a = best_model.predict_proba(diff)[0][1]
        predicted = ta if prob_a > 0.5 else tb
        correct = "YES" if predicted == actual else "NO <<"
        print(f"  {ta:<22s} {tb:<22s} {actual:<12s} {predicted:>10s} {prob_a:>8.1%} {correct:>8s}")

    # ── Plots ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(fpr, tpr, color="#e74c3c", linewidth=2, label=f"AUC={best_auc:.3f}")
    axes[0].plot([0,1],[0,1],"--",color="gray",alpha=0.5,label="Random")
    axes[0].set_title(f"ROC — 2026 Test\n{best_name}", fontsize=12)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()

    acc_by_diff = winner_rows.groupby("seed_diff")["correct"].mean()
    colors_bar = ["#e74c3c" if d<0 else "#3498db" for d in acc_by_diff.index]
    axes[1].bar(acc_by_diff.index, acc_by_diff.values, color=colors_bar, alpha=0.8)
    axes[1].axhline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(winner_rows["correct"].mean(), color="green", linestyle="--", alpha=0.7,
                    label=f"Overall ({winner_rows['correct'].mean():.1%})")
    axes[1].axhline(seed_baseline_acc, color="orange", linestyle=":", alpha=0.7,
                    label=f"Seed baseline ({seed_baseline_acc:.1%})")
    axes[1].set_title("Accuracy by Seed Diff\n(Red=Upset, Blue=Chalk)", fontsize=12)
    axes[1].set_xlabel("Loser seed - Winner seed")
    axes[1].legend(fontsize=9)

    # Calibration
    fraction_pos, mean_pred = calibration_curve(y_test, best_probs, n_bins=10, strategy="uniform")
    axes[2].plot([0,1],[0,1],"--",color="gray",alpha=0.7,label="Perfect")
    axes[2].plot(mean_pred, fraction_pos, "o-", color="#e74c3c", linewidth=2, label=best_name)
    axes[2].set_title("Probability Calibration (2026)", fontsize=12)
    axes[2].set_xlabel("Predicted Win Prob")
    axes[2].set_ylabel("Actual Win Rate")
    axes[2].legend()

    plt.suptitle(f"2026 True Holdout — {best_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("stage10_eval_2026")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Loss","Pred Win"], yticklabels=["Act Loss","Act Win"])
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12)
    plt.tight_layout()
    save_fig("stage10_confusion_matrix")

    # Save results
    pd.DataFrame(all_holdout_results).to_csv(f"{OUTPUT_DIR}/holdout_results.csv", index=False)
    log_step("[saved] holdout_results.csv")

    return all_holdout_results
