"""
NCAA March Madness 2026 — Updated Data Mining Pipeline
=====================================================
  Stage  1-3: Load data, clean, feature engineer
  Stage  4:   Statistical significance testing (Welch's t, Bonferroni, Cohen's d)
  Stage  5:   Deep run classification (RF, XGBoost, LightGBM, CatBoost + tuning)
  Stage  6:   Temporal trends & conference analysis
  Stage  7:   Exceptional player identification (weighted z-scores)
  Stage  8:   Team profile aggregation (MP-weighted)
  Stage  9:   Head-to-head matchup prediction (tuned models + stacking ensemble)
  Stage 10:   True holdout evaluation (2026) with seed baseline & Brier/log-loss

Improvements over v1:
  - XGBoost, LightGBM, CatBoost added alongside RF/GBM
  - RandomizedSearchCV hyperparameter tuning on all models
  - Stacking ensemble (RF + XGB + LGBM + CB -> LogisticRegression)
  - Seed-only baseline reported for context
  - Brier score + log-loss alongside AUC/accuracy
  - Win Shares ablation study
  - Calibration plots for deep-run classifier (Stage 5)
  - Removed clustering / archetype labeling

Run:  python -X utf8 pipeline.py
"""

import sys, time, warnings, os
warnings.filterwarnings("ignore")

# Ensure updated/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, LOG_FILE
from utils import DualWriter, log_stage

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.stdout = DualWriter(LOG_FILE)

TOTAL = 10

print("=" * 70)
print("  NCAA MARCH MADNESS 2026 — UPDATED PIPELINE")
print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Output:  {OUTPUT_DIR}/")
print(f"  Log:     {LOG_FILE}")
print("=" * 70)


# ── Stage 1-3: Data Loading, Cleaning, Feature Engineering ───────────
log_stage(1, TOTAL, "LOAD DATA, CLEAN, FEATURE ENGINEER")
from data_loader import load_pipeline
df = load_pipeline()
print(f"    >> Stage 1-3 complete. {len(df)} player-seasons ready.")


# ── Stage 4: Statistical Significance Testing ────────────────────────
log_stage(4, TOTAL, "STATISTICAL SIGNIFICANCE TESTING (2011-2025)")
from statistical_tests import run_statistical_tests
stat_results = run_statistical_tests(df)
print(f"    >> Stage 4 complete.")


# ── Stage 5: Deep Run Classification (with tuning) ───────────────────
log_stage(5, TOTAL, "DEEP RUN PREDICTION (RF + XGBoost + LightGBM + CatBoost)")
from classification import run_classification
best_clf_models, clf_results = run_classification(df)
print(f"\n    >> Models trained: {list(best_clf_models.keys())}")
print(f"    >> Stage 5 complete.")


# ── Stage 6: Temporal Trends & Conference Analysis ────────────────────
log_stage(6, TOTAL, "TEMPORAL TRENDS & CONFERENCE ANALYSIS")
from temporal_analysis import run_temporal_analysis
run_temporal_analysis(df)
print(f"    >> Stage 6 complete.")


# ── Stage 7: Exceptional Player Identification ────────────────────────
log_stage(7, TOTAL, "EXCEPTIONAL PLAYER IDENTIFICATION")
from exceptional_players import run_exceptional_identification
df = run_exceptional_identification(df)
print(f"    >> Stage 7 complete.")


# ── Stage 8-9: Team Profiles & Matchup Prediction ────────────────────
log_stage(8, TOTAL, "TEAM AGGREGATION & MATCHUP PREDICTION (with stacking)")
from matchup_predictor import run_matchup_predictor
teams, matchup_models, matchup_features, diff_cols = run_matchup_predictor(df)
print(f"\n    >> Models trained: {list(matchup_models.keys())}")
print(f"    >> Stage 8-9 complete.")


# ── Stage 10: True Holdout Evaluation (2026) ──────────────────────────
log_stage(10, TOTAL, "TRUE HOLDOUT EVALUATION (2026)")
from holdout_evaluation import run_holdout_evaluation
holdout_results = run_holdout_evaluation(teams, matchup_models, matchup_features, diff_cols)
print(f"    >> Stage 10 complete.")


# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  PIPELINE COMPLETE at {time.strftime('%H:%M:%S')}")
print(f"  Report:  {LOG_FILE}")
print(f"  Outputs: {OUTPUT_DIR}/")
print(f"{'='*70}")

# Save processed data
from config import COMBINED_COLS
df.to_csv(f"{OUTPUT_DIR}/mm2026_processed.csv", index=False)
print(f"    >> [saved] mm2026_processed.csv")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Pipeline complete. See outputs in:", OUTPUT_DIR)
