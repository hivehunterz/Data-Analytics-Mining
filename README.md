# NCAA March Madness 2026 — Data Mining Project

**SC2320 Data Analytics & Mining** | Pearl Apply Pay, Cow Peh Cow Moo Sheng Yan, Tan Ee Khai

A full data mining pipeline that predicts NCAA Division I Men's Basketball Tournament outcomes from player-level statistics, evaluated on a true 2026 holdout.

## Key Results

| Model | 2026 Holdout AUC | 2026 Accuracy |
|---|---|---|
| **Random Forest** | **0.9116** | **84.0%** |
| XGBoost (tuned) | 0.8911 | 81.4% |
| LightGBM (tuned) | 0.8896 | 80.8% |
| CatBoost (tuned) | 0.8949 | 82.2% |
| Stacking Ensemble | 0.8816 | 81.3% |
| *Seed-only baseline* | *—* | *85.0%* |

- **Correctly called Michigan as 2026 Champion** (83.0% probability vs UConn).
- **DBPM** is the strongest individual predictor of tournament success (Cohen's d = 0.717).
- Models use only **player-aggregated stats** — no seed, no SRS, no team record.
- Win Shares ablation confirmed 0.0 AUC delta — BPM fully subsumes WS information.

## Pipeline Stages

1. **Load & Patch 2026** — manually scraped tournament results from NCAA.com, ESPN, On3, Wikipedia
2. **Clean** — filter GP≥5 & MPG≥5, median imputation → 10,028 player-seasons
3. **Feature Engineering** — 7 composite features (scoring efficiency, playmaking index, etc.)
4. **Statistical Tests** — Welch's t-test with Bonferroni correction, Cohen's d
5. **Deep-Run Classification** — RF, XGBoost, LightGBM, CatBoost with RandomizedSearchCV tuning + Win Shares ablation + calibration plots
6. **Temporal Trends** — OLS regression on season-level metrics (2011–2026)
7. **Exceptional Players** — weighted z-score composite using Cohen's d as weights
8. **Team Aggregation** — MP-weighted team profiles (964 team-seasons)
9. **Matchup Prediction** — pairwise classifier on 19,101 unique pairs with stacking ensemble + seed baseline
10. **2026 Holdout Evaluation** — AUC, accuracy, Brier score, log-loss, chalk/upset breakdown

## Project Structure

```
Project/
├── data/                    # Raw dataset (mm2026_train.csv) + patched (mm2026_updated.csv)
├── old/                     # Previous report iterations (reference only)
└── updated/                 # Production pipeline
    ├── pipeline.py          # Main orchestrator (runs all stages)
    ├── config.py            # Constants, paths, feature lists, 2026 results
    ├── utils.py             # Logging, plotting, metrics helpers
    ├── data_loader.py       # Stages 1–3
    ├── statistical_tests.py # Stage 4
    ├── classification.py    # Stage 5 (with hyperparameter tuning + calibration)
    ├── temporal_analysis.py # Stage 6
    ├── exceptional_players.py # Stage 7
    ├── matchup_predictor.py # Stages 8–9 (with stacking ensemble + seed baseline)
    ├── holdout_evaluation.py# Stage 10
    ├── report.tex           # LaTeX report source
    ├── report.pdf           # Compiled report (21 pages)
    └── outputs/             # All CSVs, figures, logs
```

## Running the Pipeline

Requires Python 3.10+.

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn xgboost lightgbm catboost
cd updated
python -X utf8 pipeline.py
```

Runtime: ~2 hours (mostly hyperparameter tuning on the matchup predictor).

## Data Sources

- **Kaggle Competition**: `mm2026_train.csv` — player-season records 2011–2026
- **2026 Results**: NCAA.com, ESPN, On3, Wikipedia (manually scraped, 921 rows patched)

## Key Techniques

- **Welch's t-test** with Bonferroni correction (25 tests, α = 0.002)
- **Cohen's d** effect sizes as impact-score weights
- **5-fold Stratified Cross-Validation** throughout
- **RandomizedSearchCV** (30 iter for Stage 5, 20 iter for Stage 9)
- **StackingClassifier** with LogisticRegression meta-learner
- **Brier score** + **log-loss** for probability calibration assessment
- **Minutes-played weighted aggregation** for team profiles
- **Pairwise matchup generation** with symmetric label duplication

## License

Academic project — for educational use only.
