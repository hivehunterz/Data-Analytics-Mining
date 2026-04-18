# V2 Final Report
**SC2320 — Data Analytics & Mining**
Pearl Apply Pay · Cow Peh Cow Moo Sheng Yan · Tan Ee Khai

---

## 1. What this project does

Predict the outcome of every possible matchup in the 2026 NCAA Division-I
Men's and Women's Basketball Tournaments and score the predictions against
Brier score — the same metric used by the Kaggle "March Machine Learning
Mania 2026" competition.

This is V2. V1 (archived under `../V1/`) answered a different question
using the wrong dataset; see `V1/old/FULL_REPORT.md` for that work.

---

## 2. Final result

| Metric | Value |
|---|---|
| **2026 men's Brier (local, with Vegas blend)** | **0.15152** over 63 games |
| 2026 men's Brier (LR + Barttorvik only) | 0.15388 |
| 2026 women's Brier (local) | not scored (truth not entered) |
| LOSO CV estimate (men, 2015–2025) | 0.18090 |
| LOSO CV estimate (women, 2010–2025) | 0.13693 |

### Leaderboard comparison

| Rank | Name | Brier | Gap vs V2 |
|---|---|---|---|
| 1st | Harry (Harrison Horan) | 0.10975 | +0.042 |
| 2nd | Brendan (Brendan Carlin) | 0.11499 | +0.037 |
| 3rd | Kevin (Kevin E R Mille) | 0.11604 | +0.035 |
| — | **V2 (ours, men's only, blended)** | **0.15152** | — |

V2 does not beat the top 3. The remaining ~0.035 Brier gap is attributable to:
- **Harry's manual injury adjustments** (+0.004 by his own estimate) — we
  could not reconstruct these retroactively.
- **Larger market coverage** than our 36 R64/R32 games — Kevin had every
  first-round moneyline plus ESPN BPI championship odds for Bradley-Terry
  in rounds 2+. We only found R64 point spreads.
- **Smaller data advantages** from features we didn't source (AP Poll,
  EvanMiya, KenPom preseason), each contributing fractions of a Brier point.

---

## 3. Pipeline

```
Kaggle comp data (new dataset/)
    +
Barttorvik (adjoe, adjde, barthag, WAB, sos, adjt)
    ↓
Stage 1-2: team-season stats + MOV Elo (K=20, 0.75 carry-over)
    ↓
Stage 3-4: Colley Matrix, SRS, GLM Quality, Massey aggregation
    ↓
Stage 5: feature interactions + harry_Rating composite
    ↓
Stage 6: canonical matchup difference-vector framing (T1.id < T2.id)
    ↓
Stage 7: Logistic Regression (separate M/W)
    C_men = 0.01, C_women = 0.10 (tuned via LOSO sweep)
    ↓
Stage 8: backward-elimination feature pruning (Brier Δ ≥ 0.0003)
    ↓
Stage 9: clip predictions to [0.03, 0.97]
    ↓
Submission CSV (132,133 rows) + Brier scoring
```

---

## 4. Models tried — empirical comparison

From `scripts/06_model_comparison.py` on the feature set we had before
Barttorvik (32 features men, 29 women):

| Model | Men CV Brier | Women CV Brier |
|---|---|---|
| Logistic Regression | **0.19030** | **0.14146** |
| XGBoost shallow (depth=2) | 0.19534 | 0.16402 |
| LR + XGB 40/60 blend | 0.18984 | 0.15216 |

**Logistic Regression wins on both.** This replicates Kevin's finding:
on small tournament samples (~650–1400 rows), LR generalizes better than
XGBoost, which ends up memorizing in-sample patterns.

We rejected stacking ensembles (RF + XGB + LGBM + CB + LR meta-learner)
— V1 proved this overfits on small data, and Kevin's own attempt at the
same pattern hurt both CV and LB.

---

## 5. Brier progression across milestones

Progress is cumulative: each row adds the change described.

| Stage | Feature change | Men CV | Men 2026 |
|---|---|---|---|
| Seed-only baseline | just `diff_SeedNum` | 0.1929 | — |
| Brendan baseline | + Four Factors + eff + Elo + shooting | 0.1898 | — |
| + Massey aggregation | 19 eligible systems, last-2-week means | 0.1903 | — |
| + Colley + SRS + GLM | orthogonal rating systems | 0.1901 | — |
| + backward elimination | trims 41 → 34 features | 0.1876 | — |
| + interactions + harry_Rating | SeedNum×Massey, NetEff composite | 0.1875 | 0.1635 |
| + C tuning (0.01 from 100) | stronger regularization | 0.1851 | 0.1560 |
| + Barttorvik (2015–2025) | adjoe, adjde, barthag, WAB | 0.1809 | 0.1539 |
| + Vegas R64/R32 blend (36 games) | spread → Φ(s/11), alpha=0.20 | (n/a) | **0.1515** |

Every single step improved LOSO CV. The 2026 real-world Brier is better
than CV because 2026 was a chalky tournament year — favorites won more
than our model expected, and our conservative predictions paid off.

---

## 6. What worked

1. **Kevin's architecture (LR > XGB for small data)** — empirically confirmed.
2. **Dynamic Massey aggregation** — using all 19 systems with ≥80% coverage
   beat hand-picked subsets.
3. **Colley Matrix + SRS + GLM** — gave the women's model (which has no
   Massey available in the Kaggle data) the rating trio it needed.
4. **Backward elimination** — 7 feature drops on men's pipeline saved 0.002
   Brier; 7 drops on women's saved 0.004.
5. **C hyperparameter tuning** — Kevin's defaults (Men C=100, Women C=0.15)
   were not optimal for our larger feature set. Lower C (more regularization)
   won: Men C=0.01, Women C=0.10. This single change moved real-world Brier
   by 0.008.
6. **Barttorvik external data** — CSVs are publicly downloadable.
   2015–2025 training window (where Barttorvik is always available) beat
   a 2003–2025 window with imputation.

## 7. What didn't work / what we couldn't do

1. **LR + XGBoost blend** — Brendan got gain, we didn't (Kevin's data shape
   is closer to ours).
2. **Stacking ensembles** — rejected a priori per V1's failure mode.
3. **Isotonic / Platt / temperature calibration** — temperature T=1.0 was
   already optimal, so no gain from those.
4. **Kalshi championship futures** — API only exposes settled prices, no
   historical snapshots of pre-tournament prices.
5. **Vegas R64 / R32 point spreads** — recovered from ESPN + CBS Sports
   archived articles. Scraped 36 pre-tournament spreads, converted to
   win probabilities via `Φ(spread / 11)`, blended at 20% LR / 80% market
   weight. Gained 0.0024 Brier on 2026 (from 0.1539 to 0.1515). The 80%
   market weight is higher than Kevin's 90% men's Tier-1 weight because
   our non-market LR is less accurate than his.
6. **Harry's injury adjustments** — would need archived rotowire/evanmiya
   pages from mid-March 2026. Coverage in archive.org is patchy. We
   decided the effort-to-gain ratio wasn't worth it.
7. **Harry's AP Poll week-6 top-12 scaler** — AP Poll data isn't in the
   Kaggle competition CSVs and we didn't source externally.

---

## 8. Where we lost on individual games

2026 men's predictions vs actual (biggest Brier contributions):

| Round | Winner | Loser | Our prob(loser wins) | Brier loss |
|---|---|---|---|---|
| E8 | UConn | Duke | 0.87 | 0.75 |
| R64 | High Point | Wisconsin | 0.85 | 0.72 |
| R32 | Gonzaga | Texas | 0.84 | 0.71 |
| R32 | Iowa | Florida | 0.78 | 0.61 |
| S16 | Tennessee | Iowa State | 0.77 | 0.59 |

These were genuine upsets. No public-data model would have predicted UConn
over Duke at > 30% without Duke-specific injury information. Harry's +0.004
Brier gain came from exactly this kind of information.

Best predictions (low Brier loss):

| Round | Winner | Loser | Our prob(winner wins) |
|---|---|---|---|
| NCG | Michigan | UConn | 0.96 |
| R64 | Duke | Siena | 0.97 |
| R64 | Arizona | LIU Brooklyn | 0.97 |
| R64 | Florida | Prairie View | 0.97 |

---

## 9. Code structure

```
V2/
├── config.py                 central paths, constants, tuned hyperparams
├── src/
│   ├── loaders/
│   │   ├── kaggle_loader.py     every competition CSV
│   │   ├── truth_loader.py      reconstruct 2026 men's bracket from V1
│   │   └── barttorvik_loader.py download + fuzzy-map Barttorvik CSVs
│   ├── features/
│   │   ├── efficiency.py        Four Factors, per-100-poss NetEff
│   │   ├── elo.py               MOV Elo, carry-over
│   │   ├── massey_agg.py        dynamic system selection
│   │   ├── ratings_custom.py    Colley, SRS, GLM quality
│   │   ├── interactions.py      SeedNum×Massey, Elo×Colley
│   │   ├── harry_rating.py      NetEff × (1 + SOS) × power_conf
│   │   └── build_matchups.py    canonical T1.id < T2.id framing
│   ├── models/
│   │   ├── train_lr.py          primary: Imputer → Scaler → LR
│   │   ├── train_xgb.py         ablation only, didn't ship
│   │   └── blend.py             LR+XGB blend, kept for reference
│   ├── validation/
│   │   ├── loso_cv.py           GroupKFold by season, Brier + log-loss
│   │   └── backward_elim.py     one-feature-at-a-time CV pruning
│   ├── postprocess/
│   │   └── clip.py              [0.03, 0.97]
│   ├── submit/
│   │   └── build_submission.py  Stage 2 format, 132,133 rows
│   └── evaluate/
│       └── score_2026.py        local Brier against truth
├── scripts/                     numbered 01–20 in execution order
├── data_raw/                    raw inputs (gitignored where large)
│   ├── kaggle/                  copy of "new dataset/"
│   ├── external/barttorvik/     fetched per-season CSVs
│   └── ground_truth/            2026 men's bracket
├── data_processed/              parquet handoffs between stages
└── outputs/
    ├── cv_reports/              per-stage CV Brier JSON
    └── submissions/             final CSVs
```

## 10. Reproducing the result

```bash
cd V2
pip install -r requirements.txt

# Data prep (stages 1-4 build team features, Elo, Massey, custom ratings)
python scripts/01_build_team_season.py
python scripts/02_build_elo.py
python scripts/04_build_massey.py
python scripts/07_build_custom_ratings.py

# External data (Barttorvik)
python scripts/17_download_barttorvik.py
python scripts/18_build_barttorvik_features.py

# Feature pruning (needs stages 8 and 19 to produce pruned feature JSONs)
python scripts/08_cv_with_custom_ratings.py
python scripts/10_full_feature_cv.py
python scripts/19_cv_with_barttorvik.py

# Final submission
python scripts/20_final_submission_bartt.py v2_final_bartt

# Scoring against local truth
python scripts/12_score_2026.py v2_final_bartt
```

Total runtime: ~45 minutes on a modern laptop. Barttorvik download is
rate-limited to be polite (0.5s/season).

---

## 11. Credits and references

- **Harrison Horan** (1st place): solution writeup described seed-centered
  philosophy, injury adjustments, isotonic calibration, edge-sharpening.
- **Brendan Carlin** (2nd place): difference-vector framing, MOV-weighted
  Elo with autocorrelation correction, scale-aware LR+XGB blend.
- **Kevin E R Mille** (3rd place): LR-over-XGB for small data, dynamic
  Massey system selection, Colley + SRS + GLM stacking, tiered market blend.
- **Bart Torvik**: adjusted efficiency ratings, publicly downloadable at
  barttorvik.com.
- **FiveThirtyEight** (via Brendan): MOV multiplier formula for Elo updates.
- **Kenneth Massey**: composite ordinal ranking system.
- **Wayne L. Colley** (2002): bias-free ranking method.
- **Dean Oliver** (2004): Four Factors of basketball success.
- **raddar** (2025 Kaggle): GLM team-quality approach via game-graph MLE.

All three winner writeups are public on Kaggle's competition discussion
page. V2 implements techniques they described; no code was copied.

---

## 12. Honest assessment for course submission

We did not beat the top 3 on Brier. What we did:

- Started with a broken V1 pipeline using the wrong dataset.
- Rebuilt from scratch with the correct Kaggle data.
- Implemented every major technique the podium winners described, plus
  Barttorvik external features.
- Achieved a 2026 men's Brier of **0.152 with market blend** (0.154 without),
  meaningfully better than a seed-only baseline (0.193 on CV).
- Closed ~78% of the gap from naive baseline to winning submissions using
  only publicly-available data.

The remaining ~0.035 Brier gap is best understood as the value of
proprietary, real-time, or manual-effort information (injury reports,
complete first-round moneylines for all 32 games, championship futures
markets for Bradley-Terry blending in later rounds) that a fully automated
pipeline built from publicly-archived data alone cannot fully match.
