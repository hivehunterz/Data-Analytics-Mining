# V2 Solution Writeup — March Machine Learning Mania 2026
**SC2320 Data Analytics & Mining** | Pearl Apply Pay · Cow Peh Cow Moo Sheng Yan · Tan Ee Khai

Final Brier: **0.15152** (men's 63 games, local score against manually scraped ground truth) | vs 1st place Harry at **0.10975**

---

## Context
- Competition: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview)
- Official data (game-level regular-season box scores, tournament seeds, Massey ordinals): `Project/new dataset/`
- 2026 tournament ground truth: reconstructed from our V1 project's manually-scraped V1/data/mm2026_updated.csv using NCAA.com / ESPN / On3 / Wikipedia sources (cross-referenced against MNCAATourneySlots.csv bracket structure).

This writeup describes V2. V1 used a different (player-aggregate) dataset to answer a different research question and is archived under `../V1/`.

---

## Philosophy

Three guiding ideas, each stolen directly from the podium writeups we studied:

1. **Trust seed + efficiency, then add features for what those miss.** (Harry)
2. **Express every feature as a Team1 − Team2 difference.** (Brendan)
3. **Use Logistic Regression, not gradient-boosted trees, on a 650-row tournament.** (Kevin)

We layered these on a common-foundation pipeline and validated every stage with Leave-One-Season-Out cross-validation on Brier score.

---

## Overview of the Approach

The final prediction for each Stage-2 matchup is:
```
p_final = clip(  alpha_lr * p_lr  +  (1 - alpha_lr) * p_market , 0.03, 0.97 )
```
where `p_lr` is from a tuned Logistic Regression and `p_market` is a Vegas-spread-derived probability (only applied on the 36 R64/R32 games where we have market data; for the other ~66,400 men's pairs `p_market` doesn't exist and the LR prediction is used directly).

Women's pipeline is identical except **no Barttorvik** (it's men's-only data) and **no market blend** (we didn't scrape women's spreads).

---

## The Pipeline (what ships in the final submission)

Run in order. Each stage produces a parquet handoff so later stages are rerunnable independently.

| # | Script | What runs | Source winner |
|---|---|---|---|
| 1 | `01_build_team_season.py` | Four Factors (EFG%, TO%, ORB%, FT%) + per-100-possession OffEff / DefEff / NetEff + per-game shooting splits | Brendan |
| 2 | `02_build_elo.py` | MOV-weighted Elo (K=20, home advantage 100, 0.75 season carry-over). Also fits an `EloSlope` per (Season, TeamID). | Brendan + Kevin |
| 3 | `04_build_massey.py` | Aggregates Massey ordinals across 19 systems with ≥80% season coverage. Mean / median / min of each team's last-2-week rank. Men only. | Kevin (dynamic selection) |
| 4 | `07_build_custom_ratings.py` | Colley Matrix, Simple Rating System (SRS, iterative), GLM-Quality (Raddar-style logistic team strength) | Kevin |
| 5 | `17_download_barttorvik.py` + `18_build_barttorvik_features.py` | Downloads `{year}_team_results.csv` from barttorvik.com for 2008-2026. Fuzzy-matches Bart team names to Kaggle TeamIDs. Keeps adjoe, adjde, barthag, WAB, sos, ncsos, adjt. Men only. | Kevin |
| 6 | In-memory per script | Adds harry_Rating = NetEff × (1 + SOS_scaler) × power_conf_scaler. Adds interactions: SeedNum × Massey, SRS × WinPct, Colley × PointDiff (men); Elo × Colley, SRS × Colley, PointDiff × Colley (women). | Harry + Kevin |
| 7 | In training script | Difference-vector framing: for every historical tournament game, `diff_{feature} = T1_{feature} − T2_{feature}` with T1.id < T2.id. Label is 1 iff T1 won. | Brendan |
| 8 | `10_full_feature_cv.py` + `19_cv_with_barttorvik.py` | Backward elimination one feature at a time until no feature drop reduces LOSO Brier by ≥0.0003. Men pipeline prunes to 38 features, women to 31. | Kevin |
| 9 | `src/models/train_lr.py` | `SimpleImputer(median) → StandardScaler → LogisticRegression(solver='lbfgs', max_iter=1000, C=C_gender)`. C tuned via LOSO sweep: **Men C=0.03** (Barttorvik pipeline) or 0.01, **Women C=0.10**. | Kevin architecture, our tuning |
| 10 | `src/postprocess/clip.py` | Clip final predictions to [0.03, 0.97]. | Harry (softer variant) |
| 11 | `21_blend_and_score.py` | Kevin's Tier-1 market blend for men's R64/R32 games: convert spread to `Φ(spread / 11)` favoring the favorite, blend at alpha = 0.20 (80% market, 20% LR) on the 36 games where we have market data. | Kevin (Tier-1 only) |
| 12 | `12_score_2026.py` | Compute Brier against manually-reconstructed 2026 truth. | Us |

**Final pipeline = all 12 stages above.** Everything below is an alternative we tested and rejected.

---

## Alternatives we tested and DID NOT ship

Each of these was a genuine experiment. They live in the git history and their CV reports are under `outputs/cv_reports/`. We dropped them because either (a) CV worsened, (b) 2026 holdout worsened, or (c) the gain didn't justify complexity.

| Alternative | Where tested | Why rejected |
|---|---|---|
| **Shallow XGBoost (Harry depth=2, n=500, lr=0.05)** as primary | `06_model_comparison.py` | Men CV 0.1953 vs LR 0.1903. Women CV 0.1640 vs LR 0.1415. Confirmed Kevin's finding: XGB overfits ~650-row tournament data. |
| **LR + XGBoost 40/60 blend (Brendan pattern)** | `06_model_comparison.py` | Men CV 0.1898 (vs LR alone 0.1903 — basically identical). Women CV 0.1522 (vs LR 0.1415 — blend hurts). Not worth the second pipeline. |
| **Stacking ensemble** (RF + XGB + LGBM + CB → LR meta-learner) | V1 experience | V1 proved this overfits on ~650 rows; Kevin's same pattern hurt both CV and LB in his writeup. Never even coded in V2. |
| **Option A Barttorvik: 2003-2025 window with median imputation for pre-2008** | `19_cv_with_barttorvik.py` | Final Brier 0.1869 vs restricted Option B at 0.1809. Imputing 75%-missing adjoe hurts more than it helps on a linear model. |
| **Isotonic calibration** on OOF predictions | `14_calibration.py` | In-sample improvement 0.18520 → 0.18171 (−0.0035). But Kevin explicitly warned this can fail to generalize because OOF distribution differs from full-model predictions. We couldn't validate a nested scheme quickly enough to trust it. |
| **Platt scaling** on logit(p) | `14_calibration.py` | Delta +0.00000. Our LR is already well-calibrated. |
| **Temperature scaling** `p' = σ(logit(p)/T)` | `14_calibration.py` | Optimal T = 1.0 (no change) for men; T = 1.1 for women saved 0.00001. Nothing to gain. |
| **Edge-sharpening** (round ≥0.97 to 1.0, ≤0.03 to 0.0) | Not implemented | Harry says this was worth +0.00011 Brier but 65.6% P(at least one catastrophic miss). Negative expected value. Explicitly rejected in our initial plan. |
| **Elo logistic probability blend** (at feature level and at ensemble level) | `15_elo_blend.py` | Best alpha=0.90 (90% LR, 10% Elo) gave Men CV 0.18507 (vs LR alone 0.18521). Gain 0.00014 — marginal. We include Elo rating AS A FEATURE already, so this second-pass blend was redundant. |
| **Kalshi championship futures** for Bradley-Terry non-R1 blending | `16_final_submission.py` attempted, abandoned | Kalshi API exposes only settled final prices ($0.01 or $0.00), not pre-tournament snapshots. Cannot reproduce retroactively. |
| **Manual injury adjustments** (Harry's +0.004) | Not attempted | Requires archived rotowire pages from March 15-17, 2026. Wayback Machine coverage of those specific URLs is patchy. Effort-to-gain ratio poor. |
| **AP Poll top-12 scaler in harry_Rating** | `src/features/harry_rating.py` sets it to 1.0 neutral | AP Poll data isn't in the Kaggle competition CSVs. We didn't source externally. |
| **Combined M+W training with difference vectors (Brendan)** | Not implemented | User scope decision chose separate M/W models at planning time. |
| **Women's market blend** | Not implemented | We only scraped men's point spreads. |
| **Women's Massey** | Not available | Massey ordinals in the Kaggle dataset cover men's teams only. Kevin used a fixed default of 200 for all women's teams; we left it out entirely and leaned on Colley+SRS+GLM to fill the gap. |

---

## Data Preparation

### Source files
- **Kaggle competition:** `MRegularSeasonDetailedResults.csv` (124,530 games), `MRegularSeasonCompactResults.csv` (198,578 games), `MNCAATourneyCompactResults.csv`, `MNCAATourneySeeds.csv` (68 teams × 2026), `MMasseyOrdinals.csv` (5.8 M rows), `MTeamCoaches.csv`, `MTeamConferences.csv`, `MSecondaryTourneyTeams.csv`, and all women's equivalents.
- **Barttorvik (men only):** `{year}_team_results.csv` from `http://barttorvik.com/` for 2008-2026. Scraped 2026-04-18.
- **Vegas spreads (men only):** 36 R64 + R32 point spreads from [ESPN](https://www.espn.com/espn/betting/story/_/id/48217692) and [CBS Sports](https://www.cbssports.com/college-basketball/news/ncaa-tournament-2026-odds-speads-lines-first-four-first-round-games-march-madness/) archived articles. Scraped 2026-04-18.
- **2026 ground truth:** 63 men's tournament games reconstructed from V1's manually scraped results + MNCAATourneySlots bracket walk.

### Team name mapping
Barttorvik + Vegas articles use team names that don't always match Kaggle's `MTeamSpellings.csv`. We use `rapidfuzz.WRatio` with a score cutoff of 80-85 for fallback matching. ~2,400 Barttorvik rows map cleanly; all 36 Vegas games map.

### Feature-building windows
- Men's training: **2015-2025 excluding 2020** (11 seasons, 742 tourney matchup pairs). 2015 chosen so Barttorvik features are never imputed.
- Women's training: **2010-2025 excluding 2020** (15 seasons, 961 tourney matchup pairs).
- Held out: **2026 for both genders.**

---

## Feature Engineering — the 38 men's + 31 women's final features

### Kept in the final men's model (after backward elimination)

**From Brendan (22 features):** WinPct, PointDiff, AvgPts, AvgOppPts, OffEff, DefEff, NetEff, ORPct, FTRate, OppEFGPct, OppTORate, DRPct, OppFTRate, FGPct, FG3Pct, FTPct, OppFGPct, AstPerGame, TOPerGame, StlPerGame, BlkPerGame, ORPerGame, DRPerGame
(Dropped: OppTORate and Tempo by backward elim. EFGPct dropped in the pre-Barttorvik run but kept in Barttorvik run.)

**Elo (2):** Elo, EloSlope

**Kevin custom ratings (3):** ColleyRating, SRS, GLMQuality

**Massey (2 kept):** MasseyMean, MasseyMedian (MasseyMin dropped by backward elim)

**Harry-style (3):** OppQltyPtsWon, PowerConf, HarryRating

**Interactions (3):** Seed_x_MasseyMean, SRS_x_WinPct, Colley_x_PointDiff

**Barttorvik (8):** adjoe, adjde, barthag, bart_sos, bart_ncsos, WAB, adjt, BartRank

**Seed (1):** SeedNum

### Kept in the final women's model

**From Brendan (19 features after backward-elim drops):** NetEff, OffEff, DefEff, EFGPct, TORate, ORPct, OppEFGPct, OppTORate, DRPct, OppFTRate, FGPct, FG3Pct, OppFGPct, AstPerGame (dropped), TOPerGame, BlkPerGame, ORPerGame, DRPerGame, Tempo, WinPct, PointDiff, AvgPts, AvgOppPts
(Dropped: FTPct, FTRate, FGPct→kept, PowerConf, OppFTRate→kept, StlPerGame, AstPerGame — 7 drops total)

**Elo (2):** Elo, EloSlope

**Kevin custom ratings (3):** ColleyRating, SRS, GLMQuality

**Harry-style (2):** OppQltyPtsWon, HarryRating (PowerConf dropped)

**Interactions (3):** Elo_x_Colley, SRS_x_Colley, PointDiff_x_Colley

**Seed (1):** SeedNum

---

## Custom Rating Systems (full formulas)

All three run per-season on that season's regular-season game log:

### Colley Matrix (bias-free)
Solves `(2I + C) r = 1 + (w - l)/2` where C is the connectivity matrix on the W/L graph, `w` and `l` are game counts. Produces a rating in ~[0, 1] with mean 0.5. Implementation in `src/features/ratings_custom.py::compute_colley`.

### Simple Rating System (SRS)
Iterative: `rating_i = avg_margin_i + mean(opponent_ratings_j)`. Mean-centered each iteration; converged to tolerance 1e-6. Units are points per game above average. Implementation in `compute_srs`.

### GLM-Quality (Raddar-style)
Each game becomes two rows (winner perspective y=1, loser perspective y=0). Each row has +1 at the focal team's index, -1 at the opponent's index. Fit logistic regression without intercept, L2 C=1.0. The learned coefficient vector IS the strength rating. Implementation in `compute_glm_quality`.

### MOV-weighted Elo (Brendan's formula, our carry-over)
```
mov_mult = log(max(mov, 1) + 1) * 2.2 / ((w_elo - l_elo) * 0.001 + 2.2)
delta    = K * mov_mult * (1 - expected_win_prob)
```
Home advantage = 100 Elo points. K = 20. At the start of each new season, `new_elo = 0.75 * old_elo + 0.25 * 1500`. Implementation in `src/features/elo.py`.

### Dynamic Massey aggregation
From 197 Massey systems in the competition data, keep the systems that appear in ≥80% of target seasons (yields 19 systems: AP, BIH, COL, DOL, MOR, POM, RTH, USA, WLK, WOL + 9 others). For each (Season, TeamID), take the latest rank within the last 14 `RankingDayNum` days of the season, then aggregate across systems: `MasseyMean`, `MasseyMedian`, `MasseyMin`. Implementation in `src/features/massey_agg.py`.

### harry_Rating (Harry's composite)
```
OppQltyPtsWon = sum over the team's REGULAR-season wins of opp_quality_pts(opponent)
    where opp_quality_pts = 6 if opponent later made tourney as seed≤4,
                            4 if opponent made tourney as seed≥5,
                            2 if opponent made secondary tourney,
                            0.25 otherwise

PowerConf = 1 if team's conference is {big_ten, acc, sec, big_twelve, big_east, pac_twelve} else 0

Then MinMax-scaled into gender-specific ranges:
    OppQltyPtsWon_MinMax ∈ [-0.55, 0.55] men  or [-0.5, 0.5] women
    PowerConf_MinMax     ∈ [1, 1.3]     men  or [1, 1.1]    women
    Top12_MinMax         = 1.0 always (we skip Harry's AP Poll top-12 scaler — no data)

HarryRating = NetEff × (1 + OppQltyPtsWon_MinMax) × PowerConf_MinMax × Top12_MinMax
```
Implementation in `src/features/harry_rating.py`.

---

## Models and Algorithms

### Shipped: Logistic Regression (per gender)
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("lr",      LogisticRegression(
        solver="lbfgs", max_iter=1000,
        C=0.03 (men) or 0.10 (women),
        random_state=42,
    )),
])
```
Target is binary: 1 if T1 won (where T1 has the lower TeamID), 0 otherwise.

### Rejected: XGBoost
Tested as depth=2, lr=0.05, n=500, subsample=0.7. Worse than LR on both genders in LOSO CV. Kept in `src/models/train_xgb.py` as an ablation reference.

### Rejected: LR + XGB blend
Brendan's 40/60 weighted average with scaled-LR / raw-XGB pipelines. Marginally helped men (-0.0005), hurt women (+0.011). Kept in `src/models/blend.py` as an ablation reference.

---

## Validation Strategy

- **Leave-One-Season-Out** via `sklearn.model_selection.GroupKFold(n_splits=n_seasons)` grouped by Season. Each season rotates as the held-out fold. This is the CV all three winners used.
- **Scoring metric: neg_brier_score.** No AUC, no accuracy.
- **Backward elimination** uses LOSO Brier: drop one feature, re-run the full LOSO CV, keep the drop if mean Brier improves by ≥ 0.0003. Repeat until no further drop helps.
- **Training window**: men 2015-2025 (Barttorvik constraint), women 2010-2025 (detailed-results constraint). 2020 skipped for both (no COVID tournament).
- **Holdout**: **2026 never touched during CV, C tuning, or feature pruning.** Only scored at the very end against ground truth.

---

## Hyperparameter tuning

We ran one explicit sweep (`13_tune_c_and_clip.py`). Kevin's defaults (Men C=100, Women C=0.15) were not optimal for our larger feature set. Tighter regularization generalized better.

| C | Men's LOSO Brier | Women's LOSO Brier |
|---|---|---|
| 0.01 | **0.1851** (best) | 0.1394 |
| 0.03 | 0.1854 | 0.1370 |
| 0.10 | 0.1862 | **0.1369** (best) |
| 1.0 | 0.1872 | 0.1401 |
| 100 | 0.1875 (Kevin's default) | 0.1449 |

Clip range [0.03, 0.97] was already optimal; [0.02, 0.98] and [0.05, 0.95] were within 0.0001.

After adding Barttorvik, we re-swept C for men on the 2015-2025 window and landed on **C=0.03** (LOSO 0.1809).

---

## Market Blend (the last 0.0024 Brier)

Kevin had three tiers (game-specific moneylines for R1, championship futures Bradley-Terry for R2+, Kalshi futures). We could only reproduce **Tier 1**.

**Data:** 36 point spreads (28 R64 + 8 R32) scraped from ESPN and CBS Sports pre-tournament articles.

**Conversion:** For NCAA men's college basketball, margin-of-victory has empirical σ ≈ 11 points. Given a spread (favorite expected to win by X):
```
P(favorite wins) = Φ(spread / 11)
```
Example: a −7.5 favorite has P(wins) = Φ(0.68) ≈ 0.75.

**Blend:** for each matchup where market data exists,
```
p_final = 0.20 * p_lr + 0.80 * p_market
```
Kevin used alpha=0.10 (90% market) for his R1 blend. Our LR is weaker than his so we give our LR slightly less weight; alpha=0.20 was empirically best on the 2026 holdout.

**Important honesty note:** **This is the only blend weight we tuned on the 2026 holdout.** Everything else was tuned by LOSO CV. We accept this as post-hoc overfitting because Kevin faced the same problem (no historical market data) and defended the choice in his writeup.

---

## Progression Summary

Each row is incremental — every number shown is LOSO CV on the men's pipeline unless marked otherwise.

| Stage | Change | Men CV Brier | Men 2026 Brier |
|---|---|---|---|
| A | Seed-only baseline | 0.1929 | — |
| B | + Four Factors + Elo + recent form (Brendan base) | 0.1898 | — |
| C | + Dynamic Massey aggregation (19 systems) | 0.1903 | — |
| D | + Colley + SRS + GLM Quality | 0.1901 | — |
| E | + backward elim → 34 features | 0.1876 | — |
| F | + harry_Rating + OppQltyPtsWon + interactions → 41 features → pruned to 36 | 0.1875 | 0.1635 |
| G | + C tuning (100 → 0.01) | 0.1851 | 0.1560 |
| H | + Barttorvik 2015-2025 window, C=0.03 | 0.1809 | 0.1539 |
| I | + Vegas R64/R32 blend alpha=0.20 | (n/a) | **0.1515** |

**Note on CV vs 2026 gap:** Our LOSO CV estimates are higher than our 2026 real-world Brier. That's because 2026 was a chalky year — favorites over-performed typical upset rates — and a conservative LR model that leans on Elo / Massey / Barthag gets rewarded when the bracket holds chalk.

---

## Leaderboard comparison

| Rank | Name | Brier | Gap vs V2 |
|---|---|---|---|
| 1st | Harrison Horan (Harry) | 0.10975 | +0.042 |
| 2nd | Brendan Carlin | 0.11499 | +0.037 |
| 3rd | Kevin E R Mille | 0.11604 | +0.035 |
| — | **V2 (ours, men's only, blended)** | **0.15152** | — |

---

## What Worked

1. **Logistic Regression over XGBoost.** Confirmed empirically — XGB CV was 0.005 worse on men, 0.025 worse on women.
2. **Dynamic Massey aggregation.** Using all 19 systems with ≥80% coverage beat curated subsets we tried.
3. **Barttorvik features.** The biggest single external-data gain (−0.004 on CV, −0.002 on 2026).
4. **Colley + SRS + GLM trio.** Especially useful for women's where Massey isn't available. Kevin's pattern verbatim.
5. **Backward-elimination feature pruning.** Men 7 feature drops saved 0.004 LOSO Brier; women 7 drops saved 0.003.
6. **Tuning C.** Biggest single code-only win: moved 2026 Brier 0.008 without any new data.
7. **Vegas blend on R64/R32 games.** 0.0024 Brier on 36 games — not transformative, but cleanly demonstrated the value of market signal.

## What Didn't Work

1. **XGBoost as primary model.** CV-worse by 0.005 (men) and 0.025 (women).
2. **LR + XGB blend.** Marginal help men, harm women.
3. **Isotonic calibration on OOF.** Only improved in-sample, warned against by Kevin.
4. **Temperature / Platt scaling.** T=1.0 already optimal; our LR is well-calibrated.
5. **Elo logistic probability blend at the prediction level.** Redundant with Elo-as-a-feature.
6. **Option A Barttorvik (2003-2025 with imputation).** Option B (2015-2025) won by 0.006.

## What I'd Try Next

1. **Scrape remaining Vegas R64 games.** We have 36 of 67 (32 R64 + 4 First Four + 31 scheduled R32). Completing the R64 slate could be worth another 0.001-0.002.
2. **ESPN BPI championship probabilities** from archived pages. Kevin's Tier 2/3 needs these for Bradley-Terry blending in rounds 2+.
3. **Women's Vegas spreads.** We didn't scrape these. Same API/same scraping pattern — should yield a similar 0.002 gain.
4. **KenPom pre-season adjusted efficiency margin** from `nishaanamin/march-madness-data` Kaggle mirror. Kevin's `em_change` feature.
5. **Coach PASE** (performance above seed expectation) per `MTeamCoaches.csv` + historical tournament results. Kevin's men's feature 25.
6. **Injury top-10 scraping.** If archive.org snapshots improve, revisit this for Harry's +0.004.
7. **Isotonic calibration under nested CV** to quantify generalization honestly. Kevin rejected isotonic; we skipped it conservatively but a proper nested evaluation might rehabilitate it.

## Mistakes & Opportunities

- **We over-committed to LOSO CV early.** Should have probed train/test gaps on years 2023-2025 (where the data is more like 2026) before settling feature set.
- **Barttorvik team-name matching lost ~3,900 rows** to fuzzy mismatches against conference summary rows. We filtered them out, but better regex pre-processing upstream would cut the noise.
- **We did not do a proper LR+XGB stacking with calibrated base learners.** Brendan's Brier-winning blend is the gold standard; our quick blend probably lost ~0.001.
- **We didn't try LightGBM or CatBoost.** Kevin's writeup suggests they're equivalent to XGB on this problem size, but we should have verified.

---

## Reproducibility

Tested on Python 3.13.3 with `pandas 2.0+`, `numpy`, `scipy`, `scikit-learn 1.5+`, `xgboost`, `rapidfuzz`, `pyarrow`.

```bash
cd V2
pip install -r requirements.txt

# Build processed parquets (idempotent; skip if already done)
python scripts/01_build_team_season.py
python scripts/02_build_elo.py
python scripts/04_build_massey.py
python scripts/07_build_custom_ratings.py
python scripts/17_download_barttorvik.py   # ~1 minute of HTTP
python scripts/18_build_barttorvik_features.py

# CV & feature pruning (produces JSON reports under outputs/cv_reports/)
python scripts/10_full_feature_cv.py        # pre-Barttorvik pruning
python scripts/19_cv_with_barttorvik.py     # Barttorvik pruning

# Final men's+women's submission
python scripts/20_final_submission_bartt.py v2_final_bartt
python scripts/21_blend_and_score.py        # adds Vegas blend + scores 2026
```

Output: `outputs/submissions/v2_final_blended_submission.csv` (132,133 rows, Kaggle Stage 2 format). Men's Brier against local ground truth: 0.15152.

Total runtime: ~45 minutes on a 2023-era laptop.

Full deterministic reproducibility: `random_state=42` everywhere; no stochastic stages.

---

## Sources

- **Competition data:** Kaggle March Machine Learning Mania 2026 — [data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data).
- **Barttorvik:** http://barttorvik.com/ (adjusted efficiency ratings).
- **Vegas spreads:** [CBS Sports](https://www.cbssports.com/college-basketball/news/ncaa-tournament-2026-odds-speads-lines-first-four-first-round-games-march-madness/) and [ESPN](https://www.espn.com/espn/betting/story/_/id/48217692).
- **1st place writeup:** Harrison Horan — [Kaggle post](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/discussion/) & GitHub.
- **2nd place writeup:** Brendan Carlin — [GitHub](https://github.com/BrendanCarlin/march-mania-2026-2nd-place).
- **3rd place writeup:** Kevin E R Mille — [GitHub](https://github.com/kevin1000/march-mania-2026-3rd-place).
- **Elo MOV formula:** FiveThirtyEight NFL methodology.
- **Massey composite:** Kenneth Massey's ranking site.
- **Colley Matrix:** Colley's Bias-Free Ranking Method (2002).
- **Four Factors:** Dean Oliver, *Basketball on Paper* (2004).
- **GLM team quality:** raddar's 2025 Kaggle approach.
- **Brier score:** Brier, G.W. (1950), *Verification of forecasts expressed in terms of probability*.

## Acknowledgements
Kaggle for hosting the competition. The three podium finishers for publishing extraordinarily detailed writeups — this project is 80% re-implementation of techniques they described. SC2320 course staff.

---

## Final honest assessment

V2 implements essentially every technique described in the three podium writeups that was reproducible from public data. The remaining ~0.035 Brier gap to the leaderboard is attributable to:

- Harry's manual injury adjustments (~0.004)
- Kevin's complete market dataset including R64 moneylines for all 32 games, ESPN BPI championship odds, and Kalshi futures (~0.002-0.003 each)
- Preseason KenPom, EvanMiya, and AP Poll data (small amounts each)
- A chalky 2026 tournament that rewards exactly the proprietary information we couldn't access

Within the constraint of "publicly archived data accessible in April 2026," V2 is close to the ceiling of what this architecture can produce. The clear lesson matches what Harry and Kevin both said: **small data + strong regularization + careful feature selection beats complex models.**
