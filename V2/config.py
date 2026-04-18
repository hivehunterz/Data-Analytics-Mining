"""
Central configuration: paths, seeds, season windows.

All downstream modules import from here. Do NOT duplicate constants.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent          # .../Project/V2
PROJECT_ROOT = ROOT.parent                       # .../Project

# Raw data
KAGGLE_DIR    = PROJECT_ROOT / "new dataset"     # official competition data
V1_DATA_DIR   = PROJECT_ROOT / "V1" / "data"     # for 2026 men's ground truth

DATA_RAW      = ROOT / "data_raw"
EXTERNAL_DIR  = DATA_RAW / "external"
BARTTORVIK    = EXTERNAL_DIR / "barttorvik"
NISHAANAMIN   = EXTERNAL_DIR / "nishaanamin"
COACHES_DIR   = EXTERNAL_DIR / "coaches"
INJURIES_DIR  = EXTERNAL_DIR / "injuries"
MARKETS_DIR   = EXTERNAL_DIR / "markets"
TRUTH_DIR     = DATA_RAW / "ground_truth"

# Processed (parquet handoffs)
PROCESSED     = ROOT / "data_processed"

# Outputs
OUTPUTS       = ROOT / "outputs"
CV_REPORTS    = OUTPUTS / "cv_reports"
IMPORTANCE    = OUTPUTS / "feature_importance"
CALIBRATION   = OUTPUTS / "calibration_plots"
SUBMISSIONS   = OUTPUTS / "submissions"
LOGS          = OUTPUTS / "logs"

for _d in (DATA_RAW, EXTERNAL_DIR, BARTTORVIK, NISHAANAMIN, COACHES_DIR,
           INJURIES_DIR, MARKETS_DIR, TRUTH_DIR, PROCESSED, OUTPUTS,
           CV_REPORTS, IMPORTANCE, CALIBRATION, SUBMISSIONS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


# ── Constants ─────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Training windows (excluding 2020 — no COVID tournament)
MEN_SEASONS_FULL     = [y for y in range(2003, 2026) if y != 2020]
WOMEN_SEASONS_FULL   = [y for y in range(2010, 2026) if y != 2020]
# Barttorvik fields only exist 2008+ (some 2015+); if using them, restrict
MEN_SEASONS_BARTT    = [y for y in range(2015, 2026) if y != 2020]

HOLDOUT_SEASON = 2026

# TeamID convention
MEN_ID_CUTOFF   = 3000

# Elo params (Kevin + Brendan blend)
ELO_INITIAL        = 1500.0
ELO_K              = 20
ELO_HOME_ADV       = 100
ELO_MEAN_REVERSION = 0.75    # carry-over each season

# Model hyperparameters — tuned via LOSO CV sweep (scripts/13_tune_c_and_clip.py).
# Our feature set is larger and more correlated than Kevin's, so stronger
# regularization (lower C) generalizes better.
#   Men:   C=0.01  (Kevin used 100; we sweep preferred 0.01, Brier 0.1851)
#   Women: C=0.10  (Kevin used 0.15; we sweep preferred 0.10,  Brier 0.1370)
LR_C_MEN      = 0.01
LR_C_WOMEN    = 0.10

# Clipping
CLIP_LO, CLIP_HI = 0.03, 0.97
