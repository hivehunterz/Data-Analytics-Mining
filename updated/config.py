"""
Shared configuration: paths, feature lists, 2026 results, constants.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────
DATA_RAW      = os.path.join(os.path.dirname(__file__), "..", "data", "mm2026_train.csv")
DATA_UPDATED  = os.path.join(os.path.dirname(__file__), "..", "data", "mm2026_updated.csv")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs")
LOG_FILE      = os.path.join(OUTPUT_DIR, "pipeline_report.txt")

RANDOM_STATE = 42
CV_FOLDS     = 5
HOLDOUT_YEAR = 2026
MIN_GP       = 5
MIN_MPG      = 5

# ── Feature groups ────────────────────────────────────────────────────
PLAYER_ONLY_COLS = [
    "pts_per_game","trb_per_game","ast_per_game","stl_per_game",
    "blk_per_game","tov_per_game","fg_pct","three_p_pct","ft_pct",
    "efg_pct","ts_pct","three_par","ftr","per","orb_pct","drb_pct",
    "trb_pct","ast_pct","stl_pct","blk_pct","tov_pct","usg_pct",
    "ows","dws","ws","ws_40","obpm","dbpm","bpm",
    "minutes_per_game","starter_ratio",
    "scoring_efficiency","playmaking_index","defensive_impact",
    "rebounding_rate","usage_efficiency","win_shares_per_game",
]

# BPM-family only (no Win Shares) – for ablation
BPM_ONLY_COLS = [c for c in PLAYER_ONLY_COLS if c not in ("ows","dws","ws","ws_40","win_shares_per_game")]

TEAM_COLS     = ["team_srs","team_sos","team_win_pct","tournament_seed"]
COMBINED_COLS = PLAYER_ONLY_COLS + TEAM_COLS

# ── 2026 manual results ──────────────────────────────────────────────
RESULTS_2026 = {
    "Michigan":                (6, "Champion"),
    "Connecticut":             (5, "NCG"),
    "Arizona":                 (4, "F4"),
    "Illinois":                (4, "F4"),
    "Duke":                    (3, "E8"),
    "Purdue":                  (3, "E8"),
    "Tennessee":               (3, "E8"),
    "Iowa":                    (3, "E8"),
    "St. John's (NY)":         (2, "S16"),
    "Michigan State":          (2, "S16"),
    "Nebraska":                (2, "S16"),
    "Houston":                 (2, "S16"),
    "Arkansas":                (2, "S16"),
    "Alabama":                 (2, "S16"),
    "Iowa State":              (2, "S16"),
    "Texas":                   (3, "S16"),
    "Texas Christian":         (1, "R32"),
    "Kansas":                  (1, "R32"),
    "Louisville":              (1, "R32"),
    "UCLA":                    (1, "R32"),
    "Florida":                 (1, "R32"),
    "Vanderbilt":              (1, "R32"),
    "Virginia Commonwealth":   (1, "R32"),
    "Texas A&M":               (1, "R32"),
    "Utah State":              (1, "R32"),
    "High Point":              (1, "R32"),
    "Gonzaga":                 (1, "R32"),
    "Miami (FL)":              (1, "R32"),
    "Saint Louis":             (1, "R32"),
    "Kentucky":                (1, "R32"),
    "Virginia":                (1, "R32"),
    "Texas Tech":              (1, "R32"),
    "Siena":                   (0, "R64"),
    "Long Island University":  (0, "R64"),
    "Ohio State":              (0, "R64"),
    "Northern Iowa":           (0, "R64"),
    "California Baptist":      (0, "R64"),
    "South Florida":           (0, "R64"),
    "North Dakota State":      (0, "R64"),
    "UCF":                     (0, "R64"),
    "Furman":                  (0, "R64"),
    "Prairie View A&M":        (1, "R64"),
    "Clemson":                 (0, "R64"),
    "McNeese":                 (0, "R64"),
    "Troy":                    (0, "R64"),
    "North Carolina":          (0, "R64"),
    "Pennsylvania":            (0, "R64"),
    "Saint Mary's":            (0, "R64"),
    "Idaho":                   (0, "R64"),
    "Villanova":               (0, "R64"),
    "Wisconsin":               (0, "R64"),
    "Hawaii":                  (0, "R64"),
    "Brigham Young":           (0, "R64"),
    "Kennesaw State":          (0, "R64"),
    "Missouri":                (0, "R64"),
    "Queens (NC)":             (0, "R64"),
    "Howard":                  (1, "R64"),
    "Georgia":                 (0, "R64"),
    "Akron":                   (0, "R64"),
    "Hofstra":                 (0, "R64"),
    "Wright State":            (0, "R64"),
    "Miami (OH)":              (1, "R64"),
    "Santa Clara":             (0, "R64"),
    "Tennessee State":         (0, "R64"),
    "NC State":                (0, "R68"),
    "Southern Methodist":      (0, "R68"),
    "Lehigh":                  (0, "R68"),
    "Maryland-Baltimore County": (0, "R68"),
}

RESULT_ORDER = {"R68":0,"R64":1,"R32":2,"S16":3,"E8":4,"F4":5,"NCG":6,"Champion":7}

# Columns to drop during cleaning
DROP_COLS = ["class_year","__index_level_0__","sources","team_slug",
             "missing_fields","data_quality_flag","split","include_in_training"]

SHOOT_COLS = ["fg_pct","three_p_pct","ft_pct","efg_pct","ts_pct"]
ADV_COLS   = ["per","orb_pct","drb_pct","trb_pct","ast_pct","stl_pct","blk_pct",
              "tov_pct","usg_pct","ows","dws","ws","ws_40","obpm","dbpm","bpm"]
