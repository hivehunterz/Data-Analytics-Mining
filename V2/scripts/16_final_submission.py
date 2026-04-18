"""
Stage 16: Final submission with all improvements applied.

Pipeline:
  1. LR trained on full feature set (pruned via backward elim)
  2. C tuned: Men=0.01, Women=0.10
  3. Elo logistic probability blended at alpha=0.9 (90% LR, 10% Elo)
  4. Clipped to [0.03, 0.97]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from config import (PROCESSED, CV_REPORTS, SUBMISSIONS,
                     MEN_SEASONS_FULL, WOMEN_SEASONS_FULL,
                     LR_C_MEN, LR_C_WOMEN, MEN_ID_CUTOFF,
                     CLIP_LO, CLIP_HI, HOLDOUT_SEASON)
from src.loaders import kaggle_loader as kl
from src.features.build_matchups import build_tourney_matchups, build_submission_matchups
from src.features.interactions import add_interactions_men, add_interactions_women
from src.features.harry_rating import compute_opp_quality_pts_won, build_harry_rating
from src.models.train_lr import make_lr_pipeline
from src.postprocess.clip import clip_predictions
from src.utils import log


ELO_BLEND_ALPHA = 0.9   # weight on LR (1 - alpha on Elo)


def _elo_prob(elo_a: np.ndarray, elo_b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def _load_pruned_features(gender: str) -> list:
    with open(CV_REPORTS / f"{gender}_full_pruned.json") as f:
        payload = json.load(f)
    feats = [c.replace("diff_", "") for c in payload["final_features"]]
    return [f for f in feats if f != "SeedNum"]


def _assemble_men():
    m_team = pd.read_parquet(PROCESSED / "m_team_season.parquet")
    m_elo  = pd.read_parquet(PROCESSED / "m_elo.parquet")
    m_cust = pd.read_parquet(PROCESSED / "m_ratings_custom.parquet")
    m_mass = pd.read_parquet(PROCESSED / "m_massey_agg.parquet")
    m_seeds = kl.load_m_seeds()
    m_confs = kl.load_m_conferences()
    m_secondary = kl._read("MSecondaryTourneyTeams.csv")
    m_reg = kl.load_m_regular_compact()
    m_opq = compute_opp_quality_pts_won(m_reg, m_seeds, m_secondary)
    m_team_hr = build_harry_rating(m_team, m_opq, m_confs, is_men=True)
    m_inter = add_interactions_men(m_team_hr, m_mass, m_cust, m_seeds)
    m_feat = m_inter.merge(m_elo, on=["Season", "TeamID"], how="left")
    return m_feat, m_seeds, m_elo


def _assemble_women():
    w_team = pd.read_parquet(PROCESSED / "w_team_season.parquet")
    w_elo  = pd.read_parquet(PROCESSED / "w_elo.parquet")
    w_cust = pd.read_parquet(PROCESSED / "w_ratings_custom.parquet")
    w_seeds = kl.load_w_seeds()
    w_confs = kl.load_w_conferences()
    w_secondary = kl._read("WSecondaryTourneyTeams.csv")
    w_reg = kl.load_w_regular_compact()
    w_opq = compute_opp_quality_pts_won(w_reg, w_seeds, w_secondary)
    w_team_hr = build_harry_rating(w_team, w_opq, w_confs, is_men=False)
    w_inter = add_interactions_women(w_team_hr, w_cust, w_elo, w_seeds)
    w_feat = w_inter.merge(w_elo[["Season", "TeamID", "EloSlope"]],
                           on=["Season", "TeamID"], how="left")
    return w_feat, w_seeds, w_elo


def _predict(feat, seeds, tourney_loader, elo_df, seasons, features, C, sub):
    tourney = tourney_loader()
    tourney_in = tourney[tourney["Season"].isin(seasons)]
    train_m = build_tourney_matchups(tourney_in, feat, features, seeds_df=seeds)
    diff_cols = [f"diff_{c}" for c in features] + ["diff_SeedNum"]
    X_train = train_m[diff_cols].values
    y_train = train_m["label"].values
    model = make_lr_pipeline(C=C).fit(X_train, y_train)

    sub_m = build_submission_matchups(sub, feat, features, seeds_df=seeds)
    X_sub = sub_m[diff_cols].values
    p_lr = model.predict_proba(X_sub)[:, 1]

    # Elo probabilities
    elo_map = dict(zip(zip(elo_df["Season"], elo_df["TeamID"]), elo_df["Elo"]))
    e_t1 = sub_m.apply(lambda r: elo_map.get((HOLDOUT_SEASON, r["T1"]), 1500), axis=1).values
    e_t2 = sub_m.apply(lambda r: elo_map.get((HOLDOUT_SEASON, r["T2"]), 1500), axis=1).values
    p_elo = _elo_prob(e_t1, e_t2)

    # Blend
    blended = ELO_BLEND_ALPHA * p_lr + (1 - ELO_BLEND_ALPHA) * p_elo
    clipped = clip_predictions(blended, CLIP_LO, CLIP_HI)
    sub_m["Pred"] = clipped
    return sub_m


def main(tag: str = "v2_final_blend"):
    log("loading submission template...")
    sub = kl.load_sample_submission_stage2()
    sub_m = sub[sub["T1"] < MEN_ID_CUTOFF].copy()
    sub_w = sub[sub["T1"] >= MEN_ID_CUTOFF].copy()

    log("\n=== MEN ===")
    m_feat, m_seeds, m_elo = _assemble_men()
    m_features = _load_pruned_features("men")
    pred_m = _predict(m_feat, m_seeds, kl.load_m_tourney_compact, m_elo,
                       MEN_SEASONS_FULL, m_features, LR_C_MEN, sub_m)
    log(f"  men predictions: {len(pred_m)}")

    log("\n=== WOMEN ===")
    w_feat, w_seeds, w_elo = _assemble_women()
    w_features = _load_pruned_features("women")
    pred_w = _predict(w_feat, w_seeds, kl.load_w_tourney_compact, w_elo,
                       WOMEN_SEASONS_FULL, w_features, LR_C_WOMEN, sub_w)
    log(f"  women predictions: {len(pred_w)}")

    all_sub = pd.concat([pred_m, pred_w], ignore_index=True)
    out = all_sub[["ID", "Pred"]]
    out_path = SUBMISSIONS / f"{tag}_submission.csv"
    out.to_csv(out_path, index=False)
    log(f"\nsaved {out_path} with {len(out)} rows")
    return out_path


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_final_blend"
    main(tag)
