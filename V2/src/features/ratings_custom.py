"""
Custom rating systems that provide orthogonal signal to Elo + Massey.

Kevin's insight: Colley Matrix + SRS + GLM quality are each derived from
independent models of team strength, so together they give the LR model
more uncorrelated features to discriminate on.

Colley Matrix (bias-free):
    Solves (2I + C) r = 1 + (w - l)/2
    where C is the connectivity matrix from regular-season W/L graph.

SRS (Simple Rating System):
    rating_i = avg_margin_i + mean(opponent_ratings_j)
    Iterate to convergence (tolerance 1e-6).

GLM Quality (Raddar-style):
    MLE team strength where each game outcome is a Bernoulli trial with
    P(i beats j) = sigmoid(strength_i - strength_j).
    Use sklearn LogisticRegression with one-hot indicators for each team.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LogisticRegression


# ── Colley Matrix ─────────────────────────────────────────────────────
def compute_colley(compact: pd.DataFrame) -> pd.DataFrame:
    """
    Run Colley per season. Returns (Season, TeamID, ColleyRating).

    Higher rating = better. Range approximately [0, 1] with mean 0.5.
    """
    out = []
    for season, games in compact.groupby("Season"):
        teams = sorted(set(games["WTeamID"]).union(games["LTeamID"]))
        tid_to_i = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        wins = np.zeros(n)
        losses = np.zeros(n)
        # Dense because n ~ 350 is small
        C = np.zeros((n, n))
        for _, g in games.iterrows():
            w, l = tid_to_i[g["WTeamID"]], tid_to_i[g["LTeamID"]]
            wins[w] += 1
            losses[l] += 1
            C[w, w] += 1
            C[l, l] += 1
            C[w, l] -= 1
            C[l, w] -= 1

        # Colley: (2I + games_played) r = 1 + (wins - losses)/2
        A = 2 * np.eye(n) + C
        b = 1 + (wins - losses) / 2
        r = np.linalg.solve(A, b)
        for t, rating in zip(teams, r):
            out.append({"Season": season, "TeamID": t, "ColleyRating": float(rating)})
    return pd.DataFrame(out)


# ── SRS ───────────────────────────────────────────────────────────────
def compute_srs(
    compact: pd.DataFrame,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> pd.DataFrame:
    """
    Simple Rating System: rating = avg_margin + mean(opp_ratings).
    """
    out = []
    for season, games in compact.groupby("Season"):
        teams = sorted(set(games["WTeamID"]).union(games["LTeamID"]))
        tid_to_i = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        margins = np.zeros(n)
        opp_lists = [[] for _ in range(n)]
        game_count = np.zeros(n)
        for _, g in games.iterrows():
            w, l = tid_to_i[g["WTeamID"]], tid_to_i[g["LTeamID"]]
            m = g["WScore"] - g["LScore"]
            margins[w] += m
            margins[l] -= m
            game_count[w] += 1
            game_count[l] += 1
            opp_lists[w].append(l)
            opp_lists[l].append(w)
        avg_margin = margins / np.maximum(game_count, 1)

        ratings = avg_margin.copy()
        for _ in range(max_iter):
            new_ratings = avg_margin + np.array([
                np.mean([ratings[o] for o in opp_lists[i]]) if opp_lists[i] else 0
                for i in range(n)
            ])
            # Mean-center so ratings have mean 0
            new_ratings -= new_ratings.mean()
            if np.max(np.abs(new_ratings - ratings)) < tol:
                ratings = new_ratings
                break
            ratings = new_ratings

        for t, rating in zip(teams, ratings):
            out.append({"Season": season, "TeamID": t, "SRS": float(rating)})
    return pd.DataFrame(out)


# ── GLM Quality (Raddar) ──────────────────────────────────────────────
def compute_glm_quality(compact: pd.DataFrame, C_reg: float = 1.0) -> pd.DataFrame:
    """
    Logistic-regression team strength from game outcomes.

    For each game (i beats j), the model tries to maximize P(i wins):
        P(i beats j) = sigmoid(strength_i - strength_j)
    Equivalently, encode each game as a feature row with +1 for winner
    and -1 for loser, with constant label y=1, and fit LR without intercept.

    To prevent overfitting, we use L2 regularization (C=1 by default).
    """
    out = []
    for season, games in compact.groupby("Season"):
        teams = sorted(set(games["WTeamID"]).union(games["LTeamID"]))
        tid_to_i = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        n_games = len(games)

        # Build sparse design matrix: each game = row with +1 at winner, -1 at loser.
        # Include both (winner, loser, y=1) and (loser, winner, y=0) for symmetry.
        rows = []
        cols = []
        data = []
        y = np.zeros(2 * n_games)
        for k, (_, g) in enumerate(games.iterrows()):
            w, l = tid_to_i[g["WTeamID"]], tid_to_i[g["LTeamID"]]
            # winner perspective: y=1
            rows.append(2*k);   cols.append(w); data.append(+1.0)
            rows.append(2*k);   cols.append(l); data.append(-1.0)
            y[2*k] = 1.0
            # loser perspective: y=0
            rows.append(2*k+1); cols.append(w); data.append(-1.0)
            rows.append(2*k+1); cols.append(l); data.append(+1.0)
            y[2*k+1] = 0.0

        X = csr_matrix((data, (rows, cols)), shape=(2 * n_games, n))
        lr = LogisticRegression(
            penalty="l2", C=C_reg, fit_intercept=False,
            solver="liblinear", max_iter=1000,
        )
        lr.fit(X, y)
        strength = lr.coef_[0]
        for t, s in zip(teams, strength):
            out.append({"Season": season, "TeamID": t, "GLMQuality": float(s)})
    return pd.DataFrame(out)
