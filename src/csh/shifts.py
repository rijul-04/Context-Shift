"""Injected shifts: pseudo-deployments with a known cause.

Two uses. They enlarge the selector's meta-training set beyond the handful of
real domains, and they give the diagnostics a ground truth to be scored
against (H7: does the grouped C2ST name the right kind of change?).

Each generator takes the pooled source rows of a fold and returns the indices
of a pseudo-target plus a label vector, so the shift is real rather than
simulated at the feature level only.
"""
from __future__ import annotations

import numpy as np

from .config import SEED

SHIFT_TYPES = ["covariate", "prior", "observation", "concept", "none"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def covariate_shift(X, y, idx, rng, severity: float, feat: int):
    """Select patients with probability tilted by one feature (age, severity)."""
    v = X[idx, feat].astype(float)
    ok = np.isfinite(v)
    z = np.zeros(len(idx))
    if ok.sum() > 10:
        mu, sd = np.nanmean(v), np.nanstd(v)
        sd = sd if sd > 0 else 1.0
        z = np.nan_to_num((v - mu) / sd, nan=0.0)
    p = _sigmoid(severity * z)
    keep = idx[rng.random(len(idx)) < p]
    return keep, y[keep], X[keep], {}


def prior_shift(X, y, idx, rng, severity: float, feat: int = 0):
    """Resample to change the outcome prevalence by a factor."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    if len(pos) < 10 or len(neg) < 10:
        return idx, y[idx], X[idx], {}
    if severity >= 1:                       # more events
        n_neg = max(10, int(len(neg) / severity))
        keep = np.r_[pos, rng.choice(neg, n_neg, replace=False)]
    else:                                   # fewer events
        n_pos = max(5, int(len(pos) * severity))
        keep = np.r_[rng.choice(pos, n_pos, replace=False), neg]
    keep = np.sort(keep)
    return keep, y[keep], X[keep], {}


def observation_shift(X, y, idx, rng, severity: float, g2_idx=None,
                      g3_count_idx=None):
    """Delete a share of measured values, as a site that measures less often.
    G3 count columns are scaled to match, and missingness flags follow."""
    Xt = X[idx].copy()
    cols = g2_idx if g2_idx is not None and len(g2_idx) else range(Xt.shape[1])
    mask = rng.random((len(idx), len(list(cols)))) < severity
    cols = list(cols)
    sub = Xt[:, cols]
    sub[mask] = np.nan
    Xt[:, cols] = sub
    if g3_count_idx is not None:
        for j in g3_count_idx:
            Xt[:, j] = Xt[:, j] * (1 - severity)
    return idx, y[idx], Xt, {}


def concept_shift(X, y, idx, rng, severity: float, feat: int):
    """Change P(y|x) in one subgroup: above the median of a top feature, flip
    a share of outcomes. The features are untouched, so only the rule moves."""
    v = X[idx, feat].astype(float)
    med = np.nanmedian(v)
    grp = np.nan_to_num(v, nan=med) > med
    yt = y[idx].copy()
    flip = grp & (rng.random(len(idx)) < severity)
    yt[flip] = 1 - yt[flip]
    return idx, yt, X[idx], {}


def make_injected(X, y, src_idx, rng, kind: str, severity: float,
                  top_idx, g2_idx, g3_count_idx):
    """Return (X_target, y_target, meta) for one injected pseudo-target."""
    feat = int(top_idx[0]) if len(top_idx) else 0
    if kind == "covariate":
        _, yt, Xt, _ = covariate_shift(X, y, src_idx, rng, severity, feat)
    elif kind == "prior":
        _, yt, Xt, _ = prior_shift(X, y, src_idx, rng, severity)
    elif kind == "observation":
        _, yt, Xt, _ = observation_shift(X, y, src_idx, rng, severity,
                                         g2_idx, g3_count_idx)
    elif kind == "concept":
        _, yt, Xt, _ = concept_shift(X, y, src_idx, rng, severity, feat)
    else:
        Xt, yt = X[src_idx], y[src_idx]
    return Xt, yt, {"shift_type": kind, "severity": severity}


def expected_group(kind: str) -> str:
    """Which feature group the diagnostics should flag for each shift type."""
    return {"covariate": "G1/G2", "prior": "none", "observation": "G3",
            "concept": "none", "none": "none"}[kind]


def severities(kind: str) -> list[float]:
    return {
        "covariate": [1.0, 2.0, 4.0],
        "prior": [0.5, 1.5, 2.5],
        "observation": [0.3, 0.5, 0.7],
        "concept": [0.10, 0.20, 0.30],
        "none": [0.0],
    }[kind]


def default_rng(seed: int = SEED):
    return np.random.default_rng(seed)
