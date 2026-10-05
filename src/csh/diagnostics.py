"""Shift diagnostics: the features the selector reads at deployment time.

Tier 1 -- label-free. Available the day the model arrives, computed from the
source data and the target's unlabelled pool.

Tier 2 -- label-assisted. Needs outcomes, so it only exists once some have
accumulated. Each Tier 2 feature carries an availability flag, because they
switch on at different points (an O/E count works earlier than a slope).

The label budget n is not a diagnostic: it is the constraint the selector
optimises under, so it sits in Tier 0 and is always available.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from scipy.stats import ks_2samp

from .config import SEED
from .metrics import clip
from .strategies import domain_weights, em_prior, fit_platt

TIER0 = ["t0_log_n", "t0_n_pos", "t0_n_neg"]
TIER1 = [
    "t1_c2st_auc", "t1_c2st_G1", "t1_c2st_G2", "t1_c2st_G3", "t1_c2st_G4",
    "t1_ess", "t1_log_prior_ratio", "t1_mean_risk_diff", "t1_risk_ks",
    "t1_top_smd", "t1_missing_shift", "t1_log_n_pool", "t1_em_bound",
]
TIER2 = [
    "t2_log_oe", "t2_oe_z", "t2_cal_slope", "t2_cal_slope_se",
    "t2_brier_surprise", "t2_auroc",
]
TIER2_FLAGS = [f"{c}_avail" for c in TIER2]
ALL_FEATURES = TIER0 + TIER1 + TIER2 + TIER2_FLAGS


def _auc(y, s) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def c2st(X_src, X_tgt, cols_idx=None, seed: int = SEED) -> float:
    """Classifier two-sample test: how separable are source and target?
    0.5 means indistinguishable. Cross-fitted, class-balanced."""
    from sklearn.model_selection import StratifiedKFold

    from .models import make_model

    if cols_idx is not None:
        X_src, X_tgt = X_src[:, cols_idx], X_tgt[:, cols_idx]
    n = min(len(X_src), len(X_tgt))
    if n < 30 or X_src.shape[1] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = X_src[rng.choice(len(X_src), n, replace=False)]
    b = X_tgt[rng.choice(len(X_tgt), n, replace=False)]
    X = np.vstack([a, b])
    d = np.r_[np.zeros(n), np.ones(n)]
    oof = np.zeros(len(d))
    skf = StratifiedKFold(n_splits=min(5, n), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, d):
        m = make_model("hgb", max_iter=80, max_leaf_nodes=15,
                       learning_rate=0.1, random_state=seed)
        m.fit(X[tr], d[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return _auc(d, oof)


def standardised_mean_diff(X_src, X_tgt, idx) -> float:
    if not len(idx):
        return float("nan")
    a, b = X_src[:, idx], X_tgt[:, idx]
    mu_a, mu_b = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    sd = np.sqrt((np.nanvar(a, axis=0) + np.nanvar(b, axis=0)) / 2)
    sd = np.where((sd == 0) | ~np.isfinite(sd), np.nan, sd)
    smd = np.abs(mu_a - mu_b) / sd
    return float(np.nanmean(smd)) if np.any(np.isfinite(smd)) else float("nan")


def missingness_shift(X_src, X_tgt) -> float:
    ms = np.isnan(X_src).mean(axis=0)
    mt = np.isnan(X_tgt).mean(axis=0)
    return float(np.mean(np.abs(ms - mt)))


def tier1(ctx, group_idx: dict[str, list[int]]) -> dict:
    """Label-free diagnostics. ctx is a strategies.Ctx."""
    p_src, p_pool = ctx.p("src"), ctx.p("pool")
    src_prior = float(np.mean(ctx.y_src))
    pi_t, em_bound = em_prior(p_pool, src_prior)
    w, ess = domain_weights(ctx.X_src, ctx.X_pool)

    out = {
        "t1_c2st_auc": c2st(ctx.X_src, ctx.X_pool),
        "t1_ess": float(ess),
        "t1_log_prior_ratio": float(np.log(max(pi_t, 1e-4) / max(src_prior, 1e-4))),
        "t1_mean_risk_diff": float(np.mean(p_pool) - np.mean(p_src)),
        "t1_risk_ks": float(ks_2samp(p_src, p_pool).statistic) if len(p_pool) else np.nan,
        "t1_top_smd": standardised_mean_diff(ctx.X_src, ctx.X_pool, ctx.top_idx),
        "t1_missing_shift": missingness_shift(ctx.X_src, ctx.X_pool),
        "t1_log_n_pool": float(np.log1p(len(ctx.X_pool))),
        "t1_em_bound": float(em_bound),
    }
    for g in ("G1", "G2", "G3", "G4"):
        out[f"t1_c2st_{g}"] = (c2st(ctx.X_src, ctx.X_pool, group_idx[g])
                               if g in group_idx else np.nan)
    return out


def tier2(ctx) -> dict:
    """Label-assisted diagnostics, with per-feature availability flags."""
    out = {c: 0.0 for c in TIER2}
    out.update({f: 0 for f in TIER2_FLAGS})
    y, p = np.asarray(ctx.y_lab, dtype=float), ctx.p("lab")
    n = len(y)
    if n == 0:
        return out

    obs, exp = float(y.sum()), float(np.sum(clip(p)))
    if exp > 0 and n >= 5:
        out["t2_log_oe"] = float(np.log(max(obs, 0.5) / exp))
        var = float(np.sum(clip(p) * (1 - clip(p))))
        out["t2_oe_z"] = float((obs - exp) / np.sqrt(var)) if var > 0 else 0.0
        out["t2_log_oe_avail"] = out["t2_oe_z_avail"] = 1

    pos = int(y.sum())
    if pos >= 5 and (n - pos) >= 5:
        a, b = fit_platt(y, p)
        out["t2_cal_slope"] = float(b)
        z = logit(clip(p))
        q = clip(1 / (1 + np.exp(-(a + b * z))))
        info = float(np.sum((z - z.mean()) ** 2 * q * (1 - q)))
        out["t2_cal_slope_se"] = float(1 / np.sqrt(info)) if info > 0 else 0.0
        out["t2_cal_slope_avail"] = out["t2_cal_slope_se_avail"] = 1

    if n >= 5:
        observed = float(np.mean((clip(p) - y) ** 2))
        expected = float(np.mean(clip(p) * (1 - clip(p))))
        out["t2_brier_surprise"] = observed - expected
        out["t2_brier_surprise_avail"] = 1

    if pos >= 3 and (n - pos) >= 3:
        out["t2_auroc"] = _auc(y, p)
        out["t2_auroc_avail"] = 1
    return out


def tier0(ctx) -> dict:
    y = np.asarray(ctx.y_lab)
    pos = int(y.sum()) if len(y) else 0
    return {
        "t0_log_n": float(np.log1p(len(y))),
        "t0_n_pos": float(pos),
        "t0_n_neg": float(len(y) - pos),
    }


def all_diagnostics(ctx, group_idx, tier1_cache: dict | None = None) -> dict:
    """Tier 1 depends only on the target pool, so it is computed once per
    target and reused across budgets and repeats."""
    t1 = tier1_cache if tier1_cache is not None else tier1(ctx, group_idx)
    return {**tier0(ctx), **t1, **tier2(ctx)}
