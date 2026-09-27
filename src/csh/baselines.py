"""The two principled ways people already choose an update, both Tier 2.

TCV  -- cross-validate the candidate updates on the site's own labels.
CTP  -- the closed testing procedure of Vergouwe et al. (2017), generalised
        from logistic models to any base model by testing the nested
        recalibration models fitted on the source model's logit.

Both are honest competitors: they use exactly the information a real site has.
Neither can see the evaluation half.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold

from .config import EPS_TIE, SEED
from .metrics import brier, clip
from .strategies import (COST_RANK, LADDER, NEEDS_LABELS, Ctx, fit_intercept,
                         fit_platt, fit_revision, run_all, _standardise)

LABEL_FREE = [k for k in LADDER if not NEEDS_LABELS[k]]


def _sub_ctx(ctx: Ctx, lab_idx, eval_X) -> Ctx:
    """A copy of the context with a smaller labelled set and a new eval set.
    The S4 model is carried over: it is label-free, so it does not change."""
    cache = {}
    if "S4_model" in ctx.cache:
        cache["S4_model"] = ctx.cache["S4_model"]
    return Ctx(model=ctx.model, kind=ctx.kind, X_src=ctx.X_src, y_src=ctx.y_src,
               X_pool=ctx.X_pool, X_lab=ctx.X_lab[lab_idx],
               y_lab=ctx.y_lab[lab_idx], X_eval=eval_X, top_idx=ctx.top_idx,
               cache=cache)


def tcv_pick(ctx: Ctx, preds_full: dict[str, np.ndarray] | None = None,
             n_splits: int = 3, seed: int = SEED,
             eps: float = EPS_TIE) -> tuple[str, dict]:
    """Pick by cross-validation on the n labelled target cases."""
    y = np.asarray(ctx.y_lab)
    n = len(y)
    pos = int(y.sum())
    if n < 10 or pos < 3 or (n - pos) < 3:
        return "S0", {}

    # label-free strategies need no CV: score them directly on all n cases
    scores: dict[str, float] = {}
    lf_ctx = _sub_ctx(ctx, np.arange(n), ctx.X_lab)
    lf_preds = run_all(lf_ctx)
    for k in LABEL_FREE:
        scores[k] = brier(y, lf_preds[k])

    folds = min(n_splits, pos, n - pos)
    if folds >= 2:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        acc: dict[str, list[float]] = {k: [] for k in LADDER
                                       if NEEDS_LABELS[k]}
        for tr, te in skf.split(np.zeros(n), y):
            sub = _sub_ctx(ctx, tr, ctx.X_lab[te])
            p = run_all(sub)
            for k in acc:
                acc[k].append(brier(y[te], p[k]))
        for k, v in acc.items():
            scores[k] = float(np.mean(v))

    best = min(scores, key=scores.get)
    cheap = [k for k in scores if scores[k] <= scores[best] + eps]
    return min(cheap, key=lambda k: COST_RANK[k]), scores


def _loglik(y, p) -> float:
    p = clip(p)
    y = np.asarray(y, dtype=float)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ctp_pick(ctx: Ctx, alpha: float = 0.05) -> tuple[str, dict]:
    """Closed testing: start from the most flexible update, then step down to
    the simplest model that is not significantly worse."""
    y = np.asarray(ctx.y_lab, dtype=float)
    n = len(y)
    pos = int(y.sum())
    if n < 10 or pos < 3 or (n - pos) < 3:
        return "S0", {}

    p0 = ctx.p("lab")
    z = logit(clip(p0))

    ll = {"S0": _loglik(y, p0)}
    a = fit_intercept(y, p0)
    ll["S2"] = _loglik(y, expit(a + z))
    a3, b3 = fit_platt(y, p0)
    ll["S3"] = _loglik(y, expit(a3 + b3 * z))

    df_extra = {"S0": 0, "S2": 1, "S3": 2}
    if ctx.top_idx and pos >= 10 and (n - pos) >= 10:
        Zl, _ = _standardise(ctx.X_lab[:, ctx.top_idx], ctx.X_lab[:, ctx.top_idx])
        ar, br, c = fit_revision(y, p0, Zl)
        ll["S5"] = _loglik(y, expit(ar + br * z + np.nan_to_num(Zl, nan=0.0) @ c))
        df_extra["S5"] = 2 + len(ctx.top_idx)
    full = "S5" if "S5" in ll else "S3"

    def not_worse(simple: str) -> bool:
        stat = 2 * (ll[full] - ll[simple])
        dof = max(df_extra[full] - df_extra[simple], 1)
        return bool(chi2.sf(max(stat, 0.0), dof) > alpha)

    for cand in ("S0", "S2", "S3"):
        if cand == full:
            break
        if not_worse(cand):
            return cand, ll
    return full, ll
