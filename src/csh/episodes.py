"""Deployment episodes: the core experiment.

One episode = (dataset, target domain, base model, label budget n, repeat r).

For each episode we draw n labelled cases from the target's adaptation half,
compute the Tier 0/1/2 diagnostics, run all eight update strategies, and score
every one of them on the target's evaluation half. The result is a meta-row:
diagnostics in, a Brier score per strategy out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BUDGETS, N_REPEATS, SEED
from .baselines import ctp_pick, tcv_pick
from .diagnostics import all_diagnostics, tier1
from .metrics import brier, ece, oe_ratio
from .models import fit, make_model, predict_proba, top_features, tune
from .splits import domain_split
from .strategies import Ctx, LADDER, run_all


def budgets_for(n_adapt: int, budgets=None) -> list[int]:
    """Budgets a domain can actually support (at most half its adaptation half)."""
    budgets = budgets or BUDGETS
    cap = max(0, n_adapt)
    return [b for b in budgets if b <= cap]


def draw_labelled(y, adapt_idx, n, rng):
    if n == 0:
        return np.array([], dtype=int)
    n = min(n, len(adapt_idx))
    return rng.choice(adapt_idx, size=n, replace=False)


def run_target(X, y, dom, target, kind, group_idx, dataset, cols,
               n_repeats: int = N_REPEATS, budgets=None, seed: int = SEED,
               source_mask=None, tag: str = "real", params=None):
    """All episodes for one target domain. Returns a list of meta-rows."""
    tgt_idx = np.where(dom == target)[0]
    src_mask = (dom != target) if source_mask is None else source_mask
    Xs, ys = X[src_mask], y[src_mask]
    if len(np.unique(ys)) < 2 or len(tgt_idx) < 40:
        return []

    params = params if params is not None else tune(kind, Xs, ys, dom[src_mask])
    model = fit(make_model(kind, **params), Xs, ys)
    top_idx = top_features(model, Xs, cols, k=10)

    adapt, evalu = domain_split(y, tgt_idx, seed=seed)
    if len(np.unique(y[evalu])) < 2:
        return []

    # Tier 1 depends only on the unlabelled pool, so compute it once.
    base_ctx = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys,
                   X_pool=X[adapt], X_lab=X[adapt][:0], y_lab=y[adapt][:0],
                   X_eval=X[evalu], top_idx=top_idx)
    t1 = tier1(base_ctx, group_idx)
    s4_cache = {}

    rows = []
    for n in budgets_for(len(adapt), budgets):
        reps = 1 if n == 0 else n_repeats
        for r in range(reps):
            rng = np.random.default_rng(hash((target, kind, n, r, seed)) % 2**32)
            lab = draw_labelled(y, adapt, n, rng)
            ctx = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys,
                      X_pool=X[adapt], X_lab=X[lab], y_lab=y[lab],
                      X_eval=X[evalu], top_idx=top_idx, cache=dict(s4_cache))
            preds = run_all(ctx)
            s4_cache.setdefault("S4_model", ctx.cache.get("S4_model"))

            diag = all_diagnostics(ctx, group_idx, tier1_cache=t1)
            # the two Tier 2 competitors, decided inside the episode
            tcv, _ = tcv_pick(ctx)
            ctp, _ = ctp_pick(ctx)
            row = {
                "dataset": dataset, "target": target, "model": kind,
                "n": n, "rep": r, "tag": tag,
                "n_eval": len(evalu), "events_eval": int(y[evalu].sum()),
                "prev_eval": float(y[evalu].mean()),
                "pick_tcv": tcv, "pick_ctp": ctp,
                **diag,
            }
            for k in LADDER:
                row[f"brier_{k}"] = brier(y[evalu], preds[k])
            row["ece_S0"] = ece(y[evalu], preds["S0"])
            row["oe_S0"] = oe_ratio(y[evalu], preds["S0"])
            best = min(LADDER, key=lambda k: row[f"brier_{k}"])
            row["oracle"] = best
            row["brier_oracle"] = row[f"brier_{best}"]
            rows.append(row)
    return rows


def to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)
