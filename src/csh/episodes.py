"""Deployment episodes: the core experiment.

One episode = (dataset, target domain, base model, label budget n, repeat r).

The site labels n of its own patients, drawn at random from everyone it has,
and every strategy is scored on all of its remaining patients. Scoring on
everyone left rather than on a fixed half roughly doubles the evaluation set
and removes an arbitrary split; within an episode all eight strategies see
exactly the same patients, which is what regret compares.

Label-free strategies and the Tier 1 diagnostics read the site's unlabelled
records, including those later scored. That is transductive, and it is also
the real situation: a hospital holds its own records on day one and only
lacks the outcomes.

Two oracles are recorded. The plain oracle is the strategy that turned out
best on the evaluation patients, which is what the literature reports and
which flatters itself: with a few hundred patients, some of its advantage is
luck. The honest oracle picks the strategy on one half of the evaluation set
and is scored on the other, both ways round, so it cannot profit from noise
it could not have predicted. The gap between the two measures how much of the
"available gain" is real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import ctp_pick, tcv_pick
from .config import BUDGETS, N_REPEATS, SEED
from .diagnostics import all_diagnostics, tier1
from .metrics import brier, ece, oe_ratio
from .models import fit, make_model, predict_proba, top_features, tune
from .strategies import Ctx, LADDER, run_all

MIN_EVAL = 40          # below this an episode cannot be scored meaningfully


def budgets_for(n_target: int, budgets=None) -> list[int]:
    """Budgets a site can support while leaving enough patients to score on."""
    budgets = budgets or BUDGETS
    return [b for b in budgets if b <= max(0, n_target - MIN_EVAL)]


def draw_labelled(target_idx, n, rng):
    """The n patients whose outcomes the site has reviewed."""
    if n == 0:
        return np.array([], dtype=int)
    n = min(n, len(target_idx))
    return np.sort(rng.choice(target_idx, size=n, replace=False))


def honest_oracle(y_eval, preds: dict[str, np.ndarray], rng) -> tuple[float, str]:
    """Oracle that must choose on data it is not scored on.

    Split the evaluation patients in two, pick the best strategy on each half,
    score that choice on the other half, average the two. Any part of the
    plain oracle's advantage that is sampling noise disappears here.
    """
    n = len(y_eval)
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    if len(a) < 10 or len(b) < 10:
        return float("nan"), ""
    scores = []
    picks = []
    for fit_half, score_half in ((a, b), (b, a)):
        if len(np.unique(y_eval[fit_half])) < 2:
            continue
        best = min(LADDER, key=lambda k: brier(y_eval[fit_half], preds[k][fit_half]))
        picks.append(best)
        scores.append(brier(y_eval[score_half], preds[best][score_half]))
    if not scores:
        return float("nan"), ""
    return float(np.mean(scores)), "|".join(picks)


def run_target(X, y, dom, target, kind, group_idx, dataset, cols,
               n_repeats: int = N_REPEATS, budgets=None, seed: int = SEED,
               source_mask=None, tag: str = "real", params=None):
    """All episodes for one target domain. Returns a list of meta-rows."""
    tgt_idx = np.where(dom == target)[0]
    src_mask = (dom != target) if source_mask is None else source_mask
    Xs, ys = X[src_mask], y[src_mask]
    if len(np.unique(ys)) < 2 or len(tgt_idx) < MIN_EVAL:
        return []

    params = params if params is not None else tune(kind, Xs, ys, dom[src_mask])
    model = fit(make_model(kind, **params), Xs, ys)
    top_idx = top_features(model, Xs, cols, k=10)

    # The site's unlabelled records: everything it has. Fixed across episodes,
    # so the Tier 1 diagnostics and the S4 refit are computed once.
    X_pool = X[tgt_idx]
    base_ctx = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys, X_pool=X_pool,
                   X_lab=X_pool[:0], y_lab=y[tgt_idx][:0], X_eval=X_pool,
                   top_idx=top_idx)
    t1 = tier1(base_ctx, group_idx)
    s4_cache: dict = {}

    rows = []
    for n in budgets_for(len(tgt_idx), budgets):
        for r in range(1 if n == 0 else n_repeats):
            rng = np.random.default_rng(
                abs(hash((dataset, target, kind, n, r, seed))) % 2**32)
            lab = draw_labelled(tgt_idx, n, rng)
            evalu = np.setdiff1d(tgt_idx, lab, assume_unique=True)
            if len(evalu) < MIN_EVAL or len(np.unique(y[evalu])) < 2:
                continue

            ctx = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys, X_pool=X_pool,
                      X_lab=X[lab], y_lab=y[lab], X_eval=X[evalu],
                      top_idx=top_idx, cache=dict(s4_cache))
            preds = run_all(ctx)
            s4_cache.setdefault("S4_model", ctx.cache.get("S4_model"))

            diag = all_diagnostics(ctx, group_idx, tier1_cache=t1)
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

            hb, hpicks = honest_oracle(y[evalu], preds,
                                       np.random.default_rng(rng.integers(2**32)))
            row["brier_oracle_honest"] = hb
            row["oracle_honest_picks"] = hpicks
            # how much of the apparent gain survives having to choose blind
            row["oracle_gain"] = row["brier_S0"] - row["brier_oracle"]
            row["oracle_gain_honest"] = row["brier_S0"] - hb
            rows.append(row)
    return rows


def to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)
