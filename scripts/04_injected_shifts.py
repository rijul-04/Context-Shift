"""Injected-shift pseudo-deployments.

Two purposes: more meta-training episodes than the handful of real domains
can provide, and a ground truth for H7 (do the grouped diagnostics name the
right kind of change?).

The pseudo-target is carved out of the source pool, so the base model is
retrained without it, exactly as for a real domain.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh import features                                     # noqa: E402
from src.csh.baselines import (ctp_pick, tcv_pick,               # noqa: E402
                                tcv_se_pick)
from src.csh.config import PROC, RESULTS, SEED                   # noqa: E402
from src.csh.diagnostics import all_diagnostics, tier1           # noqa: E402
from src.csh.metrics import brier, ece, oe_ratio                 # noqa: E402
from src.csh.models import fit, make_model, predict_proba, top_features  # noqa: E402
from src.csh.episodes import (MIN_EVAL, draw_labelled,            # noqa: E402
                              honest_oracle)
from src.csh.shifts import SHIFT_TYPES, make_injected, severities  # noqa: E402
from src.csh.strategies import (Ctx, LADDER, LADDER_CORE,        # noqa: E402
                                run_all)

DATASETS = ["p2012", "eicu_demo"]
KIND = "hgb"
BUDGETS = [0, 25, 100, 400]
REPEATS = 2
CACHE = PROC / "injected_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def one_injected(ds, kind, shift_type, severity, fold, X, y, dom, cols, gi):
    tag = f"{ds}__{kind}__{shift_type}__{severity}__{fold}"
    path = CACHE / f"{tag}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    rng = np.random.default_rng(abs(hash(tag)) % 2**32)
    n = len(y)
    perm = rng.permutation(n)
    cut = int(0.3 * n)
    tgt_pool, src_pool = perm[:cut], perm[cut:]
    if len(np.unique(y[src_pool])) < 2:
        return pd.DataFrame()

    Xs, ys = X[src_pool], y[src_pool]
    model = fit(make_model(kind), Xs, ys)
    top_idx = top_features(model, Xs, cols, k=10)
    g2 = gi.get("G2", [])
    g3_counts = [i for i, c in enumerate(cols)
                 if c.startswith("G3_") and c.endswith(("count", "per_hour"))]

    Xt, yt, meta = make_injected(X, y, tgt_pool, rng, shift_type, severity,
                                 top_idx, g2, g3_counts)
    if len(yt) < 60 or len(np.unique(yt)) < 2:
        return pd.DataFrame()

    idx = np.arange(len(yt))
    base = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys, X_pool=Xt,
               X_lab=Xt[:0], y_lab=yt[:0], X_eval=Xt, top_idx=top_idx)
    t1 = tier1(base, gi)
    s4 = {}
    rows = []
    for nb in [b for b in BUDGETS if b <= len(yt) - MIN_EVAL]:
        for r in range(1 if nb == 0 else REPEATS):
            rr = np.random.default_rng(abs(hash((tag, nb, r))) % 2**32)
            lab = draw_labelled(idx, nb, rr)
            evalu = np.setdiff1d(idx, lab, assume_unique=True)
            if len(evalu) < MIN_EVAL or len(np.unique(yt[evalu])) < 2:
                continue
            ctx = Ctx(model=model, kind=kind, X_src=Xs, y_src=ys,
                      X_pool=Xt, X_lab=Xt[lab], y_lab=yt[lab],
                      X_eval=Xt[evalu], top_idx=top_idx, cache=dict(s4))
            preds = run_all(ctx)
            s4.setdefault("S4_model", ctx.cache.get("S4_model"))
            row = {
                "dataset": ds, "target": f"inj_{shift_type}_{severity}_{fold}",
                "model": kind, "n": nb, "rep": r, "tag": "injected",
                "shift_type": shift_type, "severity": severity,
                "n_eval": len(evalu), "events_eval": int(yt[evalu].sum()),
                "prev_eval": float(yt[evalu].mean()),
                "pick_tcv": tcv_pick(ctx)[0],
                "pick_tcvse": tcv_se_pick(ctx)[0],
                "pick_ctp": ctp_pick(ctx)[0],
                **all_diagnostics(ctx, gi, tier1_cache=t1),
            }
            for k in LADDER:
                row[f"brier_{k}"] = brier(yt[evalu], preds[k])
            row["ece_S0"] = ece(yt[evalu], preds["S0"])
            row["oe_S0"] = oe_ratio(yt[evalu], preds["S0"])
            best = min(LADDER, key=lambda k: row[f"brier_{k}"])
            row["oracle"], row["brier_oracle"] = best, row[f"brier_{best}"]
            bc = min(LADDER_CORE, key=lambda k: row[f"brier_{k}"])
            row["oracle_core"], row["brier_oracle_core"] = bc, row[f"brier_{bc}"]
            hb, hg, hp = honest_oracle(
                yt[evalu], preds, np.random.default_rng(rr.integers(2**32)))
            row["brier_oracle_honest"], row["oracle_honest_picks"] = hb, hp
            row["oracle_gain"] = row["brier_S0"] - row["brier_oracle"]
            row["oracle_gain_honest"] = hg
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_pickle(path)
    return out


def main():
    t0 = time.time()
    specs = []
    for ds in DATASETS:
        df = features.load(ds)
        X, y, dom, cols = features.matrix(df)
        gi = features.group_map(cols)
        for st in SHIFT_TYPES:
            for sev in severities(st):
                for fold in range(1 if st == "none" else 1):
                    specs.append((ds, KIND, st, sev, fold, X, y, dom, cols, gi))
    print(f"{len(specs)} injected pseudo-targets", flush=True)

    out = Parallel(n_jobs=2, verbose=5)(delayed(one_injected)(*s) for s in specs)
    df = pd.concat([d for d in out if len(d)], ignore_index=True)
    df.to_pickle(PROC / "episodes_injected.pkl")
    df.to_csv(RESULTS / "episodes_injected.csv", index=False)
    print(f"\n{len(df)} injected episodes in {time.time() - t0:.0f}s")

    # H7: does the grouped C2ST point at the right feature group?
    g_cols = ["t1_c2st_G1", "t1_c2st_G2", "t1_c2st_G3", "t1_c2st_G4"]
    d0 = df[df["n"] == 0].copy()
    d0["top_group"] = d0[g_cols].idxmax(axis=1).str.replace("t1_c2st_", "")
    tab = pd.crosstab(d0["shift_type"], d0["top_group"])
    tab.to_csv(RESULTS / "h7_diagnosis.csv")
    print("\nshift type vs most-separable feature group (n=0 episodes):")
    print(tab.to_string())
    print("\nmean overall C2ST AUC by shift type:")
    print(d0.groupby("shift_type")["t1_c2st_auc"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
