"""Reliability curves for one site, before and after the chosen update.

Cardiac surgery is the clearest case: the transported model predicts roughly
twice the deaths that occur (O/E 0.53), and an intercept correction fixes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh import features, metrics                          # noqa: E402
from src.csh.config import PROC, RESULTS, SEED                 # noqa: E402
from src.csh.models import fit, make_model, predict_proba, top_features  # noqa: E402
from src.csh.splits import domain_split                        # noqa: E402
from src.csh.strategies import Ctx, run_all                    # noqa: E402

TARGET = ("p2012", "cardiac_surgery", "hgb", "S2", 100)


def reliability(y, p, n_bins: int = 8) -> pd.DataFrame:
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    return pd.DataFrame([
        {"pred": float(p[b].mean()), "obs": float(y[b].mean()), "n": len(b)}
        for b in bins if len(b)
    ])


def main():
    ds, target, kind, strategy, budget = TARGET
    df = features.load(ds)
    X, y, dom, cols = features.matrix(df)
    src = dom != target
    model = fit(make_model(kind), X[src], y[src])
    top = top_features(model, X[src], cols, 10)

    tgt_idx = np.where(dom == target)[0]
    adapt, evalu = domain_split(y, tgt_idx, seed=SEED)
    rng = np.random.default_rng(SEED)
    lab = rng.choice(adapt, size=budget, replace=False)

    ctx = Ctx(model=model, kind=kind, X_src=X[src], y_src=y[src],
              X_pool=X[adapt], X_lab=X[lab], y_lab=y[lab], X_eval=X[evalu],
              top_idx=top)
    preds = run_all(ctx)

    before = reliability(y[evalu], preds["S0"]).assign(which="before")
    after = reliability(y[evalu], preds[strategy]).assign(which="after")
    out = pd.concat([before, after], ignore_index=True)
    out.attrs["meta"] = TARGET
    out.to_pickle(PROC / "reliability_example.pkl")

    rows = []
    for name, key in (("before (no update)", "S0"),
                      (f"after ({strategy}, n={budget})", strategy)):
        m = metrics.all_metrics(y[evalu], preds[key])
        m["setting"] = name
        rows.append(m)
    tab = pd.DataFrame(rows).set_index("setting")
    tab.to_csv(RESULTS / "reliability_example.csv")
    print(f"{ds} / {target} / {kind}: {budget} labelled cases")
    print(tab[["brier", "auroc", "ece", "cal_slope", "cal_intercept", "oe"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
