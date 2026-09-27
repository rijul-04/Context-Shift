"""H1: does moving a model to an unseen domain hurt?

Three references, all scored on the SAME held-out evaluation half of the
target domain, so prevalence differences between domains cannot drive the
comparison:

  external  = source model (trained without the target domain)
  local     = model trained on the target's own adaptation half
  internal  = grouped out-of-fold predictions inside the source domains
              (reported for context; not a like-for-like comparison)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh import features, metrics                      # noqa: E402
from src.csh.config import RESULTS, SEED                    # noqa: E402
from src.csh.models import fit, make_model, predict_proba, tune  # noqa: E402
from src.csh.splits import domain_split, lodo               # noqa: E402
from sklearn.model_selection import GroupKFold              # noqa: E402

DATASETS = ["p2012", "eicu_demo"]
KINDS = ["hgb", "lr"]


def internal_oof(kind, X, y, groups, params):
    """Out-of-fold predictions across source domains (grouped by domain)."""
    oof = np.full(len(y), np.nan)
    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=min(len(uniq), 4))
    for tr, te in gkf.split(X, y, groups):
        m = fit(make_model(kind, **params), X[tr], y[tr])
        oof[te] = predict_proba(m, X[te])
    return oof


def main():
    rows = []
    for ds in DATASETS:
        df = features.load(ds)
        features.check_leakage(df)
        X, y, dom, cols = features.matrix(df)
        print(f"\n=== {ds}: {len(y)} stays, {y.sum()} deaths, "
              f"{len(cols)} features, domains {sorted(set(dom))}")

        for kind in KINDS:
            for target, src_mask, tgt_mask in lodo(dom):
                Xs, ys, gs = X[src_mask], y[src_mask], dom[src_mask]
                params = tune(kind, Xs, ys, gs)
                model = fit(make_model(kind, **params), Xs, ys)

                oof = internal_oof(kind, Xs, ys, gs, params)
                ok = ~np.isnan(oof)
                internal = metrics.all_metrics(ys[ok], oof[ok])

                tgt_idx = np.where(tgt_mask)[0]
                ad, ev = domain_split(y, tgt_idx, seed=SEED)
                p_ext = predict_proba(model, X[ev])
                external = metrics.all_metrics(y[ev], p_ext)

                # local reference: same eval patients, model trained on the
                # target's own adaptation half
                local_model = fit(make_model(kind, **params), X[ad], y[ad])
                p_loc = predict_proba(local_model, X[ev])
                local = metrics.all_metrics(y[ev], p_loc)

                rows.append({
                    "dataset": ds, "model": kind, "target": target,
                    "params": str(params), "n_adapt": int(len(ad)),
                    **{f"int_{k}": v for k, v in internal.items()},
                    **{f"ext_{k}": v for k, v in external.items()},
                    **{f"loc_{k}": v for k, v in local.items()},
                    "d_brier": external["brier"] - local["brier"],
                    "d_auroc": external["auroc"] - local["auroc"],
                    "d_ece": external["ece"] - local["ece"],
                })
                print(f"  {kind} -> {target:16s} "
                      f"AUROC ext {external['auroc']:.3f} / loc {local['auroc']:.3f}  "
                      f"Brier ext {external['brier']:.4f} / loc {local['brier']:.4f}  "
                      f"ECE {external['ece']:.3f}  slope {external['cal_slope']:.2f}  "
                      f"O/E {external['oe']:.2f}  "
                      f"(n={external['n']}, events={external['events']})")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "h1_degradation.csv", index=False)
    print("\n--- H1 summary (mean over target domains) ---")
    print(out.groupby(["dataset", "model"])[
        ["ext_auroc", "loc_auroc", "ext_brier", "loc_brier",
         "ext_ece", "loc_ece", "ext_cal_slope", "ext_oe"]]
        .mean().round(4).to_string())
    print(f"\nsaved {RESULTS / 'h1_degradation.csv'}")


if __name__ == "__main__":
    main()
