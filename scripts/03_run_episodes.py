"""Run the real deployment episodes for every dataset, model and target.

Each (dataset, model, target) is cached to disk, so an interrupted run
resumes instead of starting over.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh import features                                # noqa: E402
from src.csh.config import N_REPEATS, PROC, RESULTS         # noqa: E402
from src.csh.episodes import run_target                     # noqa: E402
from src.csh.strategies import LADDER                       # noqa: E402

DATASETS = ["p2012", "eicu_demo"]
KINDS = ["hgb", "lr"]
CACHE = PROC / "episodes_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def run_one(ds, kind, target, X, y, dom, cols, gi):
    path = CACHE / f"{ds}__{kind}__{target}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    t0 = time.time()
    rows = run_target(X=X, y=y, dom=dom, target=target, kind=kind,
                      group_idx=gi, dataset=ds, cols=cols,
                      n_repeats=N_REPEATS)
    df = pd.DataFrame(rows)
    df.to_pickle(path)
    print(f"  {ds} {kind} {target}: {len(df)} episodes "
          f"in {time.time() - t0:.0f}s", flush=True)
    return df


def main():
    t0 = time.time()
    specs = []
    for ds in DATASETS:
        df = features.load(ds)
        features.check_leakage(df)
        X, y, dom, cols = features.matrix(df)
        gi = features.group_map(cols)
        for kind in KINDS:
            for target in sorted(set(dom)):
                specs.append((ds, kind, target, X, y, dom, cols, gi))
    print(f"{len(specs)} (dataset, model, target) combinations", flush=True)

    out = Parallel(n_jobs=-1, verbose=5)(delayed(run_one)(*s) for s in specs)
    df = pd.concat([d for d in out if len(d)], ignore_index=True)
    df.to_pickle(PROC / "episodes_real.pkl")
    df.to_csv(RESULTS / "episodes_real.csv", index=False)

    print(f"\n{len(df)} episodes in {time.time() - t0:.0f}s")
    print("\noracle strategy by budget:")
    print(pd.crosstab(df["n"], df["oracle"]).to_string())
    print("\nmean Brier per strategy:")
    print(df[[f"brier_{k}" for k in LADDER]].mean().round(5).to_string())


if __name__ == "__main__":
    main()
