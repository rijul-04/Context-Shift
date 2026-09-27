"""H6 transfer, H5 tier ablation, and what the selector actually reads.

Transfer  train the selector on one dataset's episodes, test on the other's.
          Nothing about the target dataset is seen, so this is the strongest
          test that the diagnostics carry something general.
Ablation  Tier 1 only vs Tier 2 only vs both, already produced by script 05;
          here we add permutation importance over the meta-features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh.config import LOW_BUDGET_MAX, PROC, RESULTS, SEED   # noqa: E402
from src.csh.selectors import (FEATURE_SETS, MetaSelector,       # noqa: E402
                               best_fixed_on, fixed_policy,
                               selection_results)
from src.csh.stats import cluster_bootstrap_ci                   # noqa: E402


def load_all() -> pd.DataFrame:
    parts = [pd.read_pickle(PROC / "episodes_real.pkl")]
    inj = PROC / "episodes_injected.pkl"
    if inj.exists():
        parts.append(pd.read_pickle(inj))
    df = pd.concat(parts, ignore_index=True)
    df["domain_key"] = df["dataset"] + ":" + df["target"]
    return df


def transfer(df: pd.DataFrame) -> pd.DataFrame:
    """Train on every dataset but one; test on the real domains of that one."""
    rows = []
    for held in sorted(df["dataset"].unique()):
        train = df[df["dataset"] != held]
        test = df[(df["dataset"] == held) & (df["tag"] == "real")]
        if len(train) < 50 or test.empty:
            continue
        policies = {
            "SEL-META-both (transferred)": MetaSelector("both").fit(train).select(test),
            "SEL-META-t1 (transferred)": MetaSelector("t1").fit(train).select(test),
            "TCV": list(test["pick_tcv"]),
            "CTP": list(test["pick_ctp"]),
            "best-fixed": fixed_policy(test, best_fixed_on(train)),
            "oracle": list(test["oracle"]),
        }
        for name, picks in policies.items():
            res = selection_results(test, picks)
            res["domain_key"] = test["domain_key"].values
            res["n"] = test["n"].values
            low = res["n"] <= LOW_BUDGET_MAX
            by_dom = {d: g["rel_regret"].to_numpy()
                      for d, g in res[low].groupby("domain_key")}
            mean, lo, hi = cluster_bootstrap_ci(by_dom)
            rows.append({
                "held_out_dataset": held, "policy": name,
                "rel_regret_low_budget": mean, "ci_lo": lo, "ci_hi": hi,
                "regret_all": res["regret"].mean(),
                "harm_rate": float((res["brier"] > res["brier_S0"] + 0.002).mean()),
                "n_episodes": len(res),
            })
    return pd.DataFrame(rows)


def permutation_importance(df: pd.DataFrame, n_repeats: int = 3,
                           seed: int = SEED) -> pd.DataFrame:
    """How much does shuffling one diagnostic cost the selector?

    Measured in selection regret, leave-one-domain-out, so it answers
    'what does the selector rely on?' rather than 'what fits the training
    data?'.
    """
    rng = np.random.default_rng(seed)
    real = df[df["tag"] == "real"]
    cols = FEATURE_SETS["both"]

    base_scores, perm_scores = [], {c: [] for c in cols}
    for held in sorted(real["domain_key"].unique()):
        train = df[df["domain_key"] != held]
        test = real[real["domain_key"] == held]
        if train.empty or test.empty:
            continue
        sel = MetaSelector("both").fit(train)
        base = selection_results(test, sel.select(test))["regret"].mean()
        base_scores.append(base)
        for c in cols:
            vals = []
            for _ in range(n_repeats):
                shuffled = test.copy()
                shuffled[c] = rng.permutation(shuffled[c].to_numpy())
                vals.append(
                    selection_results(test, sel.select(shuffled))["regret"].mean())
            perm_scores[c].append(float(np.mean(vals)) - base)

    out = pd.DataFrame({
        "feature": cols,
        "importance": [float(np.mean(perm_scores[c])) if perm_scores[c] else 0.0
                       for c in cols],
        "tier": ["Tier 0" if c.startswith("t0_") else
                 ("Tier 1" if c.startswith("t1_") else "Tier 2") for c in cols],
    }).sort_values("importance", ascending=False)
    return out


def main():
    df = load_all()
    print(f"{len(df)} episodes "
          f"({(df['tag'] == 'real').sum()} real, "
          f"{(df['tag'] == 'injected').sum()} injected)")

    t = transfer(df)
    if not t.empty:
        t.to_csv(RESULTS / "transfer.csv", index=False)
        print("\n=== H6: trained on the other dataset(s), tested here ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    imp = permutation_importance(df)
    imp.to_csv(RESULTS / "meta_importance.csv", index=False)
    print("\n=== what the selector reads (top 12) ===")
    print(imp.head(12).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nimportance summed by tier:")
    print(imp.groupby("tier")["importance"].sum().round(5).to_string())


if __name__ == "__main__":
    main()
