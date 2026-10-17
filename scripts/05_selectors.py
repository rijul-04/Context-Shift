"""Train and evaluate the selectors, leave-one-target-domain-out.

Policies compared
  oracle            best strategy in hindsight (upper bound, not attainable)
  SEL-META-both     our selector, Tier 1 + Tier 2 diagnostics
  SEL-META-t1       Tier 1 only  (label-free: the central claim)
  SEL-META-t2       Tier 2 only  (what outcome-based selection can do)
  SEL-RULE-both     the printable depth-3 policy
  TCV / CTP         what practice does today (both Tier 2)
  best-fixed        the single best strategy, chosen on training domains
  always-S0 ...     every fixed strategy, for reference
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh.config import (DELTA_HARM, LOW_BUDGET_MAX, PROC,  # noqa: E402
                            RESULTS)
from src.csh.selectors import (MetaSelector, RuleSelector,     # noqa: E402
                               best_fixed_on, fixed_policy,
                               selection_results)
from src.csh.stats import (cluster_bootstrap_ci, holm,          # noqa: E402
                           paired_domain_test, wilson_ci)
from src.csh.strategies import LADDER                           # noqa: E402

PRIMARY = "SEL-META-both"
COMPETITORS = ["best-fixed", "TCV", "CTP"]


def evaluate(df: pd.DataFrame, injected: pd.DataFrame | None = None
             ) -> pd.DataFrame:
    """Leave-one-target-domain-out evaluation of every policy.

    Selectors train on the other real domains plus every injected
    pseudo-deployment; they are tested only on real held-out domains.
    """
    df = df.copy()
    df["domain_key"] = df["dataset"] + ":" + df["target"]
    if injected is not None and len(injected):
        injected = injected.copy()
        injected["domain_key"] = injected["dataset"] + ":" + injected["target"]
    out = []

    cache_dir = PROC / "selection_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for held in sorted(df["domain_key"].unique()):
        cached = cache_dir / f"{held.replace(':', '__')}.pkl"
        if cached.exists():
            out.append(pd.read_pickle(cached))
            continue
        train = df[df["domain_key"] != held]
        if injected is not None and len(injected):
            train = pd.concat([train, injected], ignore_index=True)
        test = df[df["domain_key"] == held]
        if train.empty or test.empty:
            continue

        policies: dict[str, list[str]] = {}
        for tiers in ("both", "t1", "t2"):
            policies[f"SEL-META-{tiers}"] = (
                MetaSelector(tiers=tiers).fit(train).select(test))
        policies["SEL-RULE-both"] = RuleSelector(tiers="both").fit(train).select(test)

        policies["TCV"] = list(test["pick_tcv"])
        if "pick_tcvse" in test:
            policies["TCV-SE (ours)"] = list(test["pick_tcvse"])
        policies["CTP"] = list(test["pick_ctp"])
        bf = best_fixed_on(train)
        policies["best-fixed"] = fixed_policy(test, bf)
        for k in LADDER:
            policies[f"always-{k}"] = fixed_policy(test, k)
        policies["oracle"] = list(test["oracle"])

        fold_rows = []
        for name, picks in policies.items():
            res = selection_results(test, picks)
            res["policy"] = name
            res["domain_key"] = held
            res["dataset"] = test["dataset"].values
            res["n"] = test["n"].values
            res["model"] = test["model"].values
            res["best_fixed_choice"] = bf
            fold_rows.append(res)
        fold = pd.concat(fold_rows, ignore_index=True)
        fold.to_pickle(cached)
        print(f"  fold {held}: done", flush=True)
        out.append(fold)

    return pd.concat(out, ignore_index=True)


def summarise(res: pd.DataFrame, mask_name: str, mask) -> pd.DataFrame:
    rows = []
    sub_all = res[mask]
    for policy, sub in sub_all.groupby("policy"):
        by_dom = {d: g["rel_regret"].to_numpy()
                  for d, g in sub.groupby("domain_key")}
        mean, lo, hi = cluster_bootstrap_ci(by_dom)
        harm = (sub["brier"] - sub["brier_S0"] > DELTA_HARM)
        h, hlo, hhi = wilson_ci(int(harm.sum()), len(harm))
        rows.append({
            "regime": mask_name, "policy": policy,
            "rel_regret": mean, "ci_lo": lo, "ci_hi": hi,
            "brier": sub["brier"].mean(),
            "regret": sub["regret"].mean(),
            "harm_rate": h, "harm_lo": hlo, "harm_hi": hhi,
            "net_gain": sub["net_gain"].mean(),
            "rel_regret_honest": (sub["rel_regret_honest"].mean()
                                  if "rel_regret_honest" in sub else float("nan")),
            "helped_rate": float((sub["net_gain"] > 0.002).mean()),
            "top1_acc": sub["correct"].mean(),
            "mean_cost": sub["cost"].mean(),
            "n_episodes": len(sub),
        })
    return pd.DataFrame(rows).sort_values("rel_regret")


def primary_test(res: pd.DataFrame, mask) -> pd.DataFrame:
    sub = res[mask]
    prim = sub[sub["policy"] == PRIMARY].groupby("domain_key")["rel_regret"].mean()
    rows, pvals = [], {}
    for comp in COMPETITORS:
        other = sub[sub["policy"] == comp].groupby("domain_key")["rel_regret"].mean()
        t = paired_domain_test(prim.to_dict(), other.to_dict())
        red = (1 - prim.mean() / other.mean()) if other.mean() > 0 else np.nan
        pvals[comp] = t["p"]
        rows.append({"comparison": f"{PRIMARY} vs {comp}",
                     "mean_ours": prim.mean(), "mean_theirs": other.mean(),
                     "mean_diff": t["mean_diff"], "reduction": red,
                     "n_domains": t["n_domains"], "p_raw": t["p"]})
    adj = holm(pvals)
    for r, comp in zip(rows, COMPETITORS):
        r["p_holm"] = adj[comp]
    return pd.DataFrame(rows)


def main():
    df = pd.read_pickle(PROC / "episodes_real.pkl")
    gain = df["brier_S0"] - df["brier_oracle"]
    gain_h = df["brier_S0"] - df["brier_oracle_honest"]
    print(f"available gain: apparent {gain.mean()*1000:.2f}e-3 Brier, "
          f"honest {gain_h.mean()*1000:.2f}e-3 "
          f"({100*(1-gain_h.mean()/max(gain.mean(),1e-9)):.0f}% of the apparent "
          f"gain is sampling noise)")
    per_site = (df.assign(g=gain, gh=gain_h)
                .groupby(["dataset", "target"])[["g", "gh"]].mean() * 1000)
    print("\nreal gain available per site (Brier x1e-3):")
    print(per_site.round(2).to_string())
    print(f"\n{len(df)} episodes, "
          f"{df['dataset'].nunique()} datasets, "
          f"{df.groupby(['dataset', 'target']).ngroups} target domains")

    inj_path = PROC / "episodes_injected.pkl"
    injected = pd.read_pickle(inj_path) if inj_path.exists() else None
    if injected is not None:
        print(f"+ {len(injected)} injected episodes for meta-training")
    res = evaluate(df, injected)
    res.to_pickle(PROC / "selection_results.pkl")

    low = res["n"] <= LOW_BUDGET_MAX
    tables = []
    for name, mask in (("low_budget", low), ("all_budgets", res["n"] >= 0)):
        t = summarise(res, name, mask)
        tables.append(t)
        print(f"\n=== {name} (n <= {LOW_BUDGET_MAX} if low) ===")
        print(t[["policy", "rel_regret", "rel_regret_honest", "ci_lo", "ci_hi",
                 "net_gain", "helped_rate", "harm_rate", "top1_acc", "mean_cost"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(RESULTS / "selector_summary.csv", index=False)

    print("\n=== primary test (H3), low-budget regime ===")
    pt = primary_test(res, low)
    print(pt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pt.to_csv(RESULTS / "primary_test.csv", index=False)

    # regret by budget, for the crossover figure
    curve = (res.groupby(["policy", "n"])["rel_regret"].mean()
             .unstack("n"))
    curve.to_csv(RESULTS / "regret_by_budget.csv")
    print("\n=== relative regret by budget ===")
    keep = [p for p in curve.index
            if p.startswith("SEL-") or p in ("TCV", "CTP", "best-fixed",
                                             "always-S0", "always-S3")]
    print(curve.loc[keep].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
