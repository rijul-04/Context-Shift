"""Statistics: domains are the sample, not patients."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import SEED


def cluster_bootstrap_ci(values_by_domain: dict[str, np.ndarray],
                         n_boot: int = 2000, seed: int = SEED,
                         alpha: float = 0.05):
    """Resample domains (not rows) and return (mean, lo, hi)."""
    doms = list(values_by_domain)
    means = np.array([np.mean(values_by_domain[d]) for d in doms])
    if len(doms) < 2:
        return float(means.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.array([
        means[rng.integers(0, len(doms), len(doms))].mean()
        for _ in range(n_boot)
    ])
    return (float(means.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def paired_domain_test(a_by_domain: dict[str, float], b_by_domain: dict[str, float]):
    """Wilcoxon signed-rank over domains, plus the paired difference."""
    doms = sorted(set(a_by_domain) & set(b_by_domain))
    a = np.array([a_by_domain[d] for d in doms])
    b = np.array([b_by_domain[d] for d in doms])
    diff = a - b
    if len(doms) < 3 or np.allclose(diff, 0):
        return {"n_domains": len(doms), "mean_diff": float(diff.mean()),
                "p": float("nan")}
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return {"n_domains": len(doms), "mean_diff": float(diff.mean()),
            "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def per_domain_mean(df: pd.DataFrame, value_col: str,
                    domain_col: str = "domain_key") -> dict[str, float]:
    return df.groupby(domain_col)[value_col].mean().to_dict()
