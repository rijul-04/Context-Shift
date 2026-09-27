"""Deployment splits: leave-one-domain-out, and each target's 50/50 split."""
from __future__ import annotations

import numpy as np

from .config import EVAL_FRACTION, SEED


def domain_split(y: np.ndarray, idx: np.ndarray, seed: int = SEED):
    """Split one domain's rows into an adaptation half and an evaluation half,
    stratified on the outcome. The evaluation half is read only by the scorer."""
    rng = np.random.default_rng(seed)
    adapt, evalu = [], []
    for cls in (0, 1):
        rows = idx[y[idx] == cls]
        rows = rng.permutation(rows)
        k = int(round(len(rows) * (1 - EVAL_FRACTION)))
        adapt.append(rows[:k])
        evalu.append(rows[k:])
    return np.sort(np.concatenate(adapt)), np.sort(np.concatenate(evalu))


def lodo(domains: np.ndarray):
    """Yield (target_domain, source_mask, target_mask) for each domain."""
    for t in sorted(np.unique(domains)):
        tgt = domains == t
        yield t, ~tgt, tgt
