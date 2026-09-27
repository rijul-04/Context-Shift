"""Reliability metrics: discrimination, calibration, regret, harm."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def brier(y, p) -> float:
    p = clip(p)
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2))


def auroc(y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def ece(y, p, n_bins: int = 15) -> float:
    """Equal-mass (quantile) expected calibration error."""
    y = np.asarray(y, dtype=float)
    p = clip(p)
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p)
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(n), min(n_bins, max(1, n)))
    tot = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        tot += len(b) / n * abs(y[b].mean() - p[b].mean())
    return float(tot)


def calibration_slope_intercept(y, p):
    """Logistic recalibration fit: logit(y) ~ a + b * logit(p)."""
    from .strategies import fit_platt  # local import to avoid a cycle

    y = np.asarray(y, dtype=float)
    p = clip(p)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    a, b = fit_platt(y, p)
    return float(a), float(b)


def oe_ratio(y, p) -> float:
    e = float(np.sum(clip(p)))
    return float(np.sum(y)) / e if e > 0 else float("nan")


def scaled_brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    base = y.mean()
    b0 = float(np.mean((base - y) ** 2))
    return 1.0 - brier(y, p) / b0 if b0 > 0 else float("nan")


def all_metrics(y, p) -> dict:
    a, b = calibration_slope_intercept(y, p)
    return {
        "brier": brier(y, p),
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "ece": ece(y, p),
        "cal_intercept": a,
        "cal_slope": b,
        "oe": oe_ratio(y, p),
        "scaled_brier": scaled_brier(y, p),
        "n": int(len(y)),
        "events": int(np.sum(y)),
    }


def regret(brier_selected: float, brier_oracle: float) -> float:
    return float(brier_selected - brier_oracle)


def relative_regret(brier_selected: float, brier_oracle: float,
                    brier_none: float) -> float:
    denom = brier_none - brier_oracle + 1e-4
    return float((brier_selected - brier_oracle) / denom)


def is_harmful(brier_selected: float, brier_none: float, delta: float) -> bool:
    return bool(brier_selected - brier_none > delta)
