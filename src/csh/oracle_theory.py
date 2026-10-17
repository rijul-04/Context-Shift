"""Why a best-in-hindsight oracle overstates what is available, and by how much.

Setting. A site evaluates $K$ update strategies on $m$ patients. Strategy $k$
has true loss $b_k$ (its expected Brier score at this site) and we observe
$\\hat b_k = b_k + \\varepsilon_k$. The errors are not independent: the
strategies are scored on the *same* patients, so their errors are strongly
correlated, with equicorrelation $\\rho$ a reasonable working model.

The reported oracle is $\\min_k \\hat b_k$, and
$\\mathbb{E}[\\min_k \\hat b_k] < \\min_k b_k$ whenever the errors have spread:
picking the minimum of noisy estimates buys the noise as well as the signal.
Writing $\\varepsilon_k = \\sqrt{\\rho}\\,Z_0 + \\sqrt{1-\\rho}\\,Z_k$ with
independent standard normals, the shared component $Z_0$ shifts every
strategy together and cancels in a comparison against $S_0$, so the
inflation of the *gain over no update* is

    inflation(K, rho, sigma) = sigma * sqrt(1 - rho) * E[max of K iid N(0,1)]

where $\\sigma$ is the standard error of a single strategy's Brier estimate.
The expectation is computed exactly by quadrature rather than tabulated.

This module provides the prediction; `scripts/11_oracle_simulation.py` tests
it against simulation with known ground truth and against the real episodes.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats


def expected_max_iid(k: int) -> float:
    """E[max of k iid standard normals], by quadrature.

    k * \\int z phi(z) Phi(z)^{k-1} dz. Exact for k=2 (1/sqrt(pi)); the
    quadrature is checked against that in the tests.
    """
    if k < 2:
        return 0.0

    def integrand(z):
        return k * z * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)

    val, _ = integrate.quad(integrand, -12, 12, limit=400)
    return float(val)


def expected_max_equicorrelated(k: int, rho: float) -> float:
    """E[max] for k equicorrelated standard normals.

    The shared component contributes nothing to the spread of the maximum
    around the common mean, so only the idiosyncratic part survives.
    """
    rho = float(np.clip(rho, 0.0, 0.999))
    return np.sqrt(1.0 - rho) * expected_max_iid(k)


def predicted_inflation(sigma: float, k: int, rho: float = 0.0,
                        paired: bool = True) -> float:
    """How much a best-in-hindsight oracle overstates the available gain.

    `sigma` is the standard error of the quantity the maximum is taken over.
    With `paired=True` it is the standard error of a *difference* between two
    strategies, which has already removed the variation they share, so no
    further correlation correction applies. With `paired=False` it is a
    single strategy's marginal standard error and `rho` discounts the shared
    component.

    (The first version of this function applied the correlation factor to a
    paired standard error, correcting twice and under-predicting by more than
    an order of magnitude. The null simulation, where the true answer is zero
    by construction, is what exposed it.)
    """
    e_max = (expected_max_iid(k) if paired
             else expected_max_equicorrelated(k, rho))
    return float(sigma * e_max)


def brier_se(y, p) -> float:
    """Standard error of a Brier score estimate on these patients."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    per_case = (p - y) ** 2
    n = len(per_case)
    return float(np.std(per_case, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def diff_se(y, p_a, p_b) -> float:
    """Standard error of the *difference* in Brier between two strategies.

    Paired, so the shared patient-level variation cancels; this is the
    quantity the inflation formula needs.
    """
    y = np.asarray(y, dtype=float)
    d = (np.asarray(p_a) - y) ** 2 - (np.asarray(p_b) - y) ** 2
    n = len(d)
    return float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def effective_rho(losses: np.ndarray) -> float:
    """Correlation between strategies' per-patient losses.

    `losses` is (n_patients, K) of squared errors. Returns the mean
    off-diagonal correlation, the equicorrelation that best summarises them.
    """
    if losses.shape[1] < 2:
        return 0.0
    c = np.corrcoef(losses, rowvar=False)
    k = c.shape[0]
    off = c[~np.eye(k, dtype=bool)]
    off = off[np.isfinite(off)]
    return float(np.mean(off)) if len(off) else 0.0


def inflation_from_predictions(y, preds: dict) -> dict:
    """Predict the oracle's inflation for one episode from its own data.

    Uses the observed per-patient losses to estimate sigma and rho, then
    applies the formula. Returns the prediction alongside the pieces, so the
    paper can report them.
    """
    keys = list(preds)
    y = np.asarray(y, dtype=float)
    losses = np.column_stack([(np.asarray(preds[k]) - y) ** 2 for k in keys])
    n, k = losses.shape

    rho = effective_rho(losses)
    # sigma for the *gain over the reference strategy*: the paired SE, which
    # is what the maximum is taken over
    ref = keys[0]
    ses = [diff_se(y, preds[kk], preds[ref]) for kk in keys[1:]]
    sigma = float(np.nanmean(ses)) if ses else float("nan")

    return {
        "n": n, "k": k, "rho": rho, "sigma": sigma,
        "predicted_inflation": predicted_inflation(sigma, k, paired=True),
        "e_max": expected_max_iid(k),
    }
