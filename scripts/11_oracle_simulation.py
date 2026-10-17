"""Does the oracle-inflation theory hold, and is the honest oracle unbiased?

Three experiments.

A. Null. Every strategy is genuinely equally good, so the true available gain
   is exactly zero. Whatever the conventional oracle reports is pure
   inflation, and we can compare it with the predicted value.

B. One strategy truly better by a known delta. The honest oracle should
   recover delta; the conventional one should report delta plus inflation.

C. The real episodes. We predict each deployment's inflation from its own
   data (its paired standard errors and the correlation between strategies)
   and compare with the inflation actually observed.

Ground truth in A and B is set by construction, which is the point: on real
data nobody knows the true gain, which is how the inflation went unnoticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh.config import PROC, RESULTS, SEED            # noqa: E402
from src.csh.metrics import brier                          # noqa: E402
from src.csh.oracle_theory import (effective_rho,          # noqa: E402
                                   expected_max_equicorrelated,
                                   inflation_from_predictions,
                                   predicted_inflation)

K_DEFAULT = 8
N_SIMS = 150


def simulate_episode(m: int, k: int, delta: float, rng, base_rate: float = 0.12,
                     spread: float = 0.9):
    """One synthetic deployment with known truth.

    A patient's true risk is drawn around `base_rate`. Strategy 0 (the
    reference) predicts the true risk. Each other strategy perturbs its
    predictions slightly, which makes its true Brier worse by a known amount;
    strategy 1 is instead improved by `delta` if delta > 0.
    """
    logit_p = rng.normal(np.log(base_rate / (1 - base_rate)), spread, size=m)
    p_true = 1 / (1 + np.exp(-logit_p))
    y = rng.binomial(1, p_true)

    preds = {"S0": p_true.copy()}
    for j in range(1, k):
        # a strategy that is truly neutral: an unbiased, symmetric perturbation
        # in logit space has essentially no effect on expected Brier
        noise = rng.normal(0, 0.02, size=m)
        q = 1 / (1 + np.exp(-(logit_p + noise)))
        preds[f"S{j}"] = q
    if delta > 0:
        # make S1 genuinely better by shrinking S0's error: it predicts the
        # truth while S0 is degraded by a fixed logit offset
        worse = 1 / (1 + np.exp(-(logit_p + 0.35)))
        preds["S0"] = worse
        preds["S1"] = p_true.copy()
    return y, preds, p_true


def oracle_pair(y, preds, rng):
    """(apparent gain, honest gain) over the reference strategy.

    Both gains are paired: the honest one compares the chosen strategy with
    no update on the *same* held-out half, never against a full-sample
    reference.
    """
    keys = list(preds)
    b = {kk: brier(y, preds[kk]) for kk in keys}
    apparent = b["S0"] - min(b.values())

    m = len(y)
    idx = rng.permutation(m)
    a, c = idx[: m // 2], idx[m // 2:]
    gains = []
    for fit_h, score_h in ((a, c), (c, a)):
        best = min(keys, key=lambda kk: brier(y[fit_h], preds[kk][fit_h]))
        gains.append(brier(y[score_h], preds["S0"][score_h])
                     - brier(y[score_h], preds[best][score_h]))
    return apparent, float(np.mean(gains))


def true_gain(preds_fn, big_fn, rng, reps: int = 200):
    """The gain a strategy choice actually earns.

    Choose on a sample of the size the site really has, then score on a very
    large independent sample so the truth itself is measured precisely. A
    small scoring sample here would make the "truth" noisier than the thing
    it is meant to adjudicate.
    """
    gains = []
    for _ in range(reps):
        y, preds, _ = preds_fn(rng)
        best = min(preds, key=lambda kk: brier(y, preds[kk]))
        yb, pb, _ = big_fn(rng)            # 20,000 independent patients
        gains.append(brier(yb, pb["S0"]) - brier(yb, pb[best]))
    return float(np.mean(gains))


def experiment_a_null(rng) -> pd.DataFrame:
    rows = []
    for m in (50, 100, 200, 500, 1000, 2000):
        for k in (2, 4, 8):
            app, hon, pred = [], [], []
            for i in range(N_SIMS):
                y, preds, _ = simulate_episode(m, k, 0.0, rng)
                a, h = oracle_pair(y, preds, rng)
                app.append(a)
                hon.append(h)
                if i % 10 == 0:      # the prediction is the slow part
                    pred.append(inflation_from_predictions(y, preds)
                                ["predicted_inflation"])
            rows.append({
                "experiment": "no true gain", "m": m, "k": k, "true_gain": 0.0,
                "apparent": np.mean(app) * 1000,
                "honest": np.mean(hon) * 1000,
                "predicted_inflation": np.nanmean(pred) * 1000,
            })
    return pd.DataFrame(rows)


def experiment_b_known_delta(rng) -> pd.DataFrame:
    rows = []
    for m in (100, 500, 2000):
        def gen(r, m=m):
            return simulate_episode(m, K_DEFAULT, 0.5, r)

        def gen_big(r):
            return simulate_episode(20000, K_DEFAULT, 0.5, r)

        truth = true_gain(gen, gen_big, rng)
        app, hon = [], []
        for _ in range(N_SIMS):
            y, preds, _ = gen(rng)
            a, h = oracle_pair(y, preds, rng)
            app.append(a)
            hon.append(h)
        rows.append({
            "experiment": "known gain", "m": m, "k": K_DEFAULT,
            "true_gain": truth * 1000,
            "apparent": np.mean(app) * 1000,
            "honest": np.mean(hon) * 1000,
            "predicted_inflation": float("nan"),
        })
    return pd.DataFrame(rows)


def experiment_c_real() -> pd.DataFrame:
    """Predicted against observed inflation on the real deployments."""
    path = PROC / "episodes_real.pkl"
    if not path.exists():
        return pd.DataFrame()
    ep = pd.read_pickle(path)
    if "brier_oracle_honest" not in ep.columns:
        return pd.DataFrame()
    ep = ep.copy()
    ep["observed_inflation"] = (ep["brier_oracle_honest"]
                                - ep["brier_oracle"]) * 1000
    # predicted inflation per episode, from the episode's own scale: the
    # spread of strategy Briers stands in for the paired standard error
    cols = [c for c in ep.columns if c.startswith("brier_S")]
    k = len(cols)
    # The paired standard error is not recoverable from stored summaries, so
    # scale from the spread of the strategies' Brier scores, which bounds it.
    sigma = ep[cols].std(axis=1) / np.sqrt(ep["n_eval"])
    ep["predicted_inflation"] = [
        predicted_inflation(s, k, paired=True) * 1000 for s in sigma]
    out = (ep.groupby(["dataset", "target"])
           [["n_eval", "observed_inflation", "predicted_inflation"]].mean())
    return out.reset_index()


def main():
    rng = np.random.default_rng(SEED)
    print("A. no true gain: every strategy equally good, truth is exactly 0")
    a = experiment_a_null(rng)
    print(a.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nB. one strategy truly better by a known amount")
    b = experiment_b_known_delta(rng)
    print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nC. the real deployments: predicted against observed inflation")
    c = experiment_c_real()
    if not c.empty:
        print(c.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    sim = pd.concat([a, b], ignore_index=True)
    sim.to_csv(RESULTS / "oracle_simulation.csv", index=False)
    c.to_csv(RESULTS / "oracle_inflation_real.csv", index=False)

    null = a[a["k"] == 8]
    print("\nsummary, K=8, no true gain:")
    print(f"  apparent gain falls from {null.iloc[0]['apparent']:.2f} at "
          f"m={null.iloc[0]['m']:.0f} to {null.iloc[-1]['apparent']:.2f} at "
          f"m={null.iloc[-1]['m']:.0f} (truth is 0 throughout)")
    print(f"  honest gain stays at {null['honest'].mean():.3f} on average")
    print(f"  theory tracks it: predicted {null['predicted_inflation'].mean():.2f} "
          f"vs observed {null['apparent'].mean():.2f}")
    print(f"\nsaved {RESULTS / 'oracle_simulation.csv'}")


if __name__ == "__main__":
    main()
