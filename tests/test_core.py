"""Sanity tests that must pass before any result is believed."""
import numpy as np
from scipy.special import expit, logit

from src.csh import features, metrics
from src.csh.strategies import (apply_prior_shift, domain_weights, em_prior,
                                fit_intercept, fit_platt)

rng = np.random.default_rng(0)


def test_brier_and_ece_on_known_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert metrics.brier(y, p) < 1e-8
    assert metrics.ece(y, p) < 1e-5
    p2 = np.full(4, 0.5)
    assert abs(metrics.brier(y, p2) - 0.25) < 1e-8


def test_perfect_calibration_gives_slope_one():
    n = 20000
    p = rng.uniform(0.02, 0.9, n)
    y = rng.binomial(1, p)
    a, b = metrics.calibration_slope_intercept(y, p)
    assert abs(b - 1.0) < 0.12, b
    assert abs(a) < 0.12, a
    assert metrics.ece(y, p) < 0.02


def test_intercept_recovers_known_offset():
    n = 8000
    p = rng.uniform(0.02, 0.9, n)
    true_a = 0.8
    y = rng.binomial(1, expit(true_a + logit(p)))
    a = fit_intercept(y, p)
    assert abs(a - true_a) < 0.12, a


def test_platt_recovers_slope_and_intercept():
    n = 12000
    p = rng.uniform(0.02, 0.9, n)
    y = rng.binomial(1, expit(-0.3 + 1.4 * logit(p)))
    a, b = fit_platt(y, p)
    assert abs(a + 0.3) < 0.15 and abs(b - 1.4) < 0.15, (a, b)


def test_em_prior_recovers_resampled_prevalence():
    n = 40000
    x = rng.normal(size=n)
    p_true = expit(-2.0 + 1.5 * x)          # source prevalence ~0.16
    y = rng.binomial(1, p_true)
    src_prior = y.mean()
    # resample the target to roughly double the prevalence
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    take_neg = rng.choice(idx_neg, size=len(idx_neg) // 3, replace=False)
    idx = np.r_[idx_pos, take_neg]
    target_prior = y[idx].mean()
    pi_hat, _ = em_prior(p_true[idx], src_prior)
    assert abs(pi_hat - target_prior) < 0.03, (pi_hat, target_prior)


def test_prior_shift_moves_probabilities_the_right_way():
    p = np.array([0.1, 0.3, 0.6])
    up = apply_prior_shift(p, 0.1, 0.3)
    down = apply_prior_shift(p, 0.1, 0.03)
    assert np.all(up > p) and np.all(down < p)


def test_importance_weights_are_neutral_without_shift():
    X = rng.normal(size=(1500, 8))
    Xp = rng.normal(size=(1500, 8))
    w, ess = domain_weights(X, Xp)
    assert abs(w.mean() - 1.0) < 1e-6
    assert ess > 0.85, ess


def test_importance_weights_react_to_shift():
    X = rng.normal(size=(1500, 8))
    Xp = rng.normal(loc=1.5, size=(1500, 8))
    w, ess = domain_weights(X, Xp)
    assert ess < 0.7, ess


def test_leakage_guard_catches_forbidden_column():
    import pandas as pd
    df = pd.DataFrame({"G1_age": [1.0, 2.0], "G2_apachescore": [3.0, 4.0],
                       "y": [0, 1], "domain": ["a", "b"], "stay_id": [1, 2]})
    try:
        features.check_leakage(df)
    except AssertionError:
        return
    raise AssertionError("leakage guard failed to fire")


def test_real_cohorts_pass_leakage_guard():
    for ds in ("p2012", "eicu_demo"):
        df = features.load(ds)
        features.check_leakage(df)
        cols = features.feature_columns(df)
        assert "y" not in cols and "domain" not in cols
        assert len(features.group_map(cols)) >= 3
