"""Base clinical predictors: logistic regression and gradient boosting."""
from __future__ import annotations

import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .metrics import brier

MODEL_KINDS = ("hgb", "lr")


class DropDegenerate(BaseEstimator, TransformerMixin):
    """Drop columns with fewer than two distinct observed values.

    Such columns carry no information for any model, and scikit-learn >= 1.9
    raises inside histogram binning when it meets one ("window shape cannot be
    larger than input array shape"). They arise legitimately here: a lab that
    nobody in a subset had measured, or any column at all once a strategy fits
    on ten labelled cases. A column qualifies when its smallest observed value
    is below its largest, which is exact and costs two passes.
    """

    def fit(self, X, y=None, **kwargs):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # all-NaN columns warn here
            lo = np.nanmin(X, axis=0) if len(X) else np.zeros(X.shape[1])
            hi = np.nanmax(X, axis=0) if len(X) else np.zeros(X.shape[1])
        self.keep_ = np.asarray(lo < hi)
        # A fit on a degenerate slice keeps nothing; give the estimator one
        # alternating column so it still fits (and predicts the base rate)
        # instead of crashing.
        self.synthetic_ = not self.keep_.any()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.synthetic_:
            return (np.arange(len(X)) % 2).reshape(-1, 1).astype(float)
        return X[:, self.keep_]


def guarded(estimator) -> Pipeline:
    """Any tree estimator, protected from degenerate columns."""
    return Pipeline([("drop", DropDegenerate()), ("clf", estimator)])


def make_model(kind: str = "hgb", **over):
    if kind == "hgb":
        params = dict(
            learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, max_iter=100, max_bins=32,
            early_stopping=False, random_state=SEED,
        )
        params.update(over)
        return guarded(HistGradientBoostingClassifier(**params))
    if kind == "lr":
        params = dict(C=1.0, max_iter=2000, random_state=SEED)
        params.update(over)
        return Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
    raise ValueError(kind)


def fit(model, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        model.fit(X, y, clf__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def tune(kind: str, X, y, groups, grid=None) -> dict:
    """Grouped CV over source domains only. Small grids: this is not the
    contribution, and every extra fit costs day-1 wall clock."""
    grid = grid or (
        [{"learning_rate": lr, "max_leaf_nodes": n}
         for lr in (0.05, 0.1) for n in (15, 31)]
        if kind == "hgb" else
        [{"C": c} for c in (0.01, 0.1, 1.0)]
    )
    uniq = np.unique(groups)
    n_splits = min(len(uniq), 4)
    if n_splits < 2:
        return grid[0]
    gkf = GroupKFold(n_splits=n_splits)
    best, best_score = grid[0], np.inf
    for params in grid:
        scores = []
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = fit(make_model(kind, **params), X[tr], y[tr])
            scores.append(brier(y[te], predict_proba(m, X[te])))
        if scores and np.mean(scores) < best_score:
            best, best_score = params, float(np.mean(scores))
    return best


def top_features(model, X, cols, k: int = 10) -> list[int]:
    """Indices of the k features the source model relies on most.
    Uses |coef| for LR and a fast permutation-free proxy for HGB."""
    final = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(final, "coef_"):
        coef = np.abs(final.coef_.ravel())
        return list(np.argsort(coef)[::-1][:k])
    # HGB: variance of predictions when a feature is shuffled once (cheap)
    rng = np.random.default_rng(SEED)
    base = predict_proba(model, X)
    imp = np.zeros(X.shape[1])
    idx = rng.choice(len(X), size=min(400, len(X)), replace=False)
    Xs = X[idx]
    base_s = base[idx]
    for j in range(X.shape[1]):
        Xp = Xs.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[j] = np.mean(np.abs(predict_proba(model, Xp) - base_s))
    return list(np.argsort(imp)[::-1][:k])
