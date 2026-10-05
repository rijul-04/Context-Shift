"""Selectors and the baselines they must beat.

A selector sees the diagnostics of an episode and names one strategy. It never
sees the evaluation half. Variants differ in which tiers of diagnostics they
may read, which is how we test whether label-free information is worth
anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, export_text

from .models import guarded

from .config import EPS_TIE, SEED
from .diagnostics import TIER0, TIER1, TIER2, TIER2_FLAGS
from .strategies import COST_RANK, LADDER, NEEDS_LABELS

FEATURE_SETS = {
    "t1": TIER0 + TIER1,                       # label-free only
    "t2": TIER0 + TIER2 + TIER2_FLAGS,         # label-assisted only
    "both": TIER0 + TIER1 + TIER2 + TIER2_FLAGS,
}


def _feasible(row) -> list[str]:
    """Strategies a site could actually apply, given its labels."""
    n = int(round(np.expm1(row["t0_log_n"])))
    if n == 0:
        return [k for k in LADDER if not NEEDS_LABELS[k]]
    return list(LADDER)


def _pick_with_tie_break(pred: dict[str, float], feasible: list[str],
                         eps: float = EPS_TIE) -> str:
    best = min(feasible, key=lambda k: pred[k])
    cheap = [k for k in feasible if pred[k] <= pred[best] + eps]
    return min(cheap, key=lambda k: COST_RANK[k])


TAU_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]


class MetaSelector:
    """One regressor per strategy, predicting how much that update changes the
    Brier score *relative to doing nothing*.

    Two design points matter, and both come from what the data showed:

    1. The target is the relative change (brier_k - brier_S0) / brier_S0, not
       the absolute difference. Brier scores differ several-fold between a
       cardiac-surgery unit and a medical ICU, so an absolute target teaches
       the model the site's prevalence rather than the value of the update.

    2. A required margin tau: leave the model alone unless the predicted
       improvement beats tau. Predictions are noisy and most updates are
       slightly harmful, so an unguarded argmin acts on noise. tau is fitted
       on training domains only, never on the target.
    """

    def __init__(self, tiers: str = "both", seed: int = SEED,
                 tau: float | None = None):
        self.cols = FEATURE_SETS[tiers]
        self.tiers = tiers
        self.seed = seed
        self.tau = tau
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _fit_models(self, df: pd.DataFrame):
        X = df[self.cols].to_numpy(dtype=float)
        base = df["brier_S0"].to_numpy(dtype=float)
        models = {}
        for k in LADDER:
            if k == "S0":
                continue
            y = ((df[f"brier_{k}"].to_numpy(dtype=float) - base)
                 / np.maximum(base, 1e-4))
            m = guarded(HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.06,
                min_samples_leaf=20, early_stopping=False,
                random_state=self.seed))
            m.fit(X, y)
            models[k] = m
        return models

    def fit(self, df: pd.DataFrame):
        self.models = self._fit_models(df)
        if self.tau is None:
            self.tau = self._tune_tau(df)
        return self

    def _tune_tau(self, df: pd.DataFrame) -> float:
        """Inner leave-one-domain-out on the training episodes."""
        if "domain_key" not in df.columns:
            return 0.02
        doms = sorted(df["domain_key"].unique())
        if len(doms) < 3:
            return 0.02
        scores = {t: [] for t in TAU_GRID}
        for d in doms:
            inner_tr = df[df["domain_key"] != d]
            inner_te = df[df["domain_key"] == d]
            if inner_tr.empty or inner_te.empty:
                continue
            sub = MetaSelector(self.tiers, self.seed, tau=0.0)
            sub.models = sub._fit_models(inner_tr)
            gains = sub.predict_gain(inner_te)
            for t in TAU_GRID:
                picks = sub._select_from_gains(inner_te, gains, tau=t)
                b = np.array([inner_te.iloc[i][f"brier_{p}"]
                              for i, p in enumerate(picks)])
                scores[t].append(float(np.mean(
                    (b - inner_te["brier_oracle"].to_numpy())
                    / np.maximum(inner_te["brier_S0"].to_numpy(), 1e-4))))
        means = {t: float(np.mean(v)) for t, v in scores.items() if v}
        return min(means, key=means.get) if means else 0.02

    def predict_gain(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].to_numpy(dtype=float)
        out = {"S0": np.zeros(len(df))}
        for k, m in self.models.items():
            out[k] = m.predict(X)
        return pd.DataFrame(out, index=df.index)

    def _select_from_gains(self, df: pd.DataFrame, gains: pd.DataFrame,
                           tau: float) -> list[str]:
        picks = []
        for i, (_, row) in enumerate(df.iterrows()):
            pred = {k: float(gains.iloc[i][k]) for k in LADDER}
            feasible = _feasible(row)
            # act only if some update is predicted to beat doing nothing by tau
            candidates = [k for k in feasible if k != "S0" and pred[k] < -tau]
            if not candidates:
                picks.append("S0")
                continue
            picks.append(_pick_with_tie_break(pred, candidates))
        return picks

    def select(self, df: pd.DataFrame) -> list[str]:
        gains = self.predict_gain(df)
        tau = 0.02 if self.tau is None else self.tau
        return self._select_from_gains(df, gains, tau=tau)


class RuleSelector:
    """A depth-3 tree on the oracle label: the policy can be printed."""

    def __init__(self, tiers: str = "both", seed: int = SEED, depth: int = 3):
        self.cols = FEATURE_SETS[tiers]
        self.tiers = tiers
        self.tree = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=25, random_state=seed)

    def fit(self, df: pd.DataFrame):
        X = np.nan_to_num(df[self.cols].to_numpy(dtype=float), nan=0.0)
        self.tree.fit(X, df["oracle"].to_numpy())
        return self

    def select(self, df: pd.DataFrame) -> list[str]:
        X = np.nan_to_num(df[self.cols].to_numpy(dtype=float), nan=0.0)
        proba = self.tree.predict_proba(X)
        classes = list(self.tree.classes_)
        picks = []
        for i, (_, row) in enumerate(df.iterrows()):
            feas = _feasible(row)
            scores = {k: (proba[i][classes.index(k)] if k in classes else 0.0)
                      for k in feas}
            best = max(scores.values())
            tied = [k for k in feas if scores[k] >= best - 1e-9]
            picks.append(min(tied, key=lambda k: COST_RANK[k]))
        return picks

    def rules(self) -> str:
        return export_text(self.tree, feature_names=list(self.cols))


# --------------------------------------------------------------------------
# baselines: what people do today
# --------------------------------------------------------------------------
def fixed_policy(df: pd.DataFrame, strategy: str) -> list[str]:
    return [strategy if (strategy in _feasible(r)) else "S0"
            for _, r in df.iterrows()]


def best_fixed_on(train: pd.DataFrame) -> str:
    """The single strategy with the lowest mean Brier on meta-training data.
    Chosen without ever looking at the target domain."""
    means = {k: train[f"brier_{k}"].mean() for k in LADDER}
    return min(means, key=means.get)


def oracle_policy(df: pd.DataFrame) -> list[str]:
    return list(df["oracle"])


def selection_results(df: pd.DataFrame, picks: list[str]) -> pd.DataFrame:
    """Score a policy's picks: achieved Brier, regret, harm."""
    b = np.array([df.iloc[i][f"brier_{p}"] for i, p in enumerate(picks)])
    out = pd.DataFrame({
        "pick": picks,
        "brier": b,
        "brier_oracle": df["brier_oracle"].to_numpy(),
        "brier_S0": df["brier_S0"].to_numpy(),
        "cost": [COST_RANK[p] for p in picks],
    }, index=df.index)
    out["regret"] = out["brier"] - out["brier_oracle"]
    # normalised by the do-nothing error: "how much worse than the best
    # possible update, as a share of what the un-updated model already costs".
    # (Normalising by the oracle *gap* explodes whenever doing nothing is
    # already optimal, which happens in 29% of episodes.)
    out["rel_regret"] = out["regret"] / np.maximum(out["brier_S0"], 1e-4)
    out["net_gain"] = out["brier_S0"] - out["brier"]
    out["correct"] = [p == o for p, o in zip(picks, df["oracle"])]
    return out
