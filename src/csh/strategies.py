"""The update ladder S0-S7, ordered by cost.

Each strategy has signature f(ctx) -> probabilities on ctx.X_eval.

ctx carries: the fitted source model, the source training data, the target's
unlabelled pool, the n labelled target cases, and the evaluation matrix.
Strategies never see ctx.y_eval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .config import SEED
from .metrics import clip
from .models import fit, make_model, predict_proba

LADDER = ["S0", "S1", "S2", "S8", "S3", "S4", "S5", "S6", "S9", "S7"]
LADDER_CORE = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]  # the first 8
COST_RANK = {"S0": 0, "S1": 1, "S2": 2, "S8": 3, "S3": 4, "S4": 5,
             "S5": 6, "S6": 7, "S9": 8, "S7": 9}
NEEDS_LABELS = {"S0": False, "S1": False, "S2": True, "S8": True, "S3": True,
                "S4": False, "S5": True, "S6": True, "S9": True, "S7": True}
LABELS = {
    "S0": "no update", "S1": "prior correction", "S2": "intercept recal",
    "S8": "shrunk intercept", "S3": "logistic recal",
    "S4": "importance-weighted refit", "S5": "model revision",
    "S6": "pooled refit", "S9": "source-target ensemble",
    "S7": "target-only retrain",
}


@dataclass
class Ctx:
    model: object                     # fitted source model
    kind: str                         # "hgb" | "lr"
    X_src: np.ndarray
    y_src: np.ndarray
    X_pool: np.ndarray                # unlabelled target pool
    X_lab: np.ndarray                 # n labelled target cases
    y_lab: np.ndarray
    X_eval: np.ndarray
    top_idx: list[int] = field(default_factory=list)
    cache: dict = field(default_factory=dict)

    # predictions of the source model, computed once
    def p(self, which: str) -> np.ndarray:
        key = f"p_{which}"
        if key not in self.cache:
            X = {"src": self.X_src, "pool": self.X_pool,
                 "lab": self.X_lab, "eval": self.X_eval}[which]
            self.cache[key] = (predict_proba(self.model, X) if len(X)
                               else np.array([]))
        return self.cache[key]


def _usable(ctx: Ctx, min_per_class: int = 2) -> bool:
    if len(ctx.y_lab) == 0:
        return False
    pos = int(np.sum(ctx.y_lab))
    return pos >= min_per_class and (len(ctx.y_lab) - pos) >= min_per_class


# --------------------------------------------------------------------------
# calibration primitives
# --------------------------------------------------------------------------
def fit_intercept(y, p, prior_sd: float = 1.0) -> float:
    """logit(p') = a + logit(p), with a weak N(0, prior_sd^2) prior on a."""
    z = logit(clip(p))
    y = np.asarray(y, dtype=float)

    def nll(a):
        q = expit(a[0] + z)
        q = clip(q)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + \
            0.5 * (a[0] / prior_sd) ** 2

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B")
    return float(res.x[0])


def fit_platt(y, p, l2: float = 0.1):
    """logit(p') = a + b * logit(p); light L2 keeps small n stable."""
    z = logit(clip(p))
    y = np.asarray(y, dtype=float)

    def nll(ab):
        a, b = ab
        q = clip(expit(a + b * z))
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + \
            l2 * (a ** 2 + (b - 1) ** 2)

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def em_prior(p_target, source_prior: float, max_iter: int = 100,
             tol: float = 1e-6, max_ratio: float = 3.0):
    """Saerens-Latinne-Decaestecker EM estimate of the target prevalence.

    Plain EM is unstable when the source model discriminates weakly: the
    iteration can run away to 0 or 1 (we see this on eICU, where AUROC is
    ~0.75). We therefore damp it, constraining the estimate to within
    `max_ratio` of the source prevalence in both directions, and report
    whether the constraint bound (a sign the estimate is untrustworthy).

    Returns (estimated_prior, hit_bound).
    """
    p = clip(p_target)
    pi_s = float(np.clip(source_prior, 1e-4, 1 - 1e-4))
    if len(p) == 0:
        return pi_s, True

    lo = max(pi_s / max_ratio, 1e-4)
    hi = min(pi_s * max_ratio, 1 - 1e-4)

    pi = pi_s
    for _ in range(max_iter):
        w1 = (pi / pi_s) * p
        w0 = ((1 - pi) / (1 - pi_s)) * (1 - p)
        post = w1 / (w1 + w0)
        new = float(np.clip(np.mean(post), lo, hi))
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    hit_bound = bool(pi <= lo * (1 + 1e-6) or pi >= hi * (1 - 1e-6))
    return pi, hit_bound


def apply_prior_shift(p, source_prior: float, target_prior: float):
    p = clip(p)
    pi_s = float(np.clip(source_prior, 1e-4, 1 - 1e-4))
    pi_t = float(np.clip(target_prior, 1e-4, 1 - 1e-4))
    w1 = (pi_t / pi_s) * p
    w0 = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
    return w1 / (w1 + w0)


def domain_weights(X_src, X_pool, clip_pct: float = 99.0, seed: int = SEED):
    """Importance weights from a cross-fitted source-vs-target classifier."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_s, n_t = len(X_src), len(X_pool)
    if n_t < 20 or n_s < 20:
        return np.ones(n_s), 1.0
    X = np.vstack([X_src, X_pool])
    d = np.r_[np.zeros(n_s), np.ones(n_t)]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
    ])
    oof = np.zeros(len(d))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, d):
        pipe.fit(X[tr], d[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    ps = clip(oof[:n_s])
    w = (ps / (1 - ps)) * (n_s / max(n_t, 1))
    hi = np.percentile(w, clip_pct)
    w = np.clip(w, 0.0, hi)
    w = w / max(w.mean(), 1e-8)
    ess = float((w.sum() ** 2) / (np.sum(w ** 2) * len(w)))
    return w, ess


def fit_revision(y, p, Xtop, l2: float = 10.0):
    """logit(p') = a + b*logit(p) + c.x_top, L2 on c only."""
    z = logit(clip(p))
    y = np.asarray(y, dtype=float)
    Z = np.nan_to_num(Xtop, nan=0.0)
    k = Z.shape[1]

    def nll(th):
        a, b, c = th[0], th[1], th[2:]
        q = clip(expit(a + b * z + Z @ c))
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + \
            l2 * float(np.sum(c ** 2)) + 0.1 * (a ** 2 + (b - 1) ** 2)

    res = minimize(nll, x0=np.r_[0.0, 1.0, np.zeros(k)], method="L-BFGS-B")
    return res.x[0], res.x[1], res.x[2:]


def _standardise(Xtop_fit, Xtop_apply):
    mu = np.nanmean(Xtop_fit, axis=0)
    sd = np.nanstd(Xtop_fit, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return ((Xtop_fit - mu) / sd, (Xtop_apply - mu) / sd)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------
def S0(ctx: Ctx):
    return ctx.p("eval")


def S1(ctx: Ctx):
    src_prior = float(np.mean(ctx.y_src))
    pi_t, _ = em_prior(ctx.p("pool"), src_prior)
    return apply_prior_shift(ctx.p("eval"), src_prior, pi_t)


def S2(ctx: Ctx):
    if not _usable(ctx, 1):
        return S1(ctx)
    a = fit_intercept(ctx.y_lab, ctx.p("lab"))
    return expit(a + logit(clip(ctx.p("eval"))))


def S3(ctx: Ctx):
    if not _usable(ctx):
        return S1(ctx)
    a, b = fit_platt(ctx.y_lab, ctx.p("lab"))
    return expit(a + b * logit(clip(ctx.p("eval"))))


def S4(ctx: Ctx):
    key = "S4_model"
    if key not in ctx.cache:
        w, _ = domain_weights(ctx.X_src, ctx.X_pool)
        m = fit(make_model(ctx.kind), ctx.X_src, ctx.y_src, sample_weight=w)
        ctx.cache[key] = m
    return predict_proba(ctx.cache[key], ctx.X_eval)


def S5(ctx: Ctx):
    if not _usable(ctx) or not ctx.top_idx:
        return S3(ctx)
    Zl, Ze = _standardise(ctx.X_lab[:, ctx.top_idx], ctx.X_eval[:, ctx.top_idx])
    a, b, c = fit_revision(ctx.y_lab, ctx.p("lab"), Zl)
    z = logit(clip(ctx.p("eval")))
    return expit(a + b * z + np.nan_to_num(Ze, nan=0.0) @ c)


def S6(ctx: Ctx, target_weight: float = 5.0):
    if not _usable(ctx, 1):
        return S1(ctx)
    X = np.vstack([ctx.X_src, ctx.X_lab])
    y = np.r_[ctx.y_src, ctx.y_lab]
    w = np.r_[np.ones(len(ctx.y_src)), np.full(len(ctx.y_lab), target_weight)]
    m = fit(make_model(ctx.kind), X, y, sample_weight=w)
    return predict_proba(m, ctx.X_eval)


def S7(ctx: Ctx):
    if not _usable(ctx, 2):
        return S1(ctx)
    kind = "lr" if (ctx.kind == "lr" or len(ctx.y_lab) < 100) else "hgb"
    m = fit(make_model(kind, **({"C": 1.0} if kind == "lr" else {})),
            ctx.X_lab, ctx.y_lab)
    return predict_proba(m, ctx.X_eval)


def S8(ctx: Ctx, k0: float = 25.0):
    """Intercept correction shrunk toward no update by the label budget.

    The intercept estimated from n outcomes is itself noisy, and acting on it
    in full is what makes S2 harmful at small n. Shrinking by n / (n + k0) is
    the empirical-Bayes compromise: nothing at n=0, essentially the full
    correction once n greatly exceeds k0.
    """
    if not _usable(ctx, 1):
        return S1(ctx)
    a = fit_intercept(ctx.y_lab, ctx.p("lab"))
    n = len(ctx.y_lab)
    a_shrunk = a * (n / (n + k0))
    return expit(a_shrunk + logit(clip(ctx.p("eval"))))


def S9(ctx: Ctx):
    """Average the transported model with a pooled refit, in logit space.

    Ensembling hedges: when the refit is wrong the source model pulls it
    back, and vice versa. The weight follows the label budget, so a site
    with few outcomes stays close to the model it was given.
    """
    if not _usable(ctx, 1):
        return S1(ctx)
    p_pool = S6(ctx)
    n = len(ctx.y_lab)
    w = n / (n + 200.0)                     # trust the refit only with labels
    z = (1 - w) * logit(clip(ctx.p("eval"))) + w * logit(clip(p_pool))
    return expit(z)


FUNCS = {"S0": S0, "S1": S1, "S2": S2, "S8": S8, "S3": S3,
         "S4": S4, "S5": S5, "S6": S6, "S9": S9, "S7": S7}


def run_all(ctx: Ctx) -> dict[str, np.ndarray]:
    return {k: np.asarray(f(ctx), dtype=float) for k, f in FUNCS.items()}
