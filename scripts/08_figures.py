"""All figures for the paper, generated from the saved results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt                                  # noqa: E402

from src.csh.config import DELTA_HARM, PROC, RESULTS             # noqa: E402
from src.csh.plotstyle import (GRID, INK, INK_2, INK_MUTED,      # noqa: E402
                               POLICY_COLOR, SEQ, SERIES,
                               apply_style, label_end, save)
from src.csh.strategies import LABELS, LADDER                    # noqa: E402

FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
MAIN_POLICIES = ["SEL-META-both", "SEL-META-t1", "TCV", "CTP"]


def fig1_degradation():
    """H1: the ranking survives the move; the probabilities do not."""
    df = pd.read_csv(RESULTS / "h1_degradation.csv")
    d = df[df["model"] == "hgb"].copy()
    d["label"] = d["target"].str.replace("_", " ")
    d = d.sort_values("ext_oe")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ypos = np.arange(len(d))
    ax.hlines(ypos, d["loc_auroc"], d["ext_auroc"], color=GRID, lw=2, zorder=1)
    ax.scatter(d["loc_auroc"], ypos, color=INK_MUTED, label="local model", zorder=2)
    ax.scatter(d["ext_auroc"], ypos, color=SERIES["blue"], label="transported model",
               zorder=3)
    ax.set_yticks(ypos, d["label"], fontsize=8.5)
    ax.set_xlabel("AUROC on the held-out site")
    ax.set_title("Discrimination transfers")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.axvline(1.0, color=INK_MUTED, lw=1.2, ls="--")
    ax.scatter(d["ext_oe"], ypos, color=SERIES["orange"], zorder=3)
    for i, (oe, lab) in enumerate(zip(d["ext_oe"], d["label"])):
        ax.annotate(f"{oe:.2f}", (oe, i), xytext=(0.4, 0), color=INK_2,
                    textcoords="offset fontsize", va="center", fontsize=8)
    ax.set_yticks(ypos, [])
    ax.set_xlabel("observed / expected deaths")
    ax.set_xlim(0.2, max(3.0, d["ext_oe"].max() * 1.25))
    ax.set_title("Calibration does not")
    fig.suptitle("Moving a mortality model to an unseen site (gradient boosting)",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.04)
    save(fig, FIGS / "fig1_degradation.png")


def fig2_phase_diagram(ep: pd.DataFrame):
    """H2: which update wins, by label budget. The map practitioners want."""
    tab = pd.crosstab(ep["n"], ep["oracle"], normalize="index")
    for k in LADDER:
        if k not in tab.columns:
            tab[k] = 0.0
    tab = tab[LADDER]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    data = tab.to_numpy()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=max(0.5, data.max()),
                   aspect="auto")
    short = {"S0": "no update", "S1": "prior\ncorrection",
             "S2": "intercept\nrecal", "S3": "logistic\nrecal",
             "S4": "weighted\nrefit", "S5": "model\nrevision",
             "S6": "pooled\nrefit", "S7": "target-only\nretrain"}
    ax.set_xticks(range(len(LADDER)),
                  [f"{k}\n{short[k]}" for k in LADDER], fontsize=8)
    ax.set_yticks(range(len(tab.index)), [str(i) for i in tab.index])
    ax.set_ylabel("labelled cases at the new site")
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v >= 0.04:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.28 else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("share of deployments where this update was best", fontsize=8.5)
    cb.outline.set_visible(False)
    ax.set_title("No single update wins everywhere")
    save(fig, FIGS / "fig2_phase_diagram.png")


def fig_honest_gain(ep: pd.DataFrame):
    """How much of the apparent opportunity survives having to choose blind."""
    ep = ep.copy()
    ep["apparent"] = (ep["brier_S0"] - ep["brier_oracle"]) * 1000
    ep["honest"] = (ep["brier_S0"] - ep["brier_oracle_honest"]) * 1000

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    ax = axes[0]
    per = (ep.groupby(["dataset", "target"])[["apparent", "honest", "n_eval"]]
           .mean().sort_values("n_eval"))
    y = np.arange(len(per))
    ax.barh(y + 0.19, per["apparent"], height=0.36, color=SERIES["orange"])
    ax.barh(y - 0.19, per["honest"], height=0.36, color=SERIES["blue"])
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, [f"{t}\n{int(n)} patients" for (_, t), n
                      in zip(per.index, per["n_eval"])], fontsize=7.5)
    ax.set_xlabel("gain over no update (Brier $\\times 10^{-3}$)")
    ax.set_title("Per site, smallest first")
    ax.invert_yaxis()                      # smallest site at the top
    handles = [plt.Line2D([], [], color=SERIES["orange"], lw=7,
                          label="apparent gain"),
               plt.Line2D([], [], color=SERIES["blue"], lw=7,
                          label="gain that survives")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)

    ax = axes[1]
    by_n = ep.groupby("n")[["apparent", "honest"]].mean()
    x = np.arange(len(by_n))
    ax.axhline(0, color=INK_2, lw=1)
    ax.plot(x, by_n["apparent"], color=SERIES["orange"], marker="o")
    ax.plot(x, by_n["honest"], color=SERIES["blue"], marker="o")
    label_end(ax, x[-1], by_n["apparent"].iloc[-1], "apparent", SERIES["orange"])
    label_end(ax, x[-1], by_n["honest"].iloc[-1], "real", SERIES["blue"])
    ax.set_xticks(x, [str(i) for i in by_n.index])
    ax.set_xlim(-0.2, len(by_n) + 1.4)
    ax.set_xlabel("labelled cases at the new site")
    ax.set_ylabel("gain over no update ($\\times 10^{-3}$)")
    ax.set_title("Real opportunity appears only above ~400 labels")
    fig.suptitle("Most of the apparent gain from updating is sampling noise",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.04)
    save(fig, FIGS / "fig0_honest_gain.png")


def fig3_regret_curves():
    """H3 and the Tier 1 / Tier 2 crossover."""
    curve = pd.read_csv(RESULTS / "regret_by_budget.csv", index_col=0)
    curve.columns = [int(c) for c in curve.columns]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    budgets = list(curve.columns)
    x = np.arange(len(budgets))

    if "always-S0" in curve.index:
        v0 = curve.loc["always-S0"].to_numpy(dtype=float)
        ax.plot(x, v0, color=INK_MUTED, lw=1.8, ls="--", zorder=2)
        label_end(ax, x[-1], v0[-1], "no update", INK_MUTED)
    for pol in MAIN_POLICIES:
        if pol not in curve.index:
            continue
        v = curve.loc[pol].to_numpy(dtype=float)
        c = POLICY_COLOR[pol]
        ax.plot(x, v, color=c, marker="o", zorder=3)
        last = np.where(np.isfinite(v))[0][-1]
        label_end(ax, x[last], v[last], pol.replace("SEL-META-", "ours: "), c)

    ax.set_xticks(x, [str(b) for b in budgets])
    ax.set_xlabel("labelled cases available at the new site")
    ax.set_ylabel("relative regret vs the oracle update")
    ax.set_xlim(-0.2, len(budgets) - 0.2 + 2.2)
    ax.set_ylim(bottom=0)
    ax.set_title("Below a few hundred labelled cases, restraint is hard to beat")
    save(fig, FIGS / "fig3_regret_curves.png")


def fig4_policy_comparison():
    """Regret and harm side by side: a method must be better AND safe."""
    s = pd.read_csv(RESULTS / "selector_summary.csv")
    d = s[s["regime"] == "low_budget"].copy()
    keep = MAIN_POLICIES + ["SEL-RULE-both", "best-fixed", "always-S0",
                            "always-S3", "always-S7"]
    d = d[d["policy"].isin(keep)].sort_values("rel_regret")
    y = np.arange(len(d))
    colors = [POLICY_COLOR.get(p, INK_MUTED) for p in d["policy"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    ax = axes[0]
    ax.barh(y, d["rel_regret"], color=colors, height=0.62)
    ax.errorbar(d["rel_regret"], y,
                xerr=[d["rel_regret"] - d["ci_lo"], d["ci_hi"] - d["rel_regret"]],
                fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)
    ax.set_yticks(y, d["policy"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("relative regret (lower is better)")
    ax.set_title("How close to the best possible update")

    ax = axes[1]
    ax.barh(y, d["harm_rate"], color=colors, height=0.62)
    ax.axvline(0.05, color=INK_MUTED, ls="--", lw=1.2)
    ax.annotate("5% ceiling", (0.05, len(d) - 0.6), xytext=(0.3, 0),
                textcoords="offset fontsize", color=INK_2, fontsize=8.5)
    ax.set_xlabel(f"share of sites made worse (> {DELTA_HARM} Brier)")
    ax.set_title("How often it does harm")
    fig.suptitle("Low-budget regime (up to 100 labelled cases), "
                 "leave-one-site-out", fontsize=11.5, fontweight="bold",
                 color=INK, y=1.03)
    save(fig, FIGS / "fig4_policy_comparison.png")


def fig5_reliability(ep: pd.DataFrame):
    """What the update actually fixes, in probability space."""
    rel = PROC / "reliability_example.pkl"
    if not rel.exists():
        return
    d = pd.read_pickle(rel)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, ls="--", lw=1.2)
    for name, color in (("before", SERIES["orange"]), ("after", SERIES["blue"])):
        sub = d[d["which"] == name]
        ax.plot(sub["pred"], sub["obs"], color=color, marker="o")
        label_end(ax, sub["pred"].iloc[-1], sub["obs"].iloc[-1], name, color)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed death rate")
    ax.set_title("Recalibration in probability space")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    save(fig, FIGS / "fig5_reliability.png")


def fig6_importance():
    path = RESULTS / "meta_importance.csv"
    if not path.exists():
        return
    d = pd.read_csv(path).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [SERIES["aqua"] if f.startswith("t1_") else
              (SERIES["violet"] if f.startswith("t2_") else INK_MUTED)
              for f in d["feature"]]
    ax.barh(np.arange(len(d)), d["importance"], color=colors, height=0.62)
    ax.set_yticks(np.arange(len(d)), d["feature"], fontsize=8.5)
    ax.set_xlabel("permutation importance (increase in selection regret)")
    ax.set_title("What the selector reads")
    handles = [plt.Line2D([], [], color=SERIES["aqua"], lw=6, label="Tier 1 (label-free)"),
               plt.Line2D([], [], color=SERIES["violet"], lw=6, label="Tier 2 (needs labels)"),
               plt.Line2D([], [], color=INK_MUTED, lw=6, label="label budget")]
    ax.legend(handles=handles, loc="lower right")
    save(fig, FIGS / "fig6_importance.png")


def main():
    apply_style()
    ep = pd.read_pickle(PROC / "episodes_real.pkl")
    print("figures:")
    fig_honest_gain(ep)
    fig1_degradation()
    fig2_phase_diagram(ep)
    fig3_regret_curves()
    fig4_policy_comparison()
    fig5_reliability(ep)
    fig6_importance()


if __name__ == "__main__":
    main()
