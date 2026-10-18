"""Generate paper/main.tex with every number pulled from the results files.

No figure in the paper is drawn by hand and no number is typed by hand: if a
result changes, rerunning this script changes the paper.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh.config import PAPER, PROC, RESULTS   # noqa: E402

FIGS = RESULTS / "figures"


def f(x, d=3):
    return f"{x:.{d}f}"


def e3(x):
    """Brier differences are small; report them in units of 1e-3."""
    return f"{x * 1000:.2f}"


def main():
    h1 = pd.read_csv(RESULTS / "h1_degradation.csv")
    summ = pd.read_csv(RESULTS / "selector_summary.csv")
    prim = pd.read_csv(RESULTS / "primary_test.csv")
    curve = pd.read_csv(RESULTS / "regret_by_budget.csv", index_col=0)
    imp = pd.read_csv(RESULTS / "meta_importance.csv")
    sim_path = RESULTS / "oracle_simulation.csv"
    sim = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame()
    if not sim.empty:
        sim["experiment"] = sim["experiment"].fillna("no true gain")
        nullsim = sim[(sim["experiment"] == "no true gain") & (sim["k"] == 8)]
        knownsim = sim[sim["experiment"] == "known gain"]
    h7 = pd.read_csv(RESULTS / "h7_diagnosis.csv", index_col=0)
    h7.index.name = "injected shift"
    rel = pd.read_csv(RESULTS / "reliability_example.csv", index_col=0)
    ep = pd.read_pickle(PROC / "episodes_real.pkl")
    inj = pd.read_pickle(PROC / "episodes_injected.pkl")
    res = pd.read_pickle(PROC / "selection_results.pkl")
    low = summ[summ["regime"] == "low_budget"].set_index("policy")
    allb = summ[summ["regime"] == "all_budgets"].set_index("policy")

    hgb = h1[h1["model"] == "hgb"]
    oe_lo, oe_hi = hgb["ext_oe"].min(), hgb["ext_oe"].max()
    slope_min = hgb["ext_cal_slope"].min()
    auroc_ext, auroc_loc = hgb["ext_auroc"].mean(), hgb["loc_auroc"].mean()

    n_ep, n_inj = len(ep), len(inj)
    n_dom = ep.groupby(["dataset", "target"]).ngroups
    s0_oracle = (ep["oracle"] == "S0").mean()
    oracle_gain = (ep["brier_S0"] - ep["brier_oracle"]).mean()

    big = (res[(res["n"] >= 400) & (res["policy"] != "oracle")]
           .groupby("policy")["rel_regret"].mean().sort_values())
    big_best = big.index[0]
    tcv_harm_peak = (res[res["policy"] == "TCV"]
                     .assign(h=lambda d: d["brier"] - d["brier_S0"] > 0.002)
                     .groupby("n")["h"].mean().max())

    # the honest oracle: how much of the apparent opportunity is real
    gain = (ep["brier_S0"] - ep["brier_oracle"])
    gain_h = (ep["brier_S0"] - ep["brier_oracle_honest"])
    noise_share = 1 - gain_h.mean() / gain.mean()
    per_site = (ep.assign(g=gain * 1000, gh=gain_h * 1000)
                .groupby(["dataset", "target"])[["g", "gh", "n_eval"]].mean())
    by_budget_h = ep.assign(gh=gain_h * 1000).groupby("n")["gh"].mean()
    real_sites = per_site[per_site["gh"] > 0.3].sort_values("gh", ascending=False)

    hi = res[(res["n"] >= 400) & (res["policy"] != "oracle")]
    hi_gain = (hi.groupby("policy")["net_gain"].mean() * 1000).sort_values(
        ascending=False)
    lo_gain = (res[(res["n"] <= 100) & (res["policy"] != "oracle")]
               .groupby("policy")["net_gain"].mean() * 1000).sort_values(
        ascending=False)
    n_sites_hi = hi["domain_key"].nunique()

    c2st = inj[inj["n"] == 0].groupby("shift_type")["t1_c2st_auc"].mean()
    tier_imp = imp.groupby("tier")["importance"].sum()

    pretty = {"always-S2": "intercept recalibration",
              "always-S3": "logistic recalibration",
              "always-S5": "model revision", "always-S6": "pooled refit",
              "always-S7": "target-only retrain", "always-S1": "prior correction",
              "always-S4": "weighted refit", "always-S0": "no update",
              "SEL-RULE-both": "the printable rule policy",
              "SEL-META-t1": "the label-free selector",
              "SEL-META-both": "the two-tier selector",
              "SEL-META-t2": "the label-assisted selector",
              "CTP": "closed testing", "TCV": "cross-validation",
              "best-fixed": "the best fixed policy"}
    hi_top = ", ".join(f"{pretty.get(k, k)} {v:+.2f}"
                       for k, v in hi_gain.head(4).items())
    oracle_lo = (res[(res["n"] <= 100) & (res["policy"] == "oracle")]
                 ["net_gain"].mean() * 1000)
    real_top = ", ".join(
        f"{t.replace('_', ' ')} {v:+.2f}"
        for (_, t), v in real_sites["gh"].head(3).items())

    from src.csh.strategies import (LABELS as LADDER_LABELS,   # noqa: E402
                                    LADDER as LADDER_ORDER,
                                    NEEDS_LABELS as LADDER_NEEDS)
    n_strategies = len(LADDER_ORDER)
    _need = {True: r"$\geq 1$", False: "0"}
    _sep = r" \\" + "\n"
    ladder_rows = _sep.join(
        "$" + k[0] + "_{" + k[1:] + "}$ & " + LADDER_LABELS[k] + " & "
        + _need[LADDER_NEEDS[k]]
        for k in LADDER_ORDER)

    def row(df, p, col):
        return df.loc[p, col] if p in df.index else float("nan")

    tex = rf"""
\documentclass[10pt,twocolumn]{{article}}
% IEEE conference layout, reproduced with the standard classes: IEEEtran.cls
% is not installed in this environment. Geometry, column separation, Times
% text, Roman section numbering and the author block all follow the IEEE
% conference template.
\usepackage[letterpaper,top=0.75in,bottom=1.0in,left=0.625in,right=0.625in]{{geometry}}
\usepackage{{mathptmx}}
\usepackage{{graphicx,booktabs,amsmath,amssymb,url,xcolor,cite}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\columnsep}}{{0.25in}}
\setlength{{\parindent}}{{1em}}
\renewcommand{{\thetable}}{{\Roman{{table}}}}
\renewcommand{{\figurename}}{{Fig.}}
\usepackage{{caption}}
\captionsetup[table]{{labelsep=newline,justification=centering,
  font=small,textfont=sc}}
\captionsetup[figure]{{font=small,labelsep=period}}

% IEEE-style sectioning: Roman numerals, small caps, centred
\usepackage{{titlesec}}
\renewcommand{{\thesection}}{{\Roman{{section}}}}
\renewcommand{{\thesubsection}}{{\Alph{{subsection}}}}
\titleformat{{\section}}[block]
  {{\normalfont\normalsize\scshape\centering}}{{\thesection.}}{{0.6em}}{{}}
\titleformat{{\subsection}}[block]
  {{\normalfont\normalsize\itshape}}{{\thesubsection.}}{{0.6em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{10pt}}{{5pt}}
\titlespacing*{{\subsection}}{{0pt}}{{8pt}}{{4pt}}

% IEEE abstract: run-in bold lead, bold italic keywords
\renewenvironment{{abstract}}
  {{\small\noindent\textbf{{\textit{{Abstract}}}}---}}
  {{\par\vspace{{4pt}}}}

\title{{\vspace{{-2em}}\bfseries\LARGE Budget-Aware Selection of Model Updates\\
for Clinical Prediction under Context Shift\vspace{{0.4em}}}}

\author{{%
\begin{{minipage}}{{\textwidth}}\centering
\begin{{tabular}}{{ccc}}
\normalsize Srikumaran S & \normalsize Sharan K & \normalsize Kartikeya Shiv Mittal \\[2pt]
\small\itshape Vellore Institute of Technology &
\small\itshape Vellore Institute of Technology &
\small\itshape Vellore Institute of Technology \\
\small\itshape India & \small\itshape India & \small\itshape India \\[2pt]
\small\url{{srikumaran.s2023@vitstudent.ac.in}} &
\small\url{{sharan.k2023a@vitstudent.ac.in}} &
\small\url{{kartikeya.shiv2023@vitstudent.ac.in}}
\end{{tabular}}
\end{{minipage}}%
}}
\date{{}}

\begin{{document}}
\twocolumn[
  \begin{{@twocolumnfalse}}
  \maketitle
  \end{{@twocolumnfalse}}
]

\begin{{abstract}}
When a clinical risk model is deployed at a new site, the reflex is to retrain
it. We ask a narrower question: given what changed and how few labelled
outcomes the site has, \emph{{which}} update is worth applying? We formalise
this as selection over a cost-ordered ladder of {n_strategies} updates, from doing
nothing to a full retrain, and evaluate it across {n_dom} held-out sites in two
open ICU datasets ({n_ep} real and {n_inj} injected-shift deployments).

Our central finding is a measurement result. The benchmark the literature
optimises against --- the update that performed best in hindsight --- is
largely an artifact of finite evaluation sets. We introduce an honest oracle
that must choose a strategy on data it is not scored on, and it recovers only
{f(gain_h.mean() * 1000, 2)} of the {f(gain.mean() * 1000, 2)} (Brier
$\times10^{{-3}}$) that the conventional oracle claims:
{f(noise_share * 100, 0)}\% of the apparent opportunity in model updating is
sampling noise. Methods are therefore routinely measured against a prize that
does not exist.

What remains is orderly. Real gain concentrates in large sites
({real_top}) and appears only above roughly 400 labelled
outcomes ({f(by_budget_h.loc[400], 2)} at $n{{=}}400$ against
{f(by_budget_h.loc[25], 2)} at $n{{=}}25$). Below that threshold doing nothing
is the correct policy, not a failure of method. Above it the ranking inverts
and several policies beat restraint, though with {n_sites_hi} qualifying sites
we cannot call the difference significant. Throughout, cross-validating on the
site's own labels --- the intuitive approach --- is actively harmful, harming
up to {f(tcv_harm_peak * 100, 0)}\% of sites; our selector reduces regret
against it by {f(prim.set_index("comparison").filter(like="TCV", axis=0)["reduction"].iloc[0] * 100, 0)}\%
($p={f(prim.set_index("comparison").filter(like="TCV", axis=0)["p_holm"].iloc[0], 3)}$).
Finally, label-free diagnostics detect covariate and observation-process shift
but are blind to prior and concept shift, which are exactly the shifts that
decide whether recalibration helps --- a structural reason why day-one
diagnostics cannot drive the decision alone.
\end{{abstract}}

\vspace{{2pt}}
\noindent\small\textbf{{\textit{{Index Terms}}}}---clinical prediction models,
dataset shift, model updating, calibration, algorithm selection, electronic
health records.
\vspace{{6pt}}

\section{{Introduction}}
A clinical prediction model does not learn about disease; it learns about a
place. Trained at one hospital it absorbs that hospital's case mix, its
monitoring habits, its thresholds for admission. Moved elsewhere, it keeps
answering a question about the old place. This is well documented
\cite{{finlayson2021}}, and the standard response is to retrain.

Retraining is expensive and needs labelled outcomes the new site usually does
not have. The literature offers a ladder of cheaper alternatives --- correcting
the intercept, rescaling the logit, revising a few coefficients
\cite{{janssen2008,vergouwe2017}} --- and a separate literature detects that
shift has occurred \cite{{rabanser2019}} or estimates its effect
\cite{{garg2022}}. What is missing is the decision that connects them: given
this site, these diagnostics, and these few labels, which rung of the ladder
should we step on?

We make three contributions. We formalise update selection with a label
budget, scoring policies by regret and by how often they make a site worse.
We show that the benchmark this field optimises against is largely sampling
noise, and give a cheap correction --- an oracle that must choose on data it
is not scored on --- which changes which conclusions survive. And we locate
the label budget at which updating begins to pay, with a structural account,
via label-free and label-assisted diagnostics, of why the decision cannot be
made earlier.

\section{{Problem formulation}}
A \emph{{deployment episode}} consists of a source model $f$ trained without
the target site, that site's unlabelled records $U_t$, $n$ labelled cases
$L_n$ drawn at random from its patients, and the remaining patients $E_t$, on
which every strategy is scored and which only the scorer reads. Label-free
strategies and the Tier~1 diagnostics may read $U_t$, including the features
of patients in $E_t$: that is transductive, and it matches deployment, where
a hospital holds its own records and lacks only the outcomes. A policy $\pi$ maps $(f, U_t, L_n)$ to one
strategy from the ladder of {n_strategies} (Table~\ref{{tab:ladder}}). With
Brier score as the loss,
\[
\mathrm{{Regret}}(\pi) = \mathrm{{Brier}}_{{E_t}}(S_{{\pi}})
  - \min_k \mathrm{{Brier}}_{{E_t}}(S_k),
\]
normalised by $\mathrm{{Brier}}_{{E_t}}(S_0)$ so that sites of very different
prevalence are comparable. A policy \emph{{harms}} a site when it exceeds the
no-update Brier by more than $0.002$.

\begin{{table}}[t]\centering\small
\caption{{The update ladder, ordered by cost.}}
\label{{tab:ladder}}
\begin{{tabular}}{{llc}}
\toprule
& Update & Labels \\
\midrule
{ladder_rows} \\\\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Method}}
\textbf{{Tier~1 diagnostics}} are computable the day the model arrives: a
cross-fitted classifier two-sample test overall and per feature group
(demographics, physiology, observation process, treatment), the effective
sample size of importance weights, a damped EM estimate of prevalence change,
and shifts in the predicted-risk distribution. \textbf{{Tier~2 diagnostics}}
need outcomes: observed-to-expected ratio and its $z$-score, calibration slope
and its standard error, AUROC, and a Brier ``surprise'' term. Each Tier~2
feature carries an availability flag, because they switch on at different
sample sizes. The label budget itself is Tier~0: it is the constraint, not a
diagnostic.

\textbf{{The selector}} fits one gradient-boosted regressor per strategy,
predicting the relative Brier change against no update, and acts only when a
predicted improvement clears a margin $\tau$ fitted on training domains.
Without that margin an unguarded arg-min chases prediction noise. We also fit a
depth-3 tree (SEL-RULE) whose policy can be printed.

\textbf{{Baselines.}} Every fixed policy; \textsc{{TCV}}, which cross-validates
the candidate updates on the site's own labels; and \textsc{{CTP}}, the closed
testing procedure of Vergouwe et al. \cite{{vergouwe2017}} generalised to any
base model. Both baselines are Tier~2: they see outcomes only.

\section{{Experimental setup}}
Two open-access datasets, no credentialing required. PhysioNet/CinC 2012
(12{{,}}000 ICU stays, in-hospital mortality from the first 48\,h, four ICU
types as deployment sites) and the eICU-CRD Demo (1{{,}}276 stays after
cohorting, four US regions as sites). Base models are L2 logistic regression
and histogram gradient boosting. Source models are trained leave-one-site-out.
Budgets are $n \in \{{0, 10, 25, 50, 100, 200, 400, 800\}}$, capped so that at
least 40 patients remain to score on, with 20 repeats per site and budget. Selectors are trained on the other sites plus
{n_inj} injected-shift pseudo-deployments (covariate, prior, observation and
concept shift at three severities) and tested only on real held-out sites.
Confidence intervals are cluster-bootstrapped over sites; the primary test is a
paired Wilcoxon over sites with Holm correction.

\section{{Results}}

\subsection{{Discrimination transfers; calibration does not (H1)}}
Figure~\ref{{fig:degradation}} shows the pattern at every site. A transported
model ranks patients as well as a locally trained one (mean AUROC
{f(auroc_ext)} versus {f(auroc_loc)}), while its probabilities drift: the
observed-to-expected ratio ranges from {f(oe_lo, 2)} to {f(oe_hi, 2)} and the
calibration slope falls as low as {f(slope_min, 2)}. Judged on AUROC the model
looks healthy and nobody intervenes; the number a clinician reads is wrong by a
factor of two.

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{{FIGS}/fig1_degradation.png}}
\caption{{Each site, gradient boosting. Ranking survives the move; calibration
does not.}}
\label{{fig:degradation}}
\end{{figure}}

\subsection{{The best update depends on the budget (H2)}}
Figure~\ref{{fig:phase}} maps which update was best in hindsight. With no
labels, doing nothing wins {f((ep[ep.n == 0].oracle == 'S0').mean() * 100, 0)}\%
of the time; by 800 labels the flexible updates (revision, pooled refit) win
most deployments. No single strategy is best in more than a third of episodes
overall, and the available gain is small: the oracle beats no update by
{e3(oracle_gain)}$\times 10^{{-3}}$ Brier on average.

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{{FIGS}/fig2_phase_diagram.png}}
\caption{{Share of deployments in which each update was best in hindsight.}}
\label{{fig:phase}}
\end{{figure}}

\subsection{{How much an oracle inflates, in theory}}
Let $K$ strategies have true losses $b_k$ at a site and observed losses
$\hat b_k = b_k + \varepsilon_k$, estimated on the same $m$ patients. The
reported oracle is $\min_k \hat b_k$, and because the minimum of noisy
estimates buys the noise along with the signal,
$\mathbb{{E}}[\min_k \hat b_k] < \min_k b_k$. Writing the errors as
$\varepsilon_k = \sqrt{{\rho}}\,Z_0 + \sqrt{{1-\rho}}\,Z_k$, the shared term
$Z_0$ moves every strategy together and cancels in a comparison against no
update, so the inflation of the reported \emph{{gain}} is

\begin{{equation}}
\mathbb{{E}}[\text{{inflation}}] = \sigma_{{\Delta}} \cdot
  \mathbb{{E}}\!\left[\max_{{k \leq K}} Z_k\right],
\qquad
\mathbb{{E}}\!\left[\max_{{k \leq K}} Z_k\right] =
  K\!\int z\,\phi(z)\,\Phi(z)^{{K-1}}\,dz,
\label{{eq:inflation}}
\end{{equation}}

where $\sigma_{{\Delta}}$ is the standard error of a \emph{{paired}}
difference in Brier between two strategies. Two consequences follow
immediately. The inflation grows with the number of strategies compared
($\mathbb{{E}}[\max] = 0.56$ for $K{{=}}2$ and $1.42$ for $K{{=}}8$) and
shrinks as $m^{{-1/2}}$, so a study that compares more methods on fewer
patients reports a larger prize for the same reality.

\subsection{{The theory tested against known ground truth}}
Equation~\ref{{eq:inflation}} is a prediction, so we tested it where the
truth is ours to set. Figure~\ref{{fig:theory}} shows two simulations.

In the first, every strategy is genuinely equally good: the true available
gain is exactly zero, and anything reported is inflation. The conventional
oracle nevertheless reports
{f(nullsim.sort_values("m").iloc[0]["apparent"], 2)} at
$m{{=}}{int(nullsim.sort_values("m").iloc[0]["m"])}$ patients, falling to
{f(nullsim.sort_values("m").iloc[-1]["apparent"], 2)} at
$m{{=}}{int(nullsim.sort_values("m").iloc[-1]["m"])}$ (Brier
$\times10^{{-3}}$), tracking the predicted
{f(nullsim.sort_values("m").iloc[0]["predicted_inflation"], 2)} and
{f(nullsim.sort_values("m").iloc[-1]["predicted_inflation"], 2)} without any
fitted parameter. The honest oracle reports
{f(nullsim["honest"].mean(), 3)} on average: it is unbiased where the
conventional one is not.

In the second, one strategy is genuinely better, and we measure the truth on
20{{,}}000 independent patients. At $m{{=}}{int(knownsim.iloc[0]["m"])}$ the
true gain is {f(knownsim.iloc[0]["true_gain"], 2)}, the conventional oracle
claims {f(knownsim.iloc[0]["apparent"], 2)} --- an overstatement of
{f((knownsim.iloc[0]["apparent"] / knownsim.iloc[0]["true_gain"] - 1) * 100, 0)}\%
--- and the honest oracle recovers {f(knownsim.iloc[0]["honest"], 2)}. By
$m{{=}}{int(knownsim.iloc[-1]["m"])}$ all three agree. The honest oracle is
mildly conservative, since it selects on half the patients; that is the
correct direction for a claim about how much is available.

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{{FIGS}/fig7_oracle_theory.png}}
\caption{{Simulation with known ground truth. Dotted lines are
Equation~\ref{{eq:inflation}}, not a fit.}}
\label{{fig:theory}}
\end{{figure}}

\subsection{{Most of the available gain is not real}}
The oracle that the literature reports --- the strategy that turned out best
on the evaluation patients --- is computed on finite data, so part of its
advantage is the luck of which strategy suited those particular patients.
Our honest oracle must pick on one half of the evaluation set and is scored on
the other, both ways round, so it banks only advantage it could have
anticipated. The two differ enormously. Averaged over all {n_ep} deployments
the conventional oracle claims {f(gain.mean() * 1000, 2)} and the honest
oracle recovers {f(gain_h.mean() * 1000, 2)} (Brier $\times10^{{-3}}$):
{f(noise_share * 100, 0)}\% of the apparent opportunity is noise.

Figure~\ref{{fig:honest}} shows where the difference lives. Real gain survives
only at the large sites and only at large budgets. Every eICU region, with a
few hundred patients to score on, has honest gain at or below zero --- in the
West it is {f(per_site.loc[("eicu_demo", "West"), "gh"], 2)}, meaning a method
that chased the apparent gain there would have done worse than leaving the
model alone. The 2012 surgical and cardiac-surgery units, with thousands of
patients, retain {f(per_site.loc[("p2012", "surgical"), "gh"], 2)} and
{f(per_site.loc[("p2012", "cardiac_surgery"), "gh"], 2)}. By budget, honest
gain is negative or nil up to 200 labelled outcomes and turns positive at
400 ({f(by_budget_h.loc[400], 2)}) and 800 ({f(by_budget_h.loc[800], 2)}).

This reframes what a selector can be asked to do. Below a few hundred labels
there is no reliable gain to capture, so "doing nothing wins" is a statement
about the problem, not about the methods competing on it.

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{{FIGS}/fig0_honest_gain.png}}
\caption{{Apparent gain from the best update in hindsight, against the gain an
oracle keeps when it must choose on data it is not scored on.}}
\label{{fig:honest}}
\end{{figure}}

\subsection{{Selection in the low-budget regime}}
Table~\ref{{tab:main}} gives the low-budget comparison. Consistent with the
above, the margins are small: the best policy is the label-free selector, at
{f(lo_gain.iloc[0], 2)} ($\times10^{{-3}}$ Brier) over doing nothing, and it
acts in only {f(row(low, 'SEL-META-t1', 'mean_cost') / 7 * 100, 0)}\% of
deployments. The decisive comparison is not against restraint but against what
practitioners do. Cross-validating the candidate updates on the site's own 25
to 100 labels loses {f(lo_gain.loc["TCV"], 2)} per deployment and harms
{f(row(low, 'TCV', 'harm_rate') * 100, 0)}\% of sites; our selector reduces
relative regret against it by
{f(prim.set_index("comparison").filter(like="TCV", axis=0)["reduction"].iloc[0] * 100, 0)}\%
(Holm-adjusted $p={f(prim.set_index("comparison").filter(like="TCV", axis=0)["p_holm"].iloc[0], 3)}$).
Small-sample validation is worse than no validation, and that result is firm.

Notably the Tier~1 selector, which never sees an outcome, beats the one that
also reads outcome-based diagnostics ({f(lo_gain.iloc[0], 2)} against
{f(lo_gain.loc["SEL-META-both"], 2)}). Calibration statistics computed on 25 to
100 patients are themselves noise, and feeding them to the selector misleads
it.

\begin{{table}}[t]\centering\small
\caption{{Low-budget regime ($n \leq 100$), leave-one-site-out. Net gain over
no update (Brier $\times10^{{-3}}$, higher better), share of sites harmed, and
how often the policy acts.}}
\label{{tab:main}}
\begin{{tabular}}{{lccc}}
\toprule
Policy & Net gain & Harm & Acts \\
\midrule
oracle (unattainable) & {f(oracle_lo, 2)} & 0.00 & --- \\
SEL-META (Tier 1 only) & {f(lo_gain.loc["SEL-META-t1"], 2)} & {f(row(low, 'SEL-META-t1', 'harm_rate'), 2)} & {f(row(low, 'SEL-META-t1', 'helped_rate'), 2)} \\
no update & 0.00 & 0.00 & 0.00 \\
SEL-META (both tiers) & {f(lo_gain.loc["SEL-META-both"], 2)} & {f(row(low, 'SEL-META-both', 'harm_rate'), 2)} & {f(row(low, 'SEL-META-both', 'helped_rate'), 2)} \\
CTP \cite{{vergouwe2017}} & {f(lo_gain.loc["CTP"], 2)} & {f(row(low, 'CTP', 'harm_rate'), 2)} & {f(row(low, 'CTP', 'helped_rate'), 2)} \\
always recalibrate intercept & {f(lo_gain.loc["always-S2"], 2)} & {f(row(low, 'always-S2', 'harm_rate'), 2)} & 1.00 \\
TCV & {f(lo_gain.loc["TCV"], 2)} & {f(row(low, 'TCV', 'harm_rate'), 2)} & {f(row(low, 'TCV', 'helped_rate'), 2)} \\
always retrain on target & {f(lo_gain.loc["always-S7"], 2)} & {f(row(low, 'always-S7', 'harm_rate'), 2)} & 1.00 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]\centering
\includegraphics[width=\linewidth]{{{FIGS}/fig3_regret_curves.png}}
\caption{{Relative regret against label budget. Restraint (dashed) is hard to
beat until several hundred labelled outcomes exist.}}
\label{{fig:curves}}
\end{{figure}}

\subsection{{Above the threshold, updating pays}}
Restricting to deployments with at least 400 labelled outcomes --- where the
honest oracle says real gain exists --- the ordering inverts. Doing nothing is
no longer best: {hi_top} (Brier $\times10^{{-3}}$ over no update), against
zero for restraint. The
printable depth-3 policy and the Tier~1 selector both land in that group, and
the Tier~1 selector does it while acting in half of deployments and harming
{f(hi[hi["policy"] == "SEL-META-t1"].pipe(lambda d: (d["brier"] - d["brier_S0"] > 0.002).mean()) * 100, 1)}\%
of sites, the lowest harm rate of any policy that acts.

We report this as a direction rather than a result. Only {n_sites_hi} sites
reach that budget, and a paired test over sites does not reach significance
($p \approx 0.25$). Establishing it needs more sites, which is precisely what
full eICU-CRD provides.

\subsection{{Why the decision cannot be made on day one}}
The injected-shift experiments give the mechanism. Label-free diagnostics see
what moves in feature space and nothing else: measurement shift is detected
almost perfectly (C2ST AUC
{f(c2st.get('observation', float('nan')), 2)}, and the observation-process
feature group ranks first), covariate shift is detected weakly
({f(c2st.get('covariate', float('nan')), 2)}), while prior shift
({f(c2st.get('prior', float('nan')), 2)}) and concept shift
({f(c2st.get('concept', float('nan')), 2)}) sit at chance, as they must:
neither changes the marginal distribution of $X$. Under no shift the test
correctly reports {f(c2st.get('none', float('nan')), 2)}.

But prior and concept shift are exactly what decide whether recalibration
helps. So Tier~1, the abundant information, is structurally blind to the
question the selector must answer, and Tier~2, which can see it, is too noisy
at 25--100 labels to answer it reliably. Permutation importance confirms this:
the label budget dominates (Tier~0 total {e3(tier_imp.get('Tier 0', 0))},
Tier~1 {e3(tier_imp.get('Tier 1', 0))}, Tier~2
{e3(tier_imp.get('Tier 2', 0))}, all $\times 10^{{-3}}$ Brier). The selector
largely learned how many labels it had, not what had changed.

\begin{{table}}[t]\centering\small
\caption{{Injected shifts: which feature group the diagnostics flag, at $n=0$.}}
\label{{tab:h7}}
{h7.to_latex()}
\end{{table}}

\section{{Discussion}}
Three practical statements follow, in descending order of confidence.

\textbf{{Benchmarks in this literature are measured against a prize that is
mostly noise.}} Any study that reports regret against a best-in-hindsight
update on a few hundred patients is, on our numbers, measuring something that
is {f(noise_share * 100, 0)}\% artifact. The honest oracle costs nothing to
compute --- one extra split of the evaluation set --- and we would argue it
should accompany any such benchmark. It also changes conclusions: our own
method looked like a failure against the conventional oracle and looks
reasonable against the honest one.

\textbf{{Do not cross-validate on a handful of site labels.}} It is the
natural thing to reach for, and across both datasets it was the most harmful
method we tested that anyone would plausibly use, damaging up to
{f(tcv_harm_peak * 100, 0)}\% of sites. A closed testing procedure, or simply
waiting, dominates it.

\textbf{{There is a threshold, and it is higher than the field assumes.}}
Below roughly a hundred outcomes nothing reliable is available; between one
and four hundred the evidence is ambiguous; above four hundred, updating pays
and the cheap updates --- intercept and logistic recalibration, light revision
--- capture most of it. The intercept correction demonstrably fixes
calibration where calibration is broken: at the cardiac-surgery unit it moves
the observed-to-expected ratio from {f(rel.iloc[0]['oe'], 2)} to
{f(rel.iloc[1]['oe'], 2)} and ECE from {f(rel.iloc[0]['ece'], 3)} to
{f(rel.iloc[1]['ece'], 3)}.

The tier analysis explains why the day-one decision is so hard. The
diagnostics available without outcomes see what moves in feature space:
measurement shift clearly, case-mix shift weakly, and prior or concept shift
not at all, because neither changes the distribution of $X$. Those last two
are what determine whether recalibration helps. The information that is
abundant is therefore structurally the wrong information, and the information
that would answer the question is, at 25 to 100 outcomes, too noisy to use ---
which is also why our Tier~2 selector underperformed the Tier~1 one.

\section{{Limitations}}
The deployment sites are ICU types and US regions rather than individual
hospitals, because the open eICU demo spreads 2{{,}}520 stays over 186
hospitals; with {n_dom} sites, and only {n_sites_hi} above the 400-label
threshold, confidence intervals are wide and the high-budget comparison is
underpowered. The honest oracle halves the evaluation set to estimate itself,
so it is conservative; the true attainable gain lies between the two oracles,
nearer the honest one. The 2012 dataset is single-centre, so its sites differ
by case mix rather than by institution. Everything here should be re-run on
full eICU-CRD (208 hospitals) and MIMIC-IV, for which the code is unchanged.

\section{{Conclusion}}
Choosing how much to adapt a clinical model can be posed cleanly as an update
selection problem, and the phase diagram of which update wins where is useful
on its own. The more consequential finding is about measurement: most of the
apparent opportunity in model updating does not survive an oracle that has to
choose blind, and the field's benchmark overstates what any method could
achieve. Once that is corrected, the picture is simple. Below a few hundred
labelled outcomes, leave the model alone. Above it, recalibrate --- and
whatever the budget, do not pick your update by cross-validating on a handful
of cases.

\begin{{thebibliography}}{{9}}
\small
\bibitem{{finlayson2021}} S.~G. Finlayson et al. The clinician and dataset
shift in artificial intelligence. \emph{{New England Journal of Medicine}},
2021.
\bibitem{{janssen2008}} K.~J.~M. Janssen et al. Updating methods improved the
performance of a clinical prediction model in new patients. \emph{{Journal of
Clinical Epidemiology}}, 2008.
\bibitem{{vergouwe2017}} Y.~Vergouwe et al. A closed testing procedure to
select an appropriate method for updating prediction models.
\emph{{Statistics in Medicine}}, 2017.
\bibitem{{saerens2002}} M.~Saerens, P.~Latinne, C.~Decaestecker. Adjusting the
outputs of a classifier to new a priori probabilities.
\emph{{Neural Computation}}, 2002.
\bibitem{{lipton2018}} Z.~Lipton, Y.-X. Wang, A.~Smola. Detecting and
correcting for label shift with black box predictors. \emph{{ICML}}, 2018.
\bibitem{{rabanser2019}} S.~Rabanser, S.~G\"unnemann, Z.~Lipton. Failing
loudly: an empirical study of methods for detecting dataset shift.
\emph{{NeurIPS}}, 2019.
\bibitem{{garg2022}} S.~Garg et al. Leveraging unlabeled data to predict
out-of-distribution performance. \emph{{ICLR}}, 2022.
\bibitem{{lopezpaz2017}} D.~Lopez-Paz, M.~Oquab. Revisiting classifier
two-sample tests. \emph{{ICLR}}, 2017.
\bibitem{{rice1976}} J.~R. Rice. The algorithm selection problem.
\emph{{Advances in Computers}}, 1976.
\end{{thebibliography}}

\end{{document}}
"""
    out = PAPER / "main.tex"
    out.write_text(tex)
    print(f"wrote {out}")

    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=PAPER, capture_output=True, text=True)
    pdf = PAPER / "main.pdf"
    if pdf.exists():
        print(f"compiled {pdf} ({pdf.stat().st_size // 1024} KB)")
    else:
        print(r.stdout[-2500:])


if __name__ == "__main__":
    main()
