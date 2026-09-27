"""Build a self-contained HTML results page (figures embedded as data URIs)."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.csh.config import PROC, RESULTS   # noqa: E402

FIGS = RESULTS / "figures"
OUT = RESULTS / "dashboard.html"


def img(name: str) -> str:
    data = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def main():
    summ = pd.read_csv(RESULTS / "selector_summary.csv")
    low = summ[summ["regime"] == "low_budget"].set_index("policy")
    ep = pd.read_pickle(PROC / "episodes_real.pkl")
    inj = pd.read_pickle(PROC / "episodes_injected.pkl")
    res = pd.read_pickle(PROC / "selection_results.pkl")
    h1 = pd.read_csv(RESULTS / "h1_degradation.csv")
    c2st = inj[inj["n"] == 0].groupby("shift_type")["t1_c2st_auc"].mean()

    big = (res[(res["n"] >= 400) & (res["policy"] != "oracle")]
           .groupby("policy")["rel_regret"].mean().sort_values())
    tcv_harm = (res[res["policy"] == "TCV"]
                .assign(h=lambda d: d["brier"] - d["brier_S0"] > 0.002)
                .groupby("n")["h"].mean().max())

    def r(p, c):
        return low.loc[p, c] if p in low.index else float("nan")

    rows = [
        ("No update", r("always-S0", "rel_regret"), r("always-S0", "harm_rate"), "the baseline"),
        ("Closed testing (CTP)", r("CTP", "rel_regret"), r("CTP", "harm_rate"), "best real method"),
        ("Our selector, both tiers", r("SEL-META-both", "rel_regret"), r("SEL-META-both", "harm_rate"), "hypothesis"),
        ("Our selector, label-free only", r("SEL-META-t1", "rel_regret"), r("SEL-META-t1", "harm_rate"), "hypothesis"),
        ("Printable rule policy", r("SEL-RULE-both", "rel_regret"), r("SEL-RULE-both", "harm_rate"), "hypothesis"),
        ("Cross-validate on site labels (TCV)", r("TCV", "rel_regret"), r("TCV", "harm_rate"), "what people do"),
        ("Always recalibrate intercept", r("always-S2", "rel_regret"), r("always-S2", "harm_rate"), "fixed policy"),
        ("Always retrain on site data", r("always-S7", "rel_regret"), r("always-S7", "harm_rate"), "fixed policy"),
    ]
    table_rows = "\n".join(
        f'<tr><td>{n}</td><td class="num">{a:.4f}</td>'
        f'<td class="num">{b*100:.0f}%</td><td class="tag">{t}</td></tr>'
        for n, a, b, t in rows)

    shift_rows = "\n".join(
        f'<tr><td>{k}</td><td class="num">{v:.2f}</td>'
        f'<td>{"detected" if v > 0.55 else "invisible"}</td></tr>'
        for k, v in c2st.sort_values(ascending=False).items())

    hgb = h1[h1["model"] == "hgb"]
    stats = [
        ("8", "deployment sites", "4 ICU types, 4 US regions"),
        (f"{len(ep)}", "real deployments", "8 label budgets x 8 repeats"),
        (f"{len(inj)}", "injected shifts", "with a known cause"),
        (f"{hgb['ext_oe'].min():.2f}-{hgb['ext_oe'].max():.2f}", "O/E range", "1.0 would be perfect"),
    ]
    stat_html = "\n".join(
        f'<div class="stat"><div class="stat-v">{v}</div>'
        f'<div class="stat-l">{l}</div><div class="stat-s">{s}</div></div>'
        for v, l, s in stats)

    def figure(name, title, caption):
        return f"""<figure>
  <img src="{img(name)}" alt="{title}">
  <figcaption>{caption}</figcaption>
</figure>"""

    html = f"""<title>ContextShift-Health Findings</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --bg: #f7f6f3; --surface: #fffefc; --line: #e3e1db;
  --ink: #16150f; --ink-2: #55534a; --ink-3: #85837a;
  --accent: #2a78d6; --accent-soft: #e8f0fb;
  --good: #0f7a4a; --warn: #b35c00; --bad: #c0392b;
  --serif: "IBM Plex Serif", Georgia, serif;
  --sans: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #17171a; --surface: #1f1f23; --line: #33333a;
    --ink: #f2f1ec; --ink-2: #b9b7ae; --ink-3: #8b8981;
    --accent: #6da7ec; --accent-soft: #1d2a3c;
    --good: #46b883; --warn: #e0973f; --bad: #e57368;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #17171a; --surface: #1f1f23; --line: #33333a;
  --ink: #f2f1ec; --ink-2: #b9b7ae; --ink-3: #8b8981;
  --accent: #6da7ec; --accent-soft: #1d2a3c;
  --good: #46b883; --warn: #e0973f; --bad: #e57368;
}}
body {{
  background: var(--bg); color: var(--ink); font-family: var(--sans);
  font-size: 15.5px; line-height: 1.62; margin: 0;
}}
.wrap {{ max-width: 60rem; margin: 0 auto; padding-block: 3rem 5rem;
  padding-left: 20px; padding-right: 20px; }}
header p.kicker {{ font-family: var(--mono); font-size: .76rem;
  letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
  margin: 0 0 .6rem; }}
h1 {{ font-family: var(--serif); font-size: clamp(1.9rem, 4.4vw, 2.7rem);
  line-height: 1.12; margin: 0 0 .7rem; text-wrap: balance; }}
h2 {{ font-family: var(--serif); font-size: 1.32rem; margin: 3.2rem 0 .4rem;
  text-wrap: balance; }}
h2 .h {{ font-family: var(--mono); font-size: .72rem; color: var(--accent);
  display: block; letter-spacing: .08em; margin-bottom: .3rem; }}
p {{ max-width: 68ch; color: var(--ink-2); margin: .75rem 0; }}
p.lead {{ font-size: 1.06rem; color: var(--ink); }}
strong {{ color: var(--ink); font-weight: 600; }}
.verdict {{ background: var(--surface); border: 1px solid var(--line);
  border-left: 3px solid var(--accent); border-radius: 3px;
  padding: 1.1rem 1.25rem; margin: 2rem 0 0; }}
.verdict h3 {{ font-family: var(--sans); font-size: .95rem; margin: 0 0 .35rem; }}
.verdict p {{ margin: .3rem 0 0; }}
.stats {{ display: grid; gap: 1px; background: var(--line);
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border: 1px solid var(--line); margin: 2.2rem 0 0; }}
.stat {{ background: var(--surface); padding: .95rem 1rem; }}
.stat-v {{ font-family: var(--mono); font-size: 1.45rem; font-weight: 500;
  color: var(--ink); font-variant-numeric: tabular-nums; }}
.stat-l {{ font-size: .87rem; color: var(--ink); }}
.stat-s {{ font-size: .78rem; color: var(--ink-3); }}
figure {{ margin: 1.4rem 0 0; background: var(--surface);
  border: 1px solid var(--line); border-radius: 3px; padding: .85rem; }}
figure img {{ width: 100%; max-width: 100%; display: block; border-radius: 2px; }}
figcaption {{ font-size: .85rem; color: var(--ink-3); margin-top: .7rem;
  padding: 0 .2rem; }}
.tablewrap {{ overflow-x: auto; margin: 1.3rem 0 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem;
  background: var(--surface); border: 1px solid var(--line); }}
th, td {{ text-align: left; padding: .55rem .8rem;
  border-bottom: 1px solid var(--line); }}
th {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ink-3); font-weight: 500; }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums;
  text-align: right; }}
td.tag {{ font-size: .78rem; color: var(--ink-3); }}
tbody tr:nth-child(2) td {{ background: var(--accent-soft); }}
ul {{ max-width: 68ch; color: var(--ink-2); padding-left: 1.1rem; }}
li {{ margin: .4rem 0; }}
code {{ font-family: var(--mono); font-size: .86em; background: var(--surface);
  border: 1px solid var(--line); border-radius: 2px; padding: .1em .35em; }}
pre {{ background: var(--surface); border: 1px solid var(--line);
  border-radius: 3px; padding: .9rem 1rem; overflow-x: auto;
  font-family: var(--mono); font-size: .84rem; color: var(--ink-2); }}
hr {{ border: 0; border-top: 1px solid var(--line); margin: 3rem 0 0; }}
footer {{ color: var(--ink-3); font-size: .85rem; margin-top: 1.4rem; }}
</style>

<div class="wrap">
<header>
  <p class="kicker">ContextShift-Health &middot; experiment report</p>
  <h1>When a risk model moves hospital, should you update it?</h1>
  <p class="lead">We built the full decision machinery &mdash; eight ways to
  update a mortality model, eighteen shift measurements, a selector trained to
  choose between them &mdash; and tested it on 8 deployment sites. The honest
  answer is that below a few hundred labelled outcomes, the best thing to do is
  usually nothing.</p>
</header>

<div class="stats">{stat_html}</div>

<div class="verdict">
  <h3>The one-line verdict</h3>
  <p>Our central hypothesis failed, and the failure has a clean explanation:
  the shifts that decide whether an update helps are exactly the ones that
  cannot be seen without outcome labels.</p>
</div>

<h2><span class="h">Finding 01 &middot; hypothesis H1 supported</span>
Moving the model breaks its probabilities, not its ranking</h2>
<p>At every site the transported model sorts patients about as well as a model
trained on that site's own data. What breaks is the number itself. In cardiac
surgery it predicts roughly twice the deaths that occur; in the eICU Northeast
region it predicts under a third of them.</p>
<p>This matters because AUROC is what most teams monitor. On that metric
nothing looks wrong, while the probability a clinician reads is off by a factor
of two.</p>
{figure("fig1_degradation.png", "Discrimination versus calibration by site",
        "Left: ranking quality, transported model versus locally trained. "
        "Right: observed deaths divided by predicted deaths, where 1.0 is "
        "perfect.")}

<h2><span class="h">Finding 02 &middot; hypothesis H2 supported</span>
Which update is best depends on how many labels you have</h2>
<p>With no labels, doing nothing is best in about seven deployments out of ten.
By 800 labelled outcomes the flexible updates take over. No single strategy is
best in more than a third of cases, which is what makes the selection question
worth asking in the first place.</p>
{figure("fig2_phase_diagram.png", "Which update wins, by label budget",
        "Share of deployments in which each update turned out best. Read a row "
        "as: with this many labels, this is how often each choice was right.")}

<h2><span class="h">Finding 03 &middot; hypothesis H3 rejected</span>
The selector does not beat doing nothing</h2>
<p>This is the result we did not want. Every method is scored by regret against
the best update in hindsight, and by how often it leaves a site worse off than
before. Doing nothing cannot harm anyone by construction, and it is also close
to the best achievable score.</p>
<div class="tablewrap">
<table>
  <thead><tr><th>Policy</th><th class="num">Regret</th>
  <th class="num">Sites harmed</th><th></th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>
</div>
<p>Two things in that table are worth more than our failed hypothesis.
<strong>Cross-validating on the site's own labels harms up to
{tcv_harm*100:.0f}% of sites</strong> &mdash; and that is the approach a careful
practitioner would reach for first. With 25 to 100 outcomes, small-sample
validation is worse than no validation at all. Meanwhile the
<strong>closed testing procedure</strong>, a statistical method from 2017,
quietly does the best job of any real method.</p>
{figure("fig3_regret_curves.png", "Regret against label budget",
        "Lower is better. The dashed line is doing nothing. Every method "
        "sits above it until several hundred labelled outcomes exist.")}
{figure("fig4_policy_comparison.png", "Accuracy and safety side by side",
        "A method has to be both close to optimal and rarely harmful. "
        "Bars show the low-budget regime, up to 100 labelled cases.")}

<h2><span class="h">Finding 04 &middot; hypothesis H7 supported</span>
Why it failed: label-free diagnostics are blind to the shift that matters</h2>
<p>We injected shifts we designed ourselves, so we knew the right answer. The
label-free detector finds changes in what is measured and who is admitted, and
it is exactly at chance for the two shifts that determine whether
recalibration helps.</p>
<div class="tablewrap">
<table>
  <thead><tr><th>Injected shift</th><th class="num">Detection (AUC)</th>
  <th>Verdict</th></tr></thead>
  <tbody>{shift_rows}</tbody>
</table>
</div>
<p>Prior shift means the death rate changed; concept shift means the
relationship between patient and outcome changed. Neither alters the
distribution of the features, so no amount of unlabelled data reveals them.
That is the structural reason the abundant information cannot answer the
question, and it is the finding that makes the negative result interesting
rather than merely disappointing.</p>
{figure("fig6_importance.png", "What the selector actually used",
        "The label budget dominates. The selector largely learned how many "
        "labels it had, not what had changed at the site.")}

<h2><span class="h">Finding 05</span>What we can recommend</h2>
<ul>
  <li><strong>Under ~100 labelled outcomes:</strong> leave the model alone and
  monitor calibration. Acting on this much evidence costs more than it gains.</li>
  <li><strong>If you must act early:</strong> use the closed testing procedure.
  It acts only when a statistical test justifies it, and it harmed
  {r("CTP", "harm_rate")*100:.0f}% of sites here.</li>
  <li><strong>Above ~400 labelled outcomes:</strong> recalibrate the intercept.
  It was the best policy in that regime, at regret {big.iloc[0]:.4f}.</li>
  <li><strong>Do not</strong> cross-validate on a handful of site labels to
  pick an update.</li>
</ul>
{figure("fig5_reliability.png", "Recalibration where it is warranted",
        "Cardiac surgery, 100 labelled cases: the intercept correction pulls "
        "predicted risk back toward observed risk.")}

<hr>
<h2><span class="h">Reproducing this</span>Everything runs from the repository</h2>
<pre>make cohorts     # build both cohorts from the raw PhysioNet files
make test        # 10 sanity and leakage tests
make h1          # Finding 01
make episodes    # the 720 deployment experiments  (~30 min)
make selectors   # Findings 03
make figures     # every figure on this page</pre>
<footer>Datasets: PhysioNet/CinC 2012 and the eICU-CRD Demo, both open access.
No patient data is in the repository. Results here come from demo-scale data;
the same code runs unchanged on full eICU-CRD and MIMIC-IV.</footer>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
