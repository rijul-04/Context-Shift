# ContextShift-Health

Budget-aware selection of model updates for clinical prediction under context shift.

When a clinical risk model moves to a new site, the reflex is to retrain it. This
project asks a narrower question: given what changed and how few labelled
outcomes the new site has, **which** update is worth applying? A selector reads
label-free shift diagnostics plus the site's label budget and picks one of eight
updates, from doing nothing to a full retrain.

## Requirements

Python 3.11+ and the packages in `requirements.txt` (numpy, pandas,
scikit-learn, scipy, joblib, matplotlib). No GPU, no internet at run time.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (not in this repository)

Both datasets are open access on PhysioNet and need no credentialing. Download
them yourself and place them as shown — the repository ships no patient data.

| Dataset | Page | Files needed |
| --- | --- | --- |
| eICU-CRD Demo v2.0.1 | https://physionet.org/content/eicu-crd-demo/2.0.1/ | `patient`, `hospital`, `apacheApsVar`, `apachePredVar`, `apachePatientResult`, `lab`, `vitalPeriodic` (`.csv.gz`) |
| PhysioNet/CinC 2012 | https://physionet.org/content/challenge-2012/1.0.0/ | `set-a/b/c.tar.gz` (extracted) and `Outcomes-a/b/c.txt` |

```
data/raw/
  eicu_demo/*.csv.gz
  p2012/
    set-a/  set-b/  set-c/          # extracted, one .txt per patient
    Outcomes-a.txt  Outcomes-b.txt  Outcomes-c.txt
```

## Running

```bash
make cohorts     # build both cohorts               (~15 s)
make test        # sanity + leakage tests           (~30 s)
make h1          # cross-domain degradation         (~5 min)
make episodes    # the update-ladder experiments
make selectors   # train and evaluate the selectors
make figures     # tables and figures into results/
```

Or call the scripts directly: `python3 scripts/02_source_models.py`.

## Layout

```
src/csh/
  config.py        seeds, budgets, thresholds
  data_p2012.py    2012 cohort: 12,000 stays, domains = 4 ICU types
  data_eicu.py     eICU Demo cohort: domains = 4 US regions
  features.py      feature groups G1-G4, leakage guards
  models.py        logistic regression and gradient boosting
  strategies.py    the update ladder S0-S7
  splits.py        leave-one-domain-out, adaptation/evaluation halves
  metrics.py       Brier, ECE, calibration slope, regret, harm
scripts/           numbered pipeline steps
tests/             run with `python3 tests/run_tests.py`
results/           generated tables and figures
```

## Feature groups

Every feature belongs to one group, which is what lets the diagnostics say
*what* changed rather than only *that* something changed.

- **G1** demographics, admission details, comorbidity
- **G2** physiology values
- **G3** the observation process: how often things were measured, what is missing
- **G4** treatment

## Design rules

- Hospitals or ICU types, never patients, define the train/test boundary.
- The evaluation half of a target domain is read only by the scorer.
- Every update strategy sees the same labelled cases; none sees the eval half.
- Leakage guards run as tests, not as good intentions.
