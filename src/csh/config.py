"""Global configuration for ContextShift-Health."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
for _p in (PROC, RESULTS, PAPER):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260921

# Deployment-episode settings
BUDGETS = [0, 10, 25, 50, 100, 200, 400, 800]
N_REPEATS = 8            # day-1 setting; 20 in the full re-run
N_REPEATS_INJECTED = 3
LOW_BUDGET_MAX = 100     # "low-budget regime" for the primary test
DELTA_HARM = 0.002       # Brier worsening vs S0 that counts as harm
EPS_TIE = 0.001          # cheapest strategy within this predicted Brier of best
EVAL_FRACTION = 0.5      # half of each target domain held out for scoring

# Cohort rules
P2012_WINDOW_H = 48      # features from first 48h (challenge definition)
EICU_WINDOW_MIN = 24 * 60

# Feature-group prefixes used by the grouped shift diagnostics
GROUPS = ["G1", "G2", "G3", "G4"]
