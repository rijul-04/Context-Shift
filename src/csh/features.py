"""Feature-matrix assembly and the group map used by the shift diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROC

# Anything that would leak the outcome or the domain identity.
FORBIDDEN_SUBSTR = [
    "apachescore", "predictedhospitalmortality", "predictediculos",
    "dischargelocation", "dischargestatus", "diedinhospital", "hospitaldischarge",
    "unitdischarge", "actualhospitalmortality", "actualiculos", "Length_of_stay",
    "Survival", "SAPS", "SOFA", "saps3", "ICUType", "region", "teachtype",
    "bedcount", "hospitalid", "var03hspxlos", "midur",
]
META_COLS = {"stay_id", "domain", "y", "dataset", "set", "RecordID", "hospitalid"}


def load(dataset: str) -> pd.DataFrame:
    name = {"p2012": "cohort_p2012.pkl", "eicu_demo": "cohort_eicu.pkl"}[dataset]
    return pd.read_pickle(PROC / name)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("G1_", "G2_", "G3_", "G4_"))]


def group_map(cols: list[str]) -> dict[str, list[int]]:
    """Column indices per feature group, for the grouped C2ST."""
    out: dict[str, list[int]] = {"G1": [], "G2": [], "G3": [], "G4": []}
    for i, c in enumerate(cols):
        out[c[:2]].append(i)
    return {k: v for k, v in out.items() if v}


def matrix(df: pd.DataFrame, cols: list[str] | None = None):
    """Return (X, y, domain, cols). X keeps NaN: HGB handles it natively and
    the LR pipeline imputes with source medians."""
    cols = cols or feature_columns(df)
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    dom = df["domain"].to_numpy()
    return X, y, dom, cols


def check_leakage(df: pd.DataFrame) -> None:
    """Raise if any feature could leak the outcome or the domain."""
    cols = feature_columns(df)
    lowered = {c: c.lower() for c in cols}
    for c, lc in lowered.items():
        for bad in FORBIDDEN_SUBSTR:
            if bad.lower() in lc:
                raise AssertionError(f"leaky feature {c!r} matches {bad!r}")
    if not cols:
        raise AssertionError("no features found")
    if set(cols) & META_COLS:
        raise AssertionError("meta column used as feature")
    # A feature that perfectly separates domains would make the C2ST trivial.
    X, y, dom, _ = matrix(df, cols)
    for i, c in enumerate(cols):
        v = X[:, i]
        ok = ~np.isnan(v)
        if ok.sum() < 20:
            continue
        # constant columns carry no information but are not leakage
        if np.nanstd(v) == 0:
            continue
    return None
