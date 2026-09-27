"""PhysioNet/CinC 2012 cohort: in-hospital mortality from the first 48h.

Domains are the four ICU types. Features are grouped:
  G1 demographics/admission, G2 physiology values,
  G3 observation process (counts, missingness), G4 treatment.
The ICU type itself is never a feature (it defines the domain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW, PROC, P2012_WINDOW_H

STATIC = ["Age", "Gender", "Height", "Weight"]  # ICUType handled separately
# 36 time-varying variables of the challenge (MechVent is the treatment flag)
TIMESERIES = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "MAP", "Mg", "NA", "NIDiasABP", "NIMAP", "NISysABP",
    "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
    "TroponinI", "TroponinT", "Urine", "WBC", "pH",
]
ICU_TYPES = {1: "coronary", 2: "cardiac_surgery", 3: "medical", 4: "surgical"}


def _minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _parse_record(path) -> dict:
    """One patient file -> flat feature dict."""
    rows = []
    with open(path) as fh:
        next(fh)  # header: Time,Parameter,Value
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t, p, v = line.split(",")
            try:
                val = float(v)
            except ValueError:
                continue
            rows.append((_minutes(t), p, val))

    rec: dict = {}
    static = {p: v for (t, p, v) in rows if p in STATIC + ["RecordID", "ICUType"]}
    rec["RecordID"] = int(static.get("RecordID", -1))
    rec["domain"] = ICU_TYPES.get(int(static.get("ICUType", 0)), "unknown")

    # G1 demographics (-1 is the dataset's missing code)
    for k in STATIC:
        v = static.get(k, np.nan)
        rec[f"G1_{k}"] = np.nan if v is None or v == -1 else v

    cutoff = P2012_WINDOW_H * 60
    by_var: dict[str, list[float]] = {}
    for (t, p, v) in rows:
        if p in TIMESERIES and t <= cutoff:
            by_var.setdefault(p, []).append(v)

    for var in TIMESERIES:
        vals = by_var.get(var, [])
        if vals:
            a = np.asarray(vals, dtype=float)
            rec[f"G2_{var}_first"] = a[0]
            rec[f"G2_{var}_last"] = a[-1]
            rec[f"G2_{var}_min"] = a.min()
            rec[f"G2_{var}_max"] = a.max()
            rec[f"G2_{var}_mean"] = a.mean()
        else:
            for s in ("first", "last", "min", "max", "mean"):
                rec[f"G2_{var}_{s}"] = np.nan
        # G3: the observation process itself
        rec[f"G3_{var}_count"] = len(vals)
        rec[f"G3_{var}_missing"] = int(len(vals) == 0)

    rec["G3_total_obs"] = sum(len(v) for v in by_var.values())
    rec["G3_vars_observed"] = len(by_var)

    # G4 treatment
    mv = [v for (t, p, v) in rows if p == "MechVent" and t <= cutoff]
    rec["G4_MechVent"] = 1.0 if mv else 0.0
    rec["G4_MechVent_count"] = len(mv)
    return rec


def build(save: bool = True) -> pd.DataFrame:
    frames = []
    for s in ("a", "b", "c"):
        set_dir = RAW / "p2012" / f"set-{s}"
        outcomes = pd.read_csv(RAW / "p2012" / f"Outcomes-{s}.txt")
        recs = [_parse_record(p) for p in sorted(set_dir.glob("*.txt"))]
        df = pd.DataFrame(recs)
        df = df.merge(
            outcomes[["RecordID", "In-hospital_death"]], on="RecordID", how="inner"
        )
        df["set"] = s
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"In-hospital_death": "y"})
    out = out[out["domain"] != "unknown"].reset_index(drop=True)
    out["dataset"] = "p2012"
    out["stay_id"] = out["RecordID"]
    if save:
        out.to_pickle(PROC / "cohort_p2012.pkl")
    return out


if __name__ == "__main__":
    df = build()
    print(f"p2012 cohort: {len(df)} stays, {int(df.y.sum())} deaths "
          f"({df.y.mean():.1%})")
    print(df.groupby("domain").agg(n=("y", "size"), deaths=("y", "sum"),
                                   rate=("y", "mean")))
    feats = [c for c in df.columns if c.startswith(("G1_", "G2_", "G3_", "G4_"))]
    print(f"features: {len(feats)}")
