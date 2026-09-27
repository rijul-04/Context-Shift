"""eICU-CRD Demo cohort: in-hospital mortality, prediction at 24h.

Domains are US regions (the demo spreads 2,520 stays over 186 hospitals, so
individual hospitals are too small to serve as deployment sites).

Feature groups: G1 demographics/comorbidity, G2 first-day physiology
(APACHE APS variables), G3 observation process (lab/vital counts,
missingness), G4 treatment.

Excluded to avoid leakage: APACHE scores and predicted mortality, discharge
fields, length of stay, diedinhospital, and all hospital-level attributes
(region defines the domain; teaching status and bed count proxy it).
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from .config import RAW, PROC, EICU_WINDOW_MIN

APS_VALUES = [
    "eyes", "motor", "verbal", "meds", "urine", "wbc", "temperature",
    "respiratoryrate", "sodium", "heartrate", "meanbp", "ph", "hematocrit",
    "creatinine", "albumin", "pao2", "pco2", "bun", "glucose", "bilirubin",
    "fio2",
]
APS_TREATMENT = ["intubated", "vent", "dialysis"]
COMORBID = [
    "aids", "hepaticfailure", "lymphoma", "metastaticcancer", "leukemia",
    "immunosuppression", "cirrhosis", "diabetes", "electivesurgery",
    "readmit", "activetx", "thrombolytics",
]


def _read(name: str, usecols=None) -> pd.DataFrame:
    with gzip.open(RAW / "eicu_demo" / f"{name}.csv.gz", "rt") as fh:
        return pd.read_csv(fh, usecols=usecols, low_memory=False)


def _age(a) -> float:
    if pd.isna(a):
        return np.nan
    a = str(a).strip()
    if a == "> 89":
        return 90.0
    try:
        return float(a)
    except ValueError:
        return np.nan


def build(save: bool = True) -> pd.DataFrame:
    pat = _read("patient")
    hosp = _read("hospital")
    aps = _read("apacheApsVar")
    pred = _read("apachePredVar")

    pat["age_num"] = pat["age"].map(_age)
    pat = pat[
        (pat["age_num"] >= 18)
        & (pat["unitvisitnumber"] == 1)
        & (pat["unitdischargeoffset"] >= EICU_WINDOW_MIN)
        & (pat["hospitaldischargestatus"].isin(["Alive", "Expired"]))
    ].copy()

    pat = pat.merge(hosp[["hospitalid", "region"]], on="hospitalid", how="left")
    pat = pat[pat["region"].notna()].copy()

    df = pd.DataFrame({
        "stay_id": pat["patientunitstayid"].values,
        "domain": pat["region"].values,
        "hospitalid": pat["hospitalid"].values,
        "y": (pat["hospitaldischargestatus"] == "Expired").astype(int).values,
        "G1_age": pat["age_num"].values,
        "G1_female": (pat["gender"] == "Female").astype(int).values,
        "G1_weight": pd.to_numeric(pat["admissionweight"], errors="coerce").values,
        "G1_height": pd.to_numeric(pat["admissionheight"], errors="coerce").values,
    })
    # admission source and unit type as one-hot (G1)
    for col, prefix in (("unitadmitsource", "G1_src"), ("unittype", "G1_unit")):
        d = pd.get_dummies(pat[col].fillna("unknown"), prefix=prefix).astype(int)
        d.index = df.index
        df = pd.concat([df, d], axis=1)

    # G2 first-day physiology; the APACHE tables use -1 for missing
    aps = aps.set_index("patientunitstayid")
    for c in APS_VALUES:
        s = aps[c].reindex(pat["patientunitstayid"]).astype(float).values
        s = np.where(s == -1, np.nan, s)
        df[f"G2_{c}"] = s
        df[f"G3_{c}_missing"] = np.isnan(s).astype(int)
    for c in APS_TREATMENT:
        s = aps[c].reindex(pat["patientunitstayid"]).astype(float).values
        df[f"G4_{c}"] = np.where(s == -1, np.nan, s)

    # G1 comorbidity + G4 ventilation, from apachePredVar (no score fields)
    pred = pred.set_index("patientunitstayid")
    for c in COMORBID:
        if c in pred.columns:
            s = pred[c].reindex(pat["patientunitstayid"]).astype(float).values
            df[f"G1_{c}"] = np.where(s == -1, np.nan, s)
    for c in ("ventday1", "oobventday1", "oobintubday1"):
        if c in pred.columns:
            s = pred[c].reindex(pat["patientunitstayid"]).astype(float).values
            df[f"G4_{c}"] = np.where(s == -1, np.nan, s)

    # G3 observation process: how much was measured in the first 24h
    stays = set(pat["patientunitstayid"])
    lab = _read("lab", usecols=["patientunitstayid", "labresultoffset", "labname"])
    lab = lab[lab["patientunitstayid"].isin(stays)
              & lab["labresultoffset"].between(0, EICU_WINDOW_MIN)]
    lab_n = lab.groupby("patientunitstayid").size()
    lab_d = lab.groupby("patientunitstayid")["labname"].nunique()

    vit = _read("vitalPeriodic", usecols=["patientunitstayid", "observationoffset"])
    vit = vit[vit["patientunitstayid"].isin(stays)
              & vit["observationoffset"].between(0, EICU_WINDOW_MIN)]
    vit_n = vit.groupby("patientunitstayid").size()

    idx = pat["patientunitstayid"]
    df["G3_lab_count"] = lab_n.reindex(idx).fillna(0).values
    df["G3_lab_distinct"] = lab_d.reindex(idx).fillna(0).values
    df["G3_vital_count"] = vit_n.reindex(idx).fillna(0).values
    df["G3_vital_per_hour"] = df["G3_vital_count"] / 24.0

    df["dataset"] = "eicu_demo"
    df = df.reset_index(drop=True)
    if save:
        df.to_pickle(PROC / "cohort_eicu.pkl")
    return df


if __name__ == "__main__":
    d = build()
    print(f"eicu cohort: {len(d)} stays, {int(d.y.sum())} deaths ({d.y.mean():.1%})")
    print(d.groupby("domain").agg(n=("y", "size"), deaths=("y", "sum"),
                                  rate=("y", "mean"),
                                  hospitals=("hospitalid", "nunique")))
    feats = [c for c in d.columns if c.startswith(("G1_", "G2_", "G3_", "G4_"))]
    print(f"features: {len(feats)}")
