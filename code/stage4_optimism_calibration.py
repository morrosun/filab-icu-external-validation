# -*- coding: utf-8 -*-
"""Stage ④ (R1-5 / R2-3): bootstrap optimism correction and patient-level
calibration intervals.

Part 1: 300 patient-level bootstrap resamples; refit M3/M4 (with and without
FI-Lab) in each resample; estimate optimism in AUC, Brier and their differences.
Part 2: 1,000 patient-level bootstrap resamples for calibration slope and CITL
of the frozen parsimonious model in the three original cohorts and in the
24 h landmark cohorts.
Outputs: analysis/stage4/*
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (ANA, OUT, save_json, cluster_indices, resample_indices,
                        performance, frozen_predict, apply_nwicu_label_exclusion)

STAGE = os.path.join(OUT, "stage4")
os.makedirs(STAGE, exist_ok=True)
RES = {"stage": "4", "bootstrap_unit": "patient (subject_id / uniquepid)"}

PID_MAP = pd.read_csv(os.path.join(ANA, "db_audit", "fixed_fi14_eicu.csv"),
                      usecols=["stay_id", "subject_id"])
PID_MAP["stay_id"] = PID_MAP["stay_id"].astype(str)

def set_pid(db, a):
    if db == "eicu":
        a = a.copy()
        a["stay_id"] = a["stay_id"].astype(str)
        a = a.merge(PID_MAP, on="stay_id", how="left", suffixes=("", "_map"))
        if "subject_id_map" in a.columns:
            a["subject_id"] = a["subject_id"].fillna(a.pop("subject_id_map"))
    if "subject_id" in a.columns and a["subject_id"].notna().any():
        a["_pid"] = a["subject_id"].astype(str).values
    else:
        a["_pid"] = np.arange(len(a))
    return a

# ---------------------------------------------------------------- Part 1
r = pd.read_csv(os.path.join(ANA, "rev", "rev_mimiciv.csv"))
r["stay_id"] = r["stay_id"].astype(str)
r["fi10"] = r["fi_lab"] * 10
need = ["died", "fi10", "age", "male", "cci_excl_rheum", "sofa", "oasis",
        "vent", "vaso", "rrt", "subject_id"]
d = r.dropna(subset=need).copy()
d["_pid"] = d["subject_id"].astype(str).values
base_f = "died ~ age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
full_f = base_f + " + fi10"
y = d["died"].values.astype(int)
from patsy import dmatrix
Xb = np.asarray(dmatrix(base_f.split("~")[1], d), float)
Xf = np.asarray(dmatrix(full_f.split("~")[1], d), float)
names_b = dmatrix(base_f.split("~")[1], d).design_info.column_names
names_f = dmatrix(full_f.split("~")[1], d).design_info.column_names

def fit_glm(X, yy):
    m = sm.GLM(yy, X, family=sm.families.Binomial()).fit()
    return np.asarray(m.params, float)

def metrics(X, b, yy):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
    return roc_auc_score(yy, p), brier_score_loss(yy, p)

bb = fit_glm(Xb, y)
bf = fit_glm(Xf, y)
auc_b0, brier_b0 = metrics(Xb, bb, y)
auc_f0, brier_f0 = metrics(Xf, bf, y)
apparent = dict(
    n=int(len(d)), deaths=int(y.sum()),
    auc_base=round(auc_b0, 4), auc_full=round(auc_f0, 4),
    brier_base=round(brier_b0, 4), brier_full=round(brier_f0, 4),
    delta_auc=round(auc_f0 - auc_b0, 4),
    delta_brier=round(brier_f0 - brier_b0, 4),
)
print("[apparent]", apparent, flush=True)

B = 300
rng = np.random.default_rng(20261126)
_, idx = cluster_indices(d["_pid"].values)
op = []
for k, ii in enumerate(resample_indices(idx, B, rng)):
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    try:
        cb = fit_glm(Xb[ii], yy)
        cf = fit_glm(Xf[ii], yy)
        ab_t, br_bt = metrics(Xb[ii], cb, yy)
        af_t, br_ft = metrics(Xf[ii], cf, yy)
        ab_v, br_bv = metrics(Xb, cb, y)
        af_v, br_fv = metrics(Xf, cf, y)
    except Exception:
        continue
    op.append((ab_t - ab_v, af_t - af_v,
               br_bv - br_bt, br_fv - br_ft,
               (af_t - ab_t) - (af_v - ab_v),
               (br_ft - br_bt) - (br_fv - br_bv)))
op = np.asarray(op)
mean = op.mean(axis=0)
corrected = dict(
    replicates=int(len(op)),
    optimism_auc_base=round(float(mean[0]), 5),
    optimism_auc_full=round(float(mean[1]), 5),
    optimism_brier_base=round(float(mean[2]), 5),
    optimism_brier_full=round(float(mean[3]), 5),
    optimism_delta_auc=round(float(mean[4]), 5),
    optimism_delta_brier=round(float(mean[5]), 5),
    corrected_auc_base=round(auc_b0 - mean[0], 4),
    corrected_auc_full=round(auc_f0 - mean[1], 4),
    corrected_brier_base=round(brier_b0 + mean[2], 4),
    corrected_brier_full=round(brier_f0 + mean[3], 4),
    corrected_delta_auc=round((auc_f0 - auc_b0) - mean[4], 4),
    corrected_delta_brier=round((brier_f0 - brier_b0) - mean[5], 4),
)
# percentile CI of the optimism-corrected difference via replicate distribution
diff_reps = (auc_f0 - auc_b0) - op[:, 4]
corrected["corrected_delta_auc_lo"] = round(float(np.percentile(diff_reps, 2.5)), 4)
corrected["corrected_delta_auc_hi"] = round(float(np.percentile(diff_reps, 97.5)), 4)
diff_reps_b = (brier_f0 - brier_b0) - op[:, 5]
corrected["corrected_delta_brier_lo"] = round(float(np.percentile(diff_reps_b, 2.5)), 4)
corrected["corrected_delta_brier_hi"] = round(float(np.percentile(diff_reps_b, 97.5)), 4)
RES["optimism"] = dict(apparent=apparent, **corrected)
print("[optimism]", corrected, flush=True)
pd.DataFrame([RES["optimism"]]).to_csv(os.path.join(STAGE, "stage4_optimism.csv"), index=False)

# ---------------------------------------------------------------- Part 2
orig_coef = smf.logit("died ~ fi10 + age + male", data=r.dropna(subset=["fi10", "age", "male"])).fit(disp=0)
coef_orig = {k: float(v) for k, v in orig_coef.params.items()}
RES["calibration"] = {}
rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    a = pd.read_csv(os.path.join(ANA, f"analysis_{db}.csv"))
    a["stay_id"] = a["stay_id"].astype(str)
    a = a.dropna(subset=["fi_lab", "age", "male", "died"]).copy()
    a = apply_nwicu_label_exclusion(a, db)
    a["fi10"] = a["fi_lab"] * 10
    a = set_pid(db, a)
    p = frozen_predict(coef_orig, a, fi_col="fi10", fi10=True)
    perf = performance(a["died"], p, a["_pid"], B=1000, seed=44100 + len(rows))
    RES["calibration"][db] = perf
    rows.append(dict(cohort=db, n=perf["n"], deaths=perf["deaths"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"],
                     oe=perf["oe"], oe_lo=perf["oe_lo"], oe_hi=perf["oe_hi"],
                     citl=perf["citl"], citl_lo=perf["citl_lo"], citl_hi=perf["citl_hi"],
                     slope=perf["slope"], slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"],
                     bootstrap_replicates=perf.get("bootstrap_replicates")))
    print(f"[calib {db}] AUC {perf['auc']} slope {perf['slope']} ({perf['slope_lo']}-{perf['slope_hi']}) "
          f"CITL {perf['citl']} ({perf['citl_lo']}-{perf['citl_hi']})", flush=True)

# landmark role A (the retained sensitivity analysis) with 1,000 replicates
Ldir = os.path.join(OUT, "stage3")
for db in ["mimiciv", "nwicu", "eicu"]:
    a = pd.read_csv(os.path.join(Ldir, f"landmark_hosp_{db}.csv"))
    a = a.dropna(subset=["fi_lab", "age", "male", "died"]).copy()
    a["fi10"] = a["fi_lab"] * 10
    a = set_pid(db, a)
    p = frozen_predict(coef_orig, a, fi_col="fi10", fi10=True)
    perf = performance(a["died"], p, a["_pid"], B=1000, seed=44200 + len(rows))
    RES["calibration"][f"landmark_{db}"] = perf
    rows.append(dict(cohort=f"landmark_{db}", n=perf["n"], deaths=perf["deaths"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"],
                     oe=perf["oe"], oe_lo=perf["oe_lo"], oe_hi=perf["oe_hi"],
                     citl=perf["citl"], citl_lo=perf["citl_lo"], citl_hi=perf["citl_hi"],
                     slope=perf["slope"], slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"],
                     bootstrap_replicates=perf.get("bootstrap_replicates")))
    print(f"[calib landmark_{db}] AUC {perf['auc']} slope {perf['slope']} O:E {perf['oe']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage4_calibration_bootstrap.csv"), index=False)
save_json(RES, os.path.join(STAGE, "stage4_results.json"))
