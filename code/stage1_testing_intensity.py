# -*- coding: utf-8 -*-
"""Stage ① (R1-3 / R2-2 / R1-6):
  (A) n_assessed as an explicit covariate in every database
  (B) 14/14 fully observed harmonized panel
  (C) eICU APACHE IV / acute-physiology sensitivity
Outputs: analysis/stage1/stage1_results.json + companion CSVs.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

from rev_common import (ANA, OUT, save_json, read_analysis, read_fixed14,
                        fit_or, performance, paired_delta_auc, cluster_column,
                        frozen_predict)

STAGE = os.path.join(OUT, "stage1")
os.makedirs(STAGE, exist_ok=True)
RES = {"stage": "1", "analyses": ["n_assessed covariate", "14/14 panel", "eICU APACHE IV sensitivity"]}

# ---------------------------------------------------------------- (A)
RES["n_assessed"] = {}
rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    d = read_analysis(db)
    d["_pid"] = cluster_column(db, d).values
    base, _ = fit_or(d, "died ~ fi10 + age + male", "fi10", cluster="_pid")
    aug, m = fit_or(d, "died ~ fi10 + age + male + n_assessed", "fi10", cluster="_pid")
    b5, se5 = float(m.params["n_assessed"]) * 5, float(m.bse["n_assessed"]) * 5
    RES["n_assessed"][db] = dict(
        base_model=base, adjusted_model=aug,
        n_assessed_OR_per_item=round(float(np.exp(m.params["n_assessed"])), 3),
        n_assessed_lo=round(float(np.exp(m.params["n_assessed"] - 1.96 * m.bse["n_assessed"])), 3),
        n_assessed_hi=round(float(np.exp(m.params["n_assessed"] + 1.96 * m.bse["n_assessed"])), 3),
        n_assessed_OR_per_5_items=round(float(np.exp(b5)), 3),
        n_assessed_per5_lo=round(float(np.exp(b5 - 1.96 * se5)), 3),
        n_assessed_per5_hi=round(float(np.exp(b5 + 1.96 * se5)), 3),
        n_assessed_p=(0.0 if m.pvalues["n_assessed"] < 1e-300 else round(float(m.pvalues["n_assessed"]), 8)),
        fi10_OR_change_pct=round(100 * (aug["OR"] / base["OR"] - 1), 2),
    )
    rows.append(dict(database=db, n=base["n"], deaths=int(d["died"].sum()),
                     fi_OR_base=base["OR"], fi_OR_adjusted=aug["OR"],
                     n_assessed_OR_per5=RES["n_assessed"][db]["n_assessed_OR_per_5_items"]))
    print(f"[{db}] n_assessed model: FI OR {base['OR']} -> {aug['OR']}; "
          f"per-5-items OR {RES['n_assessed'][db]['n_assessed_OR_per_5_items']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage1_n_assessed.csv"), index=False)

# ---------------------------------------------------------------- (B)
RES["panel_14of14"] = {}
panels = {}
coef14 = None
for db in ["mimiciv", "nwicu", "eicu"]:
    d = read_fixed14(db)
    d = d[(d["n14"] == 14) & d["died"].notna()].copy()
    d["_pid"] = cluster_column(db, d).values
    panels[db] = d
    local, _ = fit_or(d, "died ~ fi10 + age + male", "fi10", cluster="_pid")
    if db == "mimiciv":
        _, mm = fit_or(d, "died ~ fi10 + age + male", "fi10")
        coef14 = {k: float(v) for k, v in mm.params.items()}
    RES["panel_14of14"][db] = dict(n=int(len(d)), deaths=int(d["died"].sum()),
                                    local_OR=local)
RES["panel_14of14"]["mimic_coefficients"] = {k: round(v, 6) for k, v in coef14.items()}
print("[14/14] MIMIC-derived coefficients:", RES["panel_14of14"]["mimic_coefficients"], flush=True)

rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    d = panels[db]
    p = frozen_predict(coef14, d, fi_col="fi14")
    perf = performance(d["died"], p, d["_pid"], B=1000, seed=41001)
    RES["panel_14of14"][db]["frozen"] = perf
    rows.append(dict(database=db, n=perf["n"], deaths=perf["deaths"], auc=perf["auc"],
                     auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"], brier=perf["brier"],
                     oe=perf["oe"], citl=perf["citl"], citl_lo=perf["citl_lo"],
                     citl_hi=perf["citl_hi"], slope=perf["slope"],
                     slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"]))
    print(f"[14/14 frozen {db}] AUC {perf['auc']} slope {perf['slope']} O:E {perf['oe']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage1_panel14of14.csv"), index=False)

# ---------------------------------------------------------------- (C)
rev = pd.read_csv(os.path.join(ANA, "rev", "rev_eicu.csv"))
rev["stay_id"] = rev["stay_id"].astype(str)
rev["fi10"] = rev["fi_lab"] * 10
pid_map = pd.read_csv(os.path.join(ANA, "db_audit", "fixed_fi14_eicu.csv"),
                      usecols=["stay_id", "subject_id"])
pid_map["stay_id"] = pid_map["stay_id"].astype(str)
rev = rev.merge(pid_map, on="stay_id", how="left", suffixes=("", "_map"))
if "subject_id_map" in rev.columns:
    rev["subject_id"] = rev["subject_id"].fillna(rev.pop("subject_id_map"))
rev["_pid"] = cluster_column("eicu", rev).values

fit_sets = {
    "age_sex_APACHEIV": ("died ~ fi10 + age + male + apache", "died ~ age + male + apache"),
    "sex_APACHEIV_no_age": ("died ~ fi10 + male + apache", "died ~ male + apache"),
    "age_sex_acute_physiology": ("died ~ fi10 + age + male + apsiii", "died ~ age + male + apsiii"),
}
RES["eicu_apache"] = {}
apache_rows = []
for label, (full_f, base_f) in fit_sets.items():
    d = rev.dropna(subset=["died", "fi10", "age", "male", "apache", "apsiii"]).copy()
    full, mf = fit_or(d, full_f, "fi10", cluster="_pid")
    mb = smf.logit(base_f, data=d).fit(disp=0)
    p_full, p_base = mf.predict(d).values, mb.predict(d).values
    da = paired_delta_auc(d["died"], p_base, p_full, d["_pid"], B=1000,
                          seed=41100 + len(RES["eicu_apache"]))
    entry = dict(n=int(len(d)), deaths=int(d["died"].sum()), fi10_OR=full,
                 auc_base=round(float(roc_auc_score(d["died"], p_base)), 4),
                 auc_full=round(float(roc_auc_score(d["died"], p_full)), 4),
                 delta_auc=da, model_full=full_f, model_base=base_f)
    RES["eicu_apache"][label] = entry
    apache_rows.append(dict(specification=label, n=entry["n"], deaths=entry["deaths"],
                            fi_OR=full["OR"], fi_lo=full["lo"], fi_hi=full["hi"],
                            auc_base=entry["auc_base"], auc_full=entry["auc_full"],
                            delta_auc=da["delta_auc"], delta_lo=da["delta_auc_lo"],
                            delta_hi=da["delta_auc_hi"]))
    print(f"[eICU {label}] n={entry['n']} FI OR={full['OR']} dAUC={da}", flush=True)
pd.DataFrame(apache_rows).to_csv(os.path.join(STAGE, "stage1_eicu_apache.csv"), index=False)

save_json(RES, os.path.join(STAGE, "stage1_results.json"))
