# -*- coding: utf-8 -*-
"""Stage ③ (R1-1): 24 h landmark, run in BOTH roles.

Role A (added sensitivity): freeze the ORIGINAL main-analysis coefficients and
apply them to landmark cohorts with the outcome restricted to death after 24 h.
Role B (new primary analysis): rebuild M1-M4 in the MIMIC-IV landmark cohort and
re-derive the frozen parsimonious coefficients for external validation.

Main landmark rule: alive and still observed in hospital >=24 h after ICU
admission; deaths recorded <24 h after ICU admission are excluded; outcome is
subsequent in-hospital death. ICU-observation variant is also reported.
Outputs: analysis/stage3/*
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf
from psycopg2.extras import execute_values
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (ANA, OUT, save_json, fit_or, performance,
                        paired_delta_auc, paired_delta_brier, cluster_column,
                        frozen_predict)

STAGE = os.path.join(OUT, "stage3")
os.makedirs(STAGE, exist_ok=True)
PID_MAP = pd.read_csv(os.path.join(ANA, "db_audit", "fixed_fi14_eicu.csv"),
                      usecols=["stay_id", "subject_id"])
PID_MAP["stay_id"] = PID_MAP["stay_id"].astype(str)
PG = dict(host="127.0.0.1", port=5432, user="postgres", password=os.environ["MIMIC_PG_PASSWORD"])
RES = {"stage": "3",
       "landmark_rule": "hospital observation >=24 h after ICU admission, no death <24 h; outcome = in-hospital death after landmark",
       "icu_variant_rule": "ICU observation >=24 h after ICU admission, no death <24 h"}

def timing(db, stay_ids):
    with psycopg2.connect(dbname=db, **PG) as conn:
        cur = conn.cursor()
        cur.execute("SET statement_timeout='1200s'")
        cur.execute("CREATE TEMP TABLE _rev_ids (stay_id bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS")
        execute_values(cur, "INSERT INTO _rev_ids VALUES %s",
                       [(int(x),) for x in stay_ids], page_size=5000)
        conn.commit()
        if db == "eicu":
            sql = """SELECT p.patientunitstayid::bigint AS stay_id,
                            p.hospitaladmitoffset, p.hospitaldischargeoffset,
                            p.unitdischargeoffset, p.hospitaldischargestatus
                     FROM _rev_ids x JOIN eicu_crd.patient p ON p.patientunitstayid=x.stay_id"""
        else:
            sch = "mimiciv_icu" if db == "mimiciv" else "icu"
            hsp = "mimiciv_hosp" if db == "mimiciv" else "hosp"
            sql = f"""SELECT i.stay_id, i.intime, i.outtime, a.dischtime, a.deathtime
                      FROM _rev_ids x JOIN {sch}.icustays i ON i.stay_id=x.stay_id
                      JOIN {hsp}.admissions a ON i.hadm_id=a.hadm_id"""
        return pd.read_sql_query(sql, conn)

def landmark_flags(db, d):
    if db == "eicu":
        hospital_obs = (d["hospitaldischargeoffset"] - d["hospitaladmitoffset"]) >= 1440
        icu_obs = d["unitdischargeoffset"] >= 1440
        early = (d["died"] == 1) & ((d["hospitaldischargeoffset"] - d["hospitaladmitoffset"]) < 1440)
    else:
        for c in ["intime", "outtime", "dischtime", "deathtime"]:
            d[c] = pd.to_datetime(d[c], errors="coerce")
        hospital_obs = (d["dischtime"] - d["intime"]).dt.total_seconds() >= 86400
        icu_obs = (d["outtime"] - d["intime"]).dt.total_seconds() >= 86400
        early = (d["died"] == 1) & ((d["deathtime"] - d["intime"]).dt.total_seconds() < 86400)
    return hospital_obs.fillna(False), icu_obs.fillna(False), early.fillna(False)

# ------------------------------------------------------------ build cohorts
cohorts = {}
RES["cohorts"] = {}
rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    a = pd.read_csv(os.path.join(ANA, f"analysis_{db}.csv"))
    a["stay_id"] = a["stay_id"].astype(str)
    a = a.dropna(subset=["fi_lab", "died"]).copy()
    t = timing(db, a["stay_id"].unique())
    t["stay_id"] = t["stay_id"].astype(str)
    d = a.merge(t, on="stay_id", how="left")
    if db == "nwicu":
        # same cleaned in-hospital death label as the primary manuscript
        lab = pd.read_csv(os.path.join(ANA, "db_audit", "nwicu_final_labels.csv"),
                          usecols=["stay_id", "died_clean"])
        lab["stay_id"] = lab["stay_id"].astype(str)
        d = d.merge(lab, on="stay_id", how="left")
        # drop the 3 pre-admission death-date records (died_clean NaN), as in stage 4,
        # so that stage 3 and the 10,816 outcome denominator are the same cohort
        n0 = len(d)
        d = d.dropna(subset=["died_clean"]).copy()
        print(f"[nwicu] dropped {n0 - len(d)} pre-admission death-date records", flush=True)
        d["died"] = d["died_clean"].astype(int)
    hosp_ok, icu_ok, early = landmark_flags(db, d)
    if db == "eicu":
        d = d.merge(PID_MAP, on="stay_id", how="left", suffixes=("", "_map"))
        if "subject_id_map" in d.columns:
            d["subject_id"] = d["subject_id"].fillna(d.pop("subject_id_map"))
    d["_pid"] = cluster_column(db, d).values
    d["fi10"] = d["fi_lab"] * 10
    d["landmark_hosp"] = hosp_ok & ~early
    d["landmark_icu"] = icu_ok & ~early
    lh = d[d["landmark_hosp"]].copy()
    li = d[d["landmark_icu"]].copy()
    cohorts[db] = dict(hosp=lh, icu=li, full=d)
    RES["cohorts"][db] = dict(
        fi_assessable=int(len(d)), early_deaths=int(early.sum()),
        landmark_hosp_n=int(len(lh)), landmark_hosp_deaths=int(lh["died"].sum()),
        landmark_icu_n=int(len(li)), landmark_icu_deaths=int(li["died"].sum()),
        missing_timing=int(t["stay_id"].isna().sum()) if "stay_id" in t else 0)
    rows.append(dict(database=db, fi_assessable=len(d), early_deaths=int(early.sum()),
                     landmark_hosp_n=len(lh), landmark_hosp_deaths=int(lh["died"].sum()),
                     landmark_icu_n=len(li), landmark_icu_deaths=int(li["died"].sum())))
    print(f"[{db}] landmark hospital n={len(lh)} deaths={int(lh['died'].sum())}; "
          f"ICU n={len(li)} deaths={int(li['died'].sum())}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage3_cohort_counts.csv"), index=False)
for db in ["mimiciv", "nwicu", "eicu"]:
    cohorts[db]["hosp"].to_csv(os.path.join(STAGE, f"landmark_hosp_{db}.csv"), index=False)

# supplementary severity covariates for MIMIC
rev = pd.read_csv(os.path.join(ANA, "rev", "rev_mimiciv.csv"))
rev["stay_id"] = rev["stay_id"].astype(str)
sev = rev[["stay_id", "cci_excl_rheum", "sofa", "oasis", "vent", "vaso", "rrt"]]
cohorts["mimiciv"]["hosp"] = cohorts["mimiciv"]["hosp"].merge(sev, on="stay_id", how="left")

# ------------------------------------------------------------ original coefficients
orig = smf.logit("died ~ fi10 + age + male", data=pd.read_csv(
    os.path.join(ANA, "analysis_mimiciv.csv")).dropna(subset=["fi_lab", "age", "male"]).assign(
    fi10=lambda x: x["fi_lab"] * 10)).fit(disp=0)
coef_orig = {k: float(v) for k, v in orig.params.items()}
RES["original_coefficients"] = {k: round(v, 6) for k, v in coef_orig.items()}

# ------------------------------------------------------------ Role A: sensitivity
def evaluate(coef, d, label, seed):
    p = frozen_predict(coef, d, fi_col="fi10", fi10=True)
    perf = performance(d["died"], p, d["_pid"], B=1000, seed=seed)
    local, _ = fit_or(d, "died ~ fi10 + age + male", "fi10", cluster="_pid")
    return dict(performance=perf, local_OR=local, label=label)

RES["role_A_sensitivity"] = {}
rows = []
for i, db in enumerate(["mimiciv", "nwicu", "eicu"]):
    out = evaluate(coef_orig, cohorts[db]["hosp"], db, 43100 + i)
    RES["role_A_sensitivity"][db] = out
    perf = out["performance"]
    rows.append(dict(database=db, n=perf["n"], deaths=perf["deaths"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"],
                     brier=perf["brier"], oe=perf["oe"], oe_lo=perf["oe_lo"], oe_hi=perf["oe_hi"],
                     citl=perf["citl"], citl_lo=perf["citl_lo"], citl_hi=perf["citl_hi"],
                     slope=perf["slope"], slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"],
                     local_OR=out["local_OR"]["OR"]))
    print(f"[A {db}] AUC {perf['auc']} slope {perf['slope']} O:E {perf['oe']} local OR {out['local_OR']['OR']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage3_roleA_sensitivity.csv"), index=False)

# ------------------------------------------------------------ Role B: primary
def complete_cases(formula, d):
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    needed = [c for c in tokens if c in d.columns]
    if "_pid" in d.columns:
        needed.append("_pid")
    return d.dropna(subset=needed).copy()

models = [
    ("M1", "died ~ fi10 + age + male"),
    ("M2", "died ~ fi10 + age + male + cci_excl_rheum"),
    ("M3", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
    ("M4", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"),
]
RES["role_B_primary"] = {"hierarchical_models": {}}
rows = []
for label, formula in models:
    du = complete_cases(formula, cohorts["mimiciv"]["hosp"])
    r, mm = fit_or(du, formula, "fi10", cluster="_pid")
    p = mm.predict(du)
    perf = performance(du["died"], p, du["_pid"], B=1000, seed=43200 + int(label[1:]))
    RES["role_B_primary"]["hierarchical_models"][label] = dict(or_ci=r, auc=perf["auc"],
                                                                auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"])
    rows.append(dict(model=label, n=r["n"], OR=r["OR"], lo=r["lo"], hi=r["hi"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"]))
    print(f"[B {label}] OR {r['OR']} ({r['lo']}-{r['hi']}) AUC {perf['auc']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage3_roleB_m1_m4.csv"), index=False)

du = complete_cases(models[-1][1], cohorts["mimiciv"]["hosp"])
base_f = "died ~ age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
mb = smf.logit(base_f, data=du).fit(disp=0)
mf = smf.logit(base_f + " + fi10", data=du).fit(disp=0)
pb, pf = mb.predict(du).values, mf.predict(du).values
RES["role_B_primary"]["incremental"] = dict(
    n=int(len(du)),
    auc_base=round(float(roc_auc_score(du["died"], pb)), 4),
    auc_full=round(float(roc_auc_score(du["died"], pf)), 4),
    delta_auc=paired_delta_auc(du["died"], pb, pf, du["_pid"], B=1000, seed=43300),
    delta_brier=paired_delta_brier(du["died"], pb, pf, du["_pid"], B=1000, seed=43301),
)
print("[B incremental]", RES["role_B_primary"]["incremental"], flush=True)

mm = smf.logit("died ~ fi10 + age + male", data=cohorts["mimiciv"]["hosp"]).fit(disp=0)
coef_new = {k: float(v) for k, v in mm.params.items()}
RES["role_B_primary"]["frozen_coefficients"] = {k: round(v, 6) for k, v in coef_new.items()}
RES["role_B_primary"]["frozen_validation"] = {}
rows = []
for i, db in enumerate(["mimiciv", "nwicu", "eicu"]):
    out = evaluate(coef_new, cohorts[db]["hosp"], db, 43400 + i)
    perf = out["performance"]
    RES["role_B_primary"]["frozen_validation"][db] = out
    rows.append(dict(database=db, n=perf["n"], deaths=perf["deaths"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"],
                     brier=perf["brier"], oe=perf["oe"], oe_lo=perf["oe_lo"], oe_hi=perf["oe_hi"],
                     citl=perf["citl"], citl_lo=perf["citl_lo"], citl_hi=perf["citl_hi"],
                     slope=perf["slope"], slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"]))
    print(f"[B frozen {db}] AUC {perf['auc']} slope {perf['slope']} O:E {perf['oe']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage3_roleB_frozen.csv"), index=False)

# ------------------------------------------------------------ ICU-observation variant
RES["icu_variant"] = {}
rows = []
for i, db in enumerate(["mimiciv", "nwicu", "eicu"]):
    out = evaluate(coef_orig, cohorts[db]["icu"], db, 43500 + i)
    perf = out["performance"]
    RES["icu_variant"][db] = out
    rows.append(dict(database=db, n=perf["n"], deaths=perf["deaths"], auc=perf["auc"],
                     slope=perf["slope"], oe=perf["oe"], citl=perf["citl"]))
    print(f"[ICU variant {db}] n={perf['n']} AUC {perf['auc']} slope {perf['slope']} O:E {perf['oe']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage3_icu_variant.csv"), index=False)

save_json(RES, os.path.join(STAGE, "stage3_results.json"))
