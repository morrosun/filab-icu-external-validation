# -*- coding: utf-8 -*-
"""Stage ② (R1-2): recompute MIMIC-IV FI-Lab with a uniform 0-24 h window.

The 25 FI items sourced from mimiciv_derived.first_day_lab are rebuilt from the
four underlying derived tables with charttime >= intime (the original table uses
intime - 6 h). The remaining 5 FI items already used a 0-24 h labevents window
and are re-extracted unchanged. Then M1-M4, incremental value and frozen
external validation are repeated.
Outputs: analysis/stage2/*
"""
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (ANA, OUT, save_json, fit_or, performance,
                        paired_delta_auc, paired_delta_brier, cluster_column,
                        frozen_predict, apply_nwicu_label_exclusion)
sys.path.insert(0, os.path.dirname(OUT))
sys.path.insert(0, os.path.dirname(os.path.dirname(OUT)))
import build_analysis as ba

STAGE = os.path.join(OUT, "stage2")
os.makedirs(STAGE, exist_ok=True)
PID_MAP = pd.read_csv(os.path.join(ANA, "db_audit", "fixed_fi14_eicu.csv"),
                      usecols=["stay_id", "subject_id"])
PID_MAP["stay_id"] = PID_MAP["stay_id"].astype(str)
PG = dict(host="127.0.0.1", port=5432, user="postgres", password=os.environ["MIMIC_PG_PASSWORD"], dbname="mimiciv")
RES = {"stage": "2", "window": "charttime >= ICU intime AND <= intime + 24 h"}

TABLE_FOR = {}
SRC_COL = {a: spec[0] for a, spec in ba.D.items() if spec[0] is not None}
SRC_COL["platelet"] = "platelet"  # derived complete_blood_count uses singular
for a, spec in ba.D.items():
    if spec[0] is None:
        continue
    if a in ("albumin", "globulin", "total_protein", "aniongap", "bicarbonate", "bun",
             "calcium", "chloride", "creatinine", "glucose", "sodium", "potassium"):
        TABLE_FOR[a] = "chemistry"
    elif a in ("hematocrit", "hemoglobin", "platelet"):
        TABLE_FOR[a] = "complete_blood_count"
    elif a in ("pt", "ptt", "inr", "fibrinogen"):
        TABLE_FOR[a] = "coagulation"
    else:
        TABLE_FOR[a] = "enzyme"

def make_temp_ids(conn, stay_ids):
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE _rev_ids (stay_id bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS")
    execute_values(cur, "INSERT INTO _rev_ids (stay_id) VALUES %s",
                   [(int(x),) for x in stay_ids], page_size=5000)
    conn.commit()
    cur.execute("ANALYZE _rev_ids")
    conn.commit()

def derived_labs(conn):
    out = None
    for tbl in sorted(set(TABLE_FOR.values())):
        cols = sorted(set(a for a, t in TABLE_FOR.items() if t == tbl))
        sel = ", ".join(f"MIN(t.{SRC_COL[a]}) AS {a}_min, MAX(t.{SRC_COL[a]}) AS {a}_max" for a in cols)
        sql = f"""
            SELECT i.stay_id, {sel}
            FROM _rev_ids x
            JOIN mimiciv_icu.icustays i ON i.stay_id = x.stay_id
            JOIN mimiciv_derived.{tbl} t
              ON t.hadm_id = i.hadm_id
             AND t.charttime >= i.intime
             AND t.charttime <= i.intime + interval '24 hour'
            GROUP BY i.stay_id"""
        d = pd.read_sql_query(sql, conn)
        d["stay_id"] = d["stay_id"].astype(str)
        out = d if out is None else out.merge(d, on="stay_id", how="outer")
        print(f"  [labs] {tbl}: {len(d)} stays", flush=True)
    return out

def extra_labs(conn):
    sql = """
        SELECT i.stay_id,
          MAX(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) rdw_max,
          MIN(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) rdw_min,
          MAX(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) mcv_max,
          MIN(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) mcv_min,
          MAX(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) lactate_max,
          MIN(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) lactate_min,
          MAX(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) magnesium_max,
          MIN(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) magnesium_min,
          MAX(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) phosphate_max,
          MIN(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) phosphate_min
        FROM _rev_ids x
        JOIN mimiciv_icu.icustays i ON i.stay_id = x.stay_id
        JOIN mimiciv_hosp.labevents le ON le.subject_id=i.subject_id
         AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
         AND le.valuenum IS NOT NULL
         AND le.itemid IN (51277,53165,51250,51691,50813,52442,53154,50960,50970)
        GROUP BY i.stay_id"""
    d = pd.read_sql_query(sql, conn)
    d["stay_id"] = d["stay_id"].astype(str)
    return d

# ------------------------------------------------------------ rebuild FI
t0 = time.time()
base = pd.read_csv(os.path.join(ANA, "analysis_mimiciv.csv"))
base["stay_id"] = base["stay_id"].astype(str)
ids = base["stay_id"].unique()
conn = psycopg2.connect(**PG)
cur = conn.cursor()
cur.execute("SET statement_timeout='1800s'")
conn.commit()
make_temp_ids(conn, ids)
L = derived_labs(conn).merge(extra_labs(conn), on="stay_id", how="outer")
conn.close()
fi, n_assessed = ba.compute_fi(L)
L = L.copy(); L["fi_0_24"] = fi.values; L["n_0_24"] = n_assessed.values
mim = base.merge(L[["stay_id", "fi_0_24", "n_0_24"]], on="stay_id", how="left")
mim["fi10"] = mim["fi_0_24"] * 10
mim["_pid"] = cluster_column("mimiciv", mim).values
RES["cohort"] = dict(parent=int(len(mim)), assessable_0_24=int(mim["fi_0_24"].notna().sum()),
                     assessable_original=int(mim["fi_lab"].notna().sum()),
                     extraction_seconds=round(time.time()-t0, 1))
common = mim.dropna(subset=["fi_lab", "fi_0_24"])
RES["fi_comparison"] = dict(
    n_common=int(len(common)),
    median_original=round(float(common["fi_lab"].median()), 4),
    median_0_24=round(float(common["fi_0_24"].median()), 4),
    pearson_r=round(float(common[["fi_lab", "fi_0_24"]].corr().iloc[0, 1]), 4),
    spearman_r=round(float(common[["fi_lab", "fi_0_24"]].corr(method="spearman").iloc[0, 1]), 4),
    mean_abs_difference=round(float((common["fi_0_24"]-common["fi_lab"]).abs().mean()), 4),
    identical_fraction=round(float((common["fi_0_24"] == common["fi_lab"]).mean()), 4),
    n_assessed_median_original=float(common["n_assessed"].median()),
    n_assessed_median_0_24=float(common["n_0_24"].median()),
)
mim.to_csv(os.path.join(STAGE, "mimic_fi_0_24.csv"), index=False)
print("[cohort]", RES["cohort"], flush=True)
print("[FI comparison]", RES["fi_comparison"], flush=True)

# ------------------------------------------------------------ severity covariates
rev = pd.read_csv(os.path.join(ANA, "rev", "rev_mimiciv.csv"))
rev["stay_id"] = rev["stay_id"].astype(str)
sev = rev[["stay_id", "cci_excl_rheum", "sofa", "oasis", "vent", "vaso", "rrt"]]
mim = mim.merge(sev, on="stay_id", how="left", suffixes=("", "_r"))

# ------------------------------------------------------------ M1-M4
models = [
    ("M1", "died ~ fi10 + age + male"),
    ("M2", "died ~ fi10 + age + male + cci_excl_rheum"),
    ("M3", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
    ("M4", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"),
]
def complete_cases(formula, d):
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    needed = [c for c in tokens if c in d.columns]
    if "_pid" in d.columns:
        needed.append("_pid")
    return d.dropna(subset=needed).copy()

RES["models"] = {}
rows = []
for label, formula in models:
    du = complete_cases(formula, mim)
    r, mm = fit_or(du, formula, "fi10", cluster="_pid")
    p = mm.predict(du)
    perf = performance(du["died"].values, p.values, du["_pid"].values,
                       B=1000, seed=42000 + int(label[1:]))
    RES["models"][label] = dict(or_ci=r, auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"])
    rows.append(dict(model=label, n=r["n"], OR=r["OR"], lo=r["lo"], hi=r["hi"],
                     auc=perf["auc"], auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"]))
    print(f"[{label}] OR {r['OR']} ({r['lo']}-{r['hi']}), AUC {perf['auc']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage2_m1_m4.csv"), index=False)

# ------------------------------------------------------------ incremental value
d4 = mim.dropna(subset=["fi10", "age", "male", "cci_excl_rheum", "sofa", "oasis",
                         "vent", "vaso", "rrt", "_pid"]).copy()
base_f = "died ~ age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
full_f = base_f + " + fi10"
mb = smf.logit(base_f, data=d4).fit(disp=0)
mf = smf.logit(full_f, data=d4).fit(disp=0)
pb, pf = mb.predict(d4).values, mf.predict(d4).values
da = paired_delta_auc(d4["died"], pb, pf, d4["_pid"], B=1000, seed=42500)
db_ = paired_delta_brier(d4["died"], pb, pf, d4["_pid"], B=1000, seed=42501)
RES["incremental"] = dict(n=int(len(d4)), auc_base=round(float(roc_auc_score(d4["died"], pb)), 4),
                          auc_full=round(float(roc_auc_score(d4["died"], pf)), 4),
                          delta_auc=da, delta_brier=db_)
print("[incremental]", RES["incremental"], flush=True)

# ------------------------------------------------------------ frozen validation
mm = smf.logit("died ~ fi10 + age + male", data=mim.dropna(subset=["fi10", "age", "male"])).fit(disp=0)
coef = {k: float(v) for k, v in mm.params.items()}
RES["frozen_coefficients"] = {k: round(v, 6) for k, v in coef.items()}
RES["frozen_validation"] = {}
rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    d = pd.read_csv(os.path.join(ANA, f"analysis_{db}.csv"))
    d["stay_id"] = d["stay_id"].astype(str)
    d = apply_nwicu_label_exclusion(d, db)
    if db == "mimiciv":
        d = d.merge(L[["stay_id", "fi_0_24"]], on="stay_id", how="left")
        d["score_fi"] = d["fi_0_24"]
    else:
        d["score_fi"] = d["fi_lab"]
    d = d.dropna(subset=["score_fi", "died", "age", "male"]).copy()
    if db == "eicu":
        d = d.merge(PID_MAP, on="stay_id", how="left", suffixes=("", "_map"))
        if "subject_id_map" in d.columns:
            d["subject_id"] = d["subject_id"].fillna(d.pop("subject_id_map"))
    d["_pid"] = cluster_column(db, d).values
    g = d.assign(fi10=d["score_fi"]*10)
    p = frozen_predict(coef, g, fi_col="fi10", fi10=True)
    perf = performance(d["died"], p, d["_pid"], B=1000, seed=42600 + len(rows))
    RES["frozen_validation"][db] = perf
    rows.append(dict(database=db, n=perf["n"], deaths=perf["deaths"], auc=perf["auc"],
                     auc_lo=perf["auc_lo"], auc_hi=perf["auc_hi"], brier=perf["brier"],
                     oe=perf["oe"], oe_lo=perf["oe_lo"], oe_hi=perf["oe_hi"],
                     citl=perf["citl"], citl_lo=perf["citl_lo"], citl_hi=perf["citl_hi"],
                     slope=perf["slope"], slope_lo=perf["slope_lo"], slope_hi=perf["slope_hi"]))
    print(f"[frozen {db}] {perf['auc']} slope {perf['slope']} O:E {perf['oe']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(STAGE, "stage2_frozen_validation.csv"), index=False)

save_json(RES, os.path.join(STAGE, "stage2_results.json"))
