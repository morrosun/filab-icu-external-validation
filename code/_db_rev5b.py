# -*- coding: utf-8 -*-
"""Round-5b：NWICU 死亡定义收紧 + 三结局重建（患者聚类稳健 SE）
- 院内死亡：admittime <= dod <= dischtime（严格），以及 <= dischtime+1d（含宽限）
- 30 天死亡：admittime <= dod <= admittime + 30d
- 30 天再入院：存活出院者中，下一次入院落在 (dischtime, dischtime+30d]
输出 analysis/db_audit/rev5_secondary.json
"""
import json, numpy as np, pandas as pd
import os
import psycopg2
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



ANA = os.path.join(ROOT, 'analysis')
OUT = f"{ANA}/db_audit/rev5_secondary.json"
PG = dict(**_DB_AUTH)
RES = {}

def conn(db):
    c = dict(PG); c["dbname"] = db; return psycopg2.connect(**c)

def orclust(df, form="y ~ fi10 + age + male", g=None):
    r = smf.glm(form, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(g if g is not None else df["subject_id"])})
    b, se = r.params["fi10"], r.bse["fi10"]
    return dict(n=int(len(df)), events=int(df["y"].sum()), rate=round(float(100*df["y"].mean()), 1),
                OR=round(float(np.exp(b)), 3), lo=round(float(np.exp(b-1.96*se)), 3),
                hi=round(float(np.exp(b+1.96*se)), 3), p=float(r.pvalues["fi10"]),
                auc=round(float(roc_auc_score(df["y"], r.predict(df))), 3))

def outcomes(db):
    """返回 hadm_id 级：strict / grace / mort30 / readmit30"""
    if db == "mimiciv":
        A, P = "mimiciv_hosp", "mimiciv_hosp"
    else:
        A, P = "hosp", "hosp"
    sql = f"""
    WITH base AS (
      SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime, p.dod,
             LEAD(a.admittime) OVER (PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id) AS next_adm,
             ROW_NUMBER() OVER (PARTITION BY a.subject_id, a.admittime, a.dischtime ORDER BY a.admittime) AS dup
      FROM {A}.admissions a JOIN {P}.patients p ON a.subject_id = p.subject_id
    )
    SELECT hadm_id,
      CASE WHEN dod IS NOT NULL AND dod >= admittime AND dod <= dischtime THEN 1 ELSE 0 END AS strict,
      CASE WHEN dod IS NOT NULL AND dod >= admittime AND dod > dischtime
                AND dod <= dischtime + interval '1 day' THEN 1 ELSE 0 END AS grace_only,
      CASE WHEN dod IS NOT NULL AND dod >= admittime
                AND dorng(dod, admittime + interval '30 day') THEN 1 ELSE 0 END AS mort30x,
      CASE WHEN next_adm IS NOT NULL AND next_adm > dischtime
                AND next_adm <= dischtime + interval '30 day' THEN 1 ELSE 0 END AS readmit30
    FROM base
    """
    sql = sql.replace("dorng(dod, admittime + interval '30 day')", "dod <= admittime + interval '30 day'")
    with conn(db) as c:
        return pd.read_sql_query(sql, c)

for db, glab in [("mimiciv", "MIMIC-IV"), ("nwicu", "NWICU")]:
    a = pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
    a["stay_id"] = a["stay_id"].astype(str)
    o = outcomes(db)
    o["hadm_id"] = pd.to_numeric(o["hadm_id"], errors="coerce")
    a["hadm_id"] = pd.to_numeric(a["hadm_id"], errors="coerce")
    d = a.merge(o.drop_duplicates("hadm_id"), on="hadm_id", how="left")
    d[["strict", "grace_only", "mort30x", "readmit30"]] = d[["strict", "grace_only", "mort30x", "readmit30"]].fillna(0).astype(int)
    d["fi10"] = d["fi_lab"] * 10
    base = d.dropna(subset=["fi_lab", "age", "male"]).copy()
    R = {"n": int(len(base)),
         "strict": int(base["strict"].sum()),
         "grace_only": int(base["grace_only"].sum()),
         "strict_plus_grace": int((base["strict"] + base["grace_only"]).sum()),
         "mort30": int(base["mort30x"].sum()),
         "readmit30_survivors": int(base.loc[base["strict"] + base["grace_only"] == 0, "readmit30"].sum())}
    for col in ["strict", "grace_only", "mort30x", "readmit30"]:
        if col == "grace_only": continue
        if col == "readmit30":
            sub = base[base["strict"] + base["grace_only"] == 0].copy()
            sub["y"] = sub["readmit30"].astype(float); R["OR_readmit30"] = orclust(sub)
        else:
            t = base.copy(); t["y"] = t[col].astype(float); R[f"OR_{col}"] = orclust(t)
    # 当前（含宽限）院内死亡
    t = base.copy(); t["y"] = (t["strict"] + t["grace_only"]).clip(0, 1).astype(float)
    R["OR_inhosp_current"] = orclust(t)
    RES[glab] = R
    print(f"[{glab}]", json.dumps(R, ensure_ascii=False)[:600], flush=True)

# eICU 院内死亡（客户唯一 ID 聚类）
e = pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
e["stay_id"] = e["stay_id"].astype(str)
e = e.drop(columns=[c for c in ["subject_id"] if c in e.columns])
with conn("eicu") as c:
    pid = pd.read_sql_query("SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", c)
e = e.merge(pid, on="stay_id", how="left")
e["fi10"] = e["fi_lab"] * 10
t = e.dropna(subset=["fi_lab", "age", "male"]).copy(); t["y"] = t["died"].astype(float)
RES["eICU"] = {"OR_inhosp": orclust(t, g=t["uniquepid"]), "n": int(len(t))}
print("[eICU]", RES["eICU"], flush=True)

json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=2)
print("WROTE", OUT)
