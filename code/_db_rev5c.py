# -*- coding: utf-8 -*-
"""第五轮b：30 天死亡统一按 ICU 入科起算 + 再入院限 died==0（患者聚类稳健）
输出 analysis/db_audit/rev5c_secondary.json
"""
import json, numpy as np, pandas as pd, psycopg2
import os
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
OUT = f"{ANA}/db_audit/rev5c_secondary.json"
PG = dict(**_DB_AUTH)
RES = {}

def conn(db):
    c = dict(PG); c["dbname"] = db; return psycopg2.connect(**c)

def orclust(df, g=None):
    r = smf.glm("y ~ fi10 + age + male", data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(g if g is not None else df["subject_id"])})
    b, se = r.params["fi10"], r.bse["fi10"]
    return dict(n=int(len(df)), events=int(df["y"].sum()), rate=round(float(100*df["y"].mean()), 1),
                OR=round(float(np.exp(b)), 3), lo=round(float(np.exp(b-1.96*se)), 3),
                hi=round(float(np.exp(b+1.96*se)), 3), p=float(r.pvalues["fi10"]),
                auc=round(float(roc_auc_score(df["y"], r.predict(df))), 3))

for db, glab in [("mimiciv", "MIMIC-IV"), ("nwicu", "NWICU")]:
    A, P = ("mimiciv_hosp", "mimiciv_hosp") if db == "mimiciv" else ("hosp", "hosp")
    with conn(db) as c:
        o = pd.read_sql_query(f"""
        WITH s AS (
          SELECT i.stay_id::text AS stay_id, i.hadm_id, i.subject_id, i.intime, a.dischtime, p.dod
          FROM {"mimiciv_icu.icustays" if db=="mimiciv" else "icu.icustays"} i
          JOIN {A}.admissions a ON a.hadm_id = i.hadm_id
          JOIN {P}.patients p   ON p.subject_id = i.subject_id
        ), nxt AS (
          SELECT a.subject_id, a.hadm_id,
                 LEAD(a.admittime) OVER (PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id) AS next_adm,
                 a.dischtime
          FROM {A}.admissions a
        )
        SELECT s.stay_id,
          CASE WHEN s.dod IS NOT NULL AND s.dod >= s.intime::date
                    AND s.dod <= (s.intime + interval '30 day')::date THEN 1 ELSE 0 END AS mort30_icu,
          CASE WHEN n.next_adm IS NOT NULL AND n.next_adm > n.dischtime
                    AND n.next_adm <= n.dischtime + interval '30 day' THEN 1 ELSE 0 END AS readmit30
        FROM s LEFT JOIN nxt n ON n.hadm_id = s.hadm_id""", c)
    a = pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
    a["stay_id"] = a["stay_id"].astype(str); a["fi10"] = a["fi_lab"]*10
    d = a.merge(o, on="stay_id", how="left")
    d[["mort30_icu", "readmit30"]] = d[["mort30_icu", "readmit30"]].fillna(0).astype(int)
    R = {}
    t = d.copy(); t["y"] = t["died"].astype(float); R["inhosp"] = orclust(t)
    t = d.copy(); t["y"] = t["mort30_icu"].astype(float); R["mort30_icu"] = orclust(t)
    s = d[d["died"] == 0].copy(); s["y"] = s["readmit30"].astype(float)
    R["readmit30"] = orclust(s)
    R["n_survivors"] = int(len(s))
    RES[glab] = R
    print(f"[{glab}]", json.dumps(R, ensure_ascii=False), flush=True)

json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=2)
print("WROTE", OUT)
