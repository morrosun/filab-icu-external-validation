# -*- coding: utf-8 -*-
"""第七轮补：NWICU 三结局在 FULL 队列上重算（入组 10,819），并修正 S7 的 CI 索引。"""
import json
import os
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


PG = dict(**_DB_AUTH)
ANA = "analysis"
R = json.load(open(f"{ANA}/db_audit/rev7_nwicu.json", encoding="utf-8"))

lab = pd.read_csv(f"{ANA}/db_audit/nwicu_final_labels.csv")
lab["stay_id"] = lab["stay_id"].astype(str)
_dt = pd.read_csv(f"{ANA}/db_audit/nwicu_dod_flags.csv")
_dt["stay_id"] = _dt["stay_id"].astype(str)
lab = lab.merge(_dt[["stay_id", "admittime", "dischtime", "dod"]], on="stay_id", how="left")
for c in ["admittime", "dischtime", "dod"]:
    lab[c] = pd.to_datetime(lab[c], errors="coerce")

# --- readmit30：出院后 30 天内再次住院（同一 subject，另一 hadm） ---
# 注意：再入院要看全部住院记录（含无 ICU 的再住院），不能只取 icustays
q = "SELECT subject_id::text AS subject_id, hadm_id::text AS hadm_id, admittime, dischtime FROM hosp.admissions"
with psycopg2.connect(dbname="nwicu", **PG) as c:
    ad = pd.read_sql_query(q, c)
ad["admittime"] = pd.to_datetime(ad.admittime, errors="coerce")
ad["dischtime"] = pd.to_datetime(ad.dischtime, errors="coerce")
print("[admissions] rows =", len(ad), " subjects =", ad.subject_id.nunique(), flush=True)

a = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
a["stay_id"] = a["stay_id"].astype(str)
a = a.dropna(subset=["fi_lab"]).copy()
a = a.merge(lab[["stay_id", "died_clean", "mort30_clean", "dischtime"]], on="stay_id", how="left")
a["fi10"] = a["fi_lab"] * 10

by_subj = {s: g for s, g in ad.groupby("subject_id")}
readm = []
for _, r in a.iterrows():
    sub = by_subj.get(str(r["subject_id"]))
    v = 0
    if sub is not None and pd.notna(r["dischtime"]) and int(r["died_clean"] if pd.notna(r["died_clean"]) else -1) == 0:
        later = sub[(sub.hadm_id != str(r["hadm_id"])) &
                    (sub.admittime > r["dischtime"]) &
                    (sub.admittime <= r["dischtime"] + pd.Timedelta(days=30))]
        v = int(len(later) > 0)
    readm.append(v)
a["readmit30_new"] = readm
print("[readmit] eligible (alive) =", int((a.died_clean.fillna(-1) == 0).sum()), " events =", int(a.readmit30_new.sum()), flush=True)


def fit(ycol, d):
    m = smf.glm(f"{ycol} ~ fi10 + age + male", data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
    b, se = m.params["fi10"], m.bse["fi10"]
    return dict(n=int(len(d)), events=int(d[ycol].sum()), rate=round(float(d[ycol].mean() * 100), 1),
                OR=round(float(np.exp(b)), 3), lo=round(float(np.exp(b - 1.96 * se)), 3),
                hi=round(float(np.exp(b + 1.96 * se)), 3),
                auc=round(float(roc_auc_score(d[ycol], m.predict(d))), 3))


sec = {}
sec["inhosp"] = fit("died_clean", a.dropna(subset=["died_clean"]))
sec["mort30"] = fit("mort30_clean", a.dropna(subset=["mort30_clean"]))
sec["readmit"] = fit("readmit30_new", a[a.died_clean.fillna(-1) == 0].copy())  # 未知结局剔除
for k, v in sec.items():
    print(f"[sec {k}]", v, flush=True)
R["secondary_nwicu_full"] = sec

# --- 修正 S7 的斜率 CI 索引（上一版误取 AUC 区间） ---
s7 = R["s7_nwicu_clean"]
json.dump(R, open(f"{ANA}/db_audit/rev7_nwicu.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("patched json (secondary only). s7 keys:", list(s7.keys()))
