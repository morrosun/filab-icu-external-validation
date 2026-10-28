# -*- coding: utf-8 -*-
"""NWICU 死亡日期异常核查：51 例 dod < admittime 的成因
逐条输出 gap（天）、该 subject 的其它住院区间是否覆盖 dod、是否同一 hadm 多次 ICU。
输出 analysis/db_audit/nwicu_dod_audit.json
"""
import json, numpy as np, pandas as pd, psycopg2
import os

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



PG = dict(**_DB_AUTH, dbname="nwicu")
RES = {}
with psycopg2.connect(**PG) as c:
    ev = pd.read_sql_query("""
        SELECT i.stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime,
               a.admittime, a.dischtime, p.dod,
               EXTRACT(EPOCH FROM (a.admittime - p.dod))/86400.0 AS gap_days_before_adm,
               EXTRACT(EPOCH FROM (a.dischtime - p.dod))/86400.0 AS gap_days_before_disch
        FROM icu.icustays i
        JOIN hosp.admissions a ON a.hadm_id = i.hadm_id
        JOIN hosp.patients p   ON p.subject_id = i.subject_id
        WHERE p.dod IS NOT NULL""", c)
    adm = pd.read_sql_query("""
        SELECT subject_id, hadm_id, admittime, dischtime FROM hosp.admissions""", c)

ev["stay_id"] = ev["stay_id"].astype(str)
for _c in ["intime","outtime","admittime","dischtime","dod"]:
    ev[_c] = pd.to_datetime(ev[_c], errors="coerce")
for _c in ["admittime","dischtime"]:
    adm[_c] = pd.to_datetime(adm[_c], errors="coerce")
ana = pd.read_csv(os.path.join(ROOT, 'analysis', 'analysis_nwicu.csv')).dropna(subset=["fi_lab"])
ana["stay_id"] = ana["stay_id"].astype(str)
d = ana[["stay_id", "subject_id", "died"]].merge(ev, on="stay_id", how="left", suffixes=("", "_ev"))

bad = d[(d["gap_days_before_adm"] > 0)].copy()
RES["n_flagged"] = int(len(bad))
RES["n_died"] = int(d["died"].sum())
RES["gap_days_stats"] = {
    "min": round(float(bad["gap_days_before_adm"].min()), 2),
    "median": round(float(bad["gap_days_before_adm"].median()), 2),
    "max": round(float(bad["gap_days_before_adm"].max()), 2),
    "p90": round(float(bad["gap_days_before_adm"].quantile(0.9)), 2),
}
RES["gap_bins"] = {
    "<=0.5d": int((bad["gap_days_before_adm"] <= 0.5).sum()),
    "0.5-1d": int(((bad["gap_days_before_adm"] > 0.5) & (bad["gap_days_before_adm"] <= 1)).sum()),
    "1-2d": int(((bad["gap_days_before_adm"] > 1) & (bad["gap_days_before_adm"] <= 2)).sum()),
    "2-7d": int(((bad["gap_days_before_adm"] > 2) & (bad["gap_days_before_adm"] <= 7)).sum()),
    ">7d": int((bad["gap_days_before_adm"] > 7).sum()),
}
# dod 是否落在同一 subject 的另一次住院窗口内（说明是"别的住院"的死亡）
cov = []
for _, r in bad.iterrows():
    others = adm[(adm["subject_id"] == r["subject_id"]) & (adm["hadm_id"] != r["hadm_id"])]
    _dod = pd.Timestamp(r["dod"])
    hit = others[(others["admittime"] <= _dod) & (others["dischtime"] >= _dod)]
    cov.append(len(hit) > 0)
bad["dod_in_other_admission"] = cov
RES["dod_in_other_admission"] = int(np.sum(cov))
# 同一 hadm 内是否还有别的 ICU stay 覆盖 dod
with psycopg2.connect(**PG) as c:
    n_icu = pd.read_sql_query("""SELECT hadm_id, count(*) AS n FROM icu.icustays GROUP BY hadm_id""", c)
d = d.merge(n_icu, on="hadm_id", how="left")
bad = bad.merge(n_icu, on="hadm_id", how="left")
RES["multi_icu_same_hadm"] = int((bad["n"] > 1).sum())
# 这 51 例里 died 标记
RES["of_flagged_marked_died"] = int(bad["died"].sum())
RES["sample_rows"] = bad.sort_values("gap_days_before_adm", ascending=False)[[
    "stay_id", "hadm_id", "intime", "admittime", "dischtime", "dod",
    "gap_days_before_adm", "n", "dod_in_other_admission"]].head(10).astype(str).to_dict("records")

# 各规则的事件数（同一批 10,819 例）
def cnt(mask): return int(mask.sum())
RES["rule_counts"] = {
    "no_lower_bound (dod<=dischtime)": cnt(d["gap_days_before_disch"] >= 0),
    "admit_to_discharge": cnt((d["gap_days_before_adm"] >= 0) & (d["gap_days_before_disch"] >= 0)),
    "admit_to_discharge+1d": cnt((d["gap_days_before_adm"] >= 0) & (d["gap_days_before_disch"] >= -1)),
    "discharge+1d_no_lower": cnt(d["gap_days_before_disch"] >= -1),
}
json.dump(RES, open(os.path.join(ROOT, 'analysis', 'db_audit', 'nwicu_dod_audit.json'), "w"),
          ensure_ascii=False, indent=2, default=str)
print(json.dumps(RES, ensure_ascii=False, indent=2, default=str)[:3000])
