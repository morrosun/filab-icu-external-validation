# -*- coding: utf-8 -*-
"""Round-5 核查（需数据库）：
 A. 从最终分析队列重新输出 30 项逐项缺失率（写作 S1 + SF1 热图的唯一来源）
 B. NWICU 死亡定义敏感性（dod<=出院 vs dod<=出院+1天；dod<入院；30 天）
 C. 父队列 ≥10/≥15/≥20 与早死排除，改用 died~fi10+age+male + 患者聚类稳健 SE
输出 analysis/db_audit/rev5_db.json + per_lab_missingness_final.csv
"""
import os, json, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import psycopg2
import build_analysis as ba

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



ROOT = ROOT
ANA = os.path.join(ROOT, "analysis")
OUT = os.path.join(ANA, "db_audit")
os.makedirs(OUT, exist_ok=True)
RES = {}
PG = dict(**_DB_AUTH)

def conn(db):
    c = dict(PG); c["dbname"] = db; return psycopg2.connect(**c)

def orclust(df, form="died ~ fi10 + age + male", groups=None):
    g = groups if groups is not None else df["subject_id"].values
    r = smf.glm(form, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": g})
    b, se = r.params["fi10"], r.bse["fi10"]
    return dict(n=int(len(df)), deaths=int(df["died"].sum()),
                OR=round(float(np.exp(b)), 3), lo=round(float(np.exp(b - 1.96 * se)), 3),
                hi=round(float(np.exp(b + 1.96 * se)), 3), p=float(r.pvalues["fi10"]))

# ---------------- A. 逐项缺失率（最终队列）----------------
print("[A] per-lab missingness ...", flush=True)
miss = {}
for db, fn in [("MIMIC-IV", ba.fi_mimic), ("NWICU", ba.fi_nwicu), ("eICU", ba.fi_eicu)]:
    key = {"MIMIC-IV": "mimiciv", "NWICU": "nwicu", "eICU": "eicu"}[db]
    a = pd.read_csv(f"{ANA}/analysis_{key}.csv").dropna(subset=["fi_lab"]).copy()
    a["stay_id"] = a["stay_id"].astype(str)
    ids = tuple(a["stay_id"].tolist())
    L = fn(ids); L["stay_id"] = L["stay_id"].astype(str)
    rows = {}
    for it in ba.ANALYTES:
        mn, mx = L.get(f"{it}_min"), L.get(f"{it}_max")
        if mn is None:
            rows[it] = 100.0; continue
        assessed = (mn.notna() | mx.notna())
        rows[it] = round(float(100 * (1 - assessed.mean())), 1)
    miss[db] = rows
    print(f"   {db}: n={len(a)} rdw={rows.get('rdw')} mcv={rows.get('mcv')} inr={rows.get('inr')} ld={rows.get('ld_ldh')}", flush=True)
RES["missingness"] = miss
mdf = pd.DataFrame(miss)
mdf.index.name = "item"
mdf.to_csv(os.path.join(OUT, "per_lab_missingness_final.csv"), encoding="utf-8")
print("   → per_lab_missingness_final.csv", flush=True)

# ---------------- B. NWICU 死亡定义敏感性 ----------------
print("[B] NWICU death definition ...", flush=True)
n = pd.read_csv(f"{ANA}/analysis_nwicu.csv").dropna(subset=["fi_lab"]).copy()
n["stay_id"] = n["stay_id"].astype(str)
with conn("nwicu") as c:
    ev = pd.read_sql_query("""
        SELECT i.stay_id, a.admittime, a.dischtime, p.dod,
               CASE WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime THEN 1 ELSE 0 END AS invivo_strict,
               CASE WHEN p.dod IS NOT NULL AND p.dod >  a.dischtime
                     AND p.dod <= a.dischtime + INTERVAL '1 day' THEN 1 ELSE 0 END AS grace_only,
               CASE WHEN p.dod IS NOT NULL AND p.dod <  a.admittime THEN 1 ELSE 0 END AS dod_before_admin,
               CASE WHEN p.dod IS NOT NULL AND p.dod <= a.admittime + INTERVAL '30 day' THEN 1 ELSE 0 END AS d30
        FROM icu.icustays i
        JOIN hosp.admissions a ON a.hadm_id = i.hadm_id
        LEFT JOIN hosp.patients p ON p.subject_id = i.subject_id""", c)
ev["stay_id"] = ev["stay_id"].astype(str)
m = n.merge(ev, on="stay_id", how="left")
strict = m["invivo_strict"].fillna(0).astype(int)
grace = m["grace_only"].fillna(0).astype(int)
RES["nwicu_death"] = {
    "n": int(len(m)),
    "in_hospital_strict": int(strict.sum()),
    "grace_only_added": int(grace.sum()),
    "dod_before_admission": int(m["dod_before_admin"].fillna(0).sum()),
    "d30": int(m["d30"].fillna(0).sum()),
    "current_definition": int(((strict + grace) > 0).sum()),
}
# 用严格定义重跑
for lab, dd, y in [("strict", m.copy(), strict), ("current", m.copy(), (strict + grace).clip(0, 1))]:
    dd = dd.dropna(subset=["fi_lab", "age", "male"]).copy()
    dd["died"] = y.loc[dd.index].values
    dd["fi10"] = dd["fi_lab"] * 10
    RES["nwicu_death"][f"OR_{lab}"] = orclust(dd)
print("   ", RES["nwicu_death"], flush=True)

# ---------------- C. 父队列阈值 + 早死（cluster-robust）----------------
def deficits(L):
    cols = []
    for a in ba.ANALYTES:
        lo, hi = ba.D[a][3], ba.D[a][4]
        mn, mx = L.get(f"{a}_min"), L.get(f"{a}_max")
        if mn is None: continue
        assessed = mn.notna() | mx.notna()
        cols.append(((mn < lo) | (mx > hi)).fillna(False).where(assessed))
    M = pd.concat(cols, axis=1)
    return M.notna().sum(axis=1), M.sum(axis=1, min_count=1).fillna(0)

def parent(db, fn, gkey="subject_id"):
    ana = pd.read_csv(f"{ANA}/analysis_{db}.csv"); ana["stay_id"] = ana["stay_id"].astype(str)
    ana["n_assessed"] = pd.to_numeric(ana["n_assessed"], errors="coerce")
    band = (ana.n_assessed >= 10) & (ana.n_assessed < 15)
    ids = tuple(int(x) for x in ana.loc[band, "stay_id"])
    print(f"   [{db}] band 10-14 = {int(band.sum())}, extracting...", flush=True)
    if ids:
        L = fn(ids); L["stay_id"] = L["stay_id"].astype(str)
        an2, dn2 = deficits(L)
        fv = np.where(an2 > 0, dn2 / an2, np.nan)
        ana.loc[band, "fi_lab"] = ana.loc[band, "stay_id"].map(dict(zip(L["stay_id"], fv)))
    an = ana["n_assessed"]
    res = {"total": int(len(ana)), "n_ge10": int((an >= 10).sum()), "n_ge20": int((an >= 20).sum())}
    for T in [10, 15, 20]:
        d = ana[an >= T].dropna(subset=["fi_lab", "died", "age", "male"]).copy()
        d["fi10"] = d["fi_lab"] * 10
        res[f">={T}"] = orclust(d, groups=d[gkey].values)
    # 早死排除：用真实 ICU 停留时间
    with conn(db) as c:
        tbl = "mimiciv_icu.icustays" if db == "mimiciv" else "icu.icustays"
        los = pd.read_sql_query(f"SELECT stay_id, EXTRACT(EPOCH FROM (outtime-intime))/86400.0 AS los FROM {tbl}", c)
    los["stay_id"] = los["stay_id"].astype(str)
    d15 = ana[ana.n_assessed >= 15].dropna(subset=["fi_lab", "died", "age", "male"]).copy()
    d15["fi10"] = d15["fi_lab"] * 10
    d15["los"] = d15["stay_id"].map(dict(zip(los["stay_id"], los["los"])))
    early = (d15["died"] == 1) & (d15["los"] < 1)
    res["early_death_n"] = int(early.sum())
    res["exclude_early"] = orclust(d15[~early], groups=d15.loc[~early, gkey].values)
    return res

RES["parent_mimic"] = parent("mimiciv", ba.fi_mimic)
print("   MIMIC:", RES["parent_mimic"], flush=True)
try:
    RES["parent_nwicu"] = parent("nwicu", ba.fi_nwicu)
    print("   NWICU:", RES["parent_nwicu"], flush=True)
except Exception as e:
    RES["parent_nwicu"] = {"ERROR": str(e)}
    print("   NWICU err:", e, flush=True)

json.dump(RES, open(os.path.join(OUT, "rev5_db.json"), "w"), ensure_ascii=False, indent=2)
print("WROTE rev5_db.json")
