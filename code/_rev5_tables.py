# -*- coding: utf-8 -*-
"""Round-5 表值重算（仅用本地 CSV）：全部 OR 改用患者聚类稳健 SE
 1) MIMIC M1–M4 FI-Lab 的聚类稳健 Wald 检验（替代普通 LR）
 2) S5 次要结局（院内死亡/30天死亡/30天再入院）× 三库
 3) S6A 精简 26 项：Original 与 Reduced 两列（cluster-robust）
 4) eICU APACHE 增量：n/事件/APACHE 缺失/配对 ΔAUC CI
输出 analysis/db_audit/rev5_tables.json
"""
import json, numpy as np, pandas as pd
import os
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
import psycopg2

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
OUT = f"{ANA}/db_audit/rev5_tables.json"
rng = np.random.default_rng(20261025)
RES = {}

def fit_clust(df, form, g):
    r = smf.glm(form, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(g)})
    return r

def orrow(r, term):
    b, se = r.params[term], r.bse[term]
    return dict(OR=round(float(np.exp(b)), 3), lo=round(float(np.exp(b-1.96*se)), 3),
                hi=round(float(np.exp(b+1.96*se)), 3), p=float(r.pvalues[term]),
                wald_chi2=round(float((b/se)**2), 1))

def boot_idx(g):
    g = np.asarray(g); ug = np.unique(g)
    return ug, {u: np.where(g == u)[0] for u in ug}

def auc_ci(y, p, g, B=1000):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ug, idx = boot_idx(g); a = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if y[ii].min() == y[ii].max(): continue
        a.append(roc_auc_score(y[ii], p[ii]))
    return (round(float(roc_auc_score(y, p)), 3),
            round(float(np.nanpercentile(a, 2.5)), 3), round(float(np.nanpercentile(a, 97.5)), 3))

# ---------- 1) M1–M4 cluster-robust Wald ----------
mim = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
mim["stay_id"] = mim["stay_id"].astype(str); mim["fi10"] = mim["fi_lab"]*10
rev = pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv"); rev["stay_id"] = rev["stay_id"].astype(str)
m = mim.merge(rev[["stay_id","sofa","oasis","charlson","vent","vaso","rrt"]],
              on="stay_id", how="left", suffixes=("","_rev"))
m = m.dropna(subset=["died","fi10","age","male","charlson","sofa","oasis","vent","vaso","rrt"])
specs = [("M1","died ~ fi10 + age + male"),
         ("M2","died ~ fi10 + age + male + charlson"),
         ("M3","died ~ fi10 + age + male + charlson + sofa + oasis"),
         ("M4","died ~ fi10 + age + male + charlson + sofa + oasis + vent + vaso + rrt")]
RES["wald"] = {}
for lab, f in specs:
    r = fit_clust(m, f, m["subject_id"])
    RES["wald"][lab] = dict(n=int(len(m)), **orrow(r, "fi10"))
    print(f"[Wald {lab}] OR={RES['wald'][lab]['OR']} χ²={RES['wald'][lab]['wald_chi2']} p={RES['wald'][lab]['p']:.3g}", flush=True)

# ---------- 2) S5 次要结局（cluster-robust）----------
def load_eicu_pid():
    c = psycopg2.connect(**_DB_AUTH, dbname="eicu")
    pid = pd.read_sql_query("SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", c)
    c.close(); return pid

RES["secondary"] = []
for db, glab in [("mimiciv","MIMIC-IV")]:
    o = pd.read_csv(f"{ANA}/outcomes_{db}.csv"); o["stay_id"] = o["stay_id"].astype(str)
    if db == "eicu":
        o = o.drop(columns=[c for c in ["subject_id"] if c in o.columns])
        o = o.merge(load_eicu_pid(), on="stay_id", how="left").rename(columns={"uniquepid":"subject_id"})
    o["fi10"] = o["fi_lab"]*10
    for oc, col in [("In-hospital mortality","died"), ("30-day mortality","mort30"), ("30-day readmission","readmit30")]:
        if col not in o.columns: continue
        d = o.dropna(subset=[col,"fi10","age","male"]).copy()
        if col == "readmit30":
            d = d[d["died"] == 0].copy()
        if len(d) < 50: continue
        d["y"] = d[col].astype(float)
        r = fit_clust(d, "y ~ fi10 + age + male", d["subject_id"])
        pr = r.predict(d)
        auc = roc_auc_score(d["y"], pr)
        row = dict(db=glab, outcome=oc, n=int(len(d)), events=int(d["y"].sum()),
                   rate=round(float(100*d["y"].mean()),1), **orrow(r,"fi10"), auc=round(float(auc),3))
        RES["secondary"].append(row)
        print(f"[S5] {glab} {oc}: n={row['n']} OR={row['OR']} ({row['lo']}-{row['hi']})", flush=True)

# ---------- 3) S6A 精简 26 项（cluster-robust）----------
red = pd.read_csv(f"{ANA}/filab_reduced_mimiciv.csv") if False else pd.read_csv(f"{ANA}/../filab/filab_reduced_mimiciv.csv")
red["stay_id"] = red["stay_id"].astype(str)
m2 = m.merge(red[["stay_id","fi_lab_reduced","n_assessed_reduced"]], on="stay_id", how="left")
m2 = m2.dropna(subset=["fi_lab_reduced"])
m2["fi10r"] = m2["fi_lab_reduced"]*10
rspecs = [("M1","y ~ fi10 + age + male","y ~ fi10r + age + male"),
          ("M2","y ~ fi10 + age + male + charlson","y ~ fi10r + age + male + charlson"),
          ("M3","y ~ fi10 + age + male + charlson + sofa + oasis","y ~ fi10r + age + male + charlson + sofa + oasis"),
          ("M4","y ~ fi10 + age + male + charlson + sofa + oasis + vent + vaso + rrt",
                "y ~ fi10r + age + male + charlson + sofa + oasis + vent + vaso + rrt")]
m2["y"] = m2["died"].astype(float)
RES["reduced"] = {}
for lab, fo, fr in rspecs:
    ro = fit_clust(m2, fo, m2["subject_id"]); rr = fit_clust(m2, fr, m2["subject_id"])
    RES["reduced"][lab] = dict(n=int(len(m2)),
        original=orrow(ro,"fi10"), reduced=orrow(rr,"fi10r"),
        auc_reduced=round(float(roc_auc_score(m2["y"], rr.predict(m2))),3))
    print(f"[S6A {lab}] orig {RES['reduced'][lab]['original']['OR']} / red {RES['reduced'][lab]['reduced']['OR']}", flush=True)

# ---------- 4) eICU APACHE 增量 ----------
e = pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
e["stay_id"] = e["stay_id"].astype(str); e["fi10"] = e["fi_lab"]*10
re_ = pd.read_csv(f"{ANA}/rev/rev_eicu.csv"); re_["stay_id"] = re_["stay_id"].astype(str)
e = e.drop(columns=[c for c in ["subject_id"] if c in e.columns])
e = e.merge(re_[["stay_id","apache"]], on="stay_id", how="left")
e = e.merge(load_eicu_pid(), on="stay_id", how="left").rename(columns={"uniquepid":"subject_id"})
miss_apache = float(100*e["apache"].isna().mean())
ea = e.dropna(subset=["apache","died","age","male","fi10"]).copy()
ea["y"] = ea["died"].astype(float)
ra = fit_clust(ea, "y ~ age + male + apache", ea["subject_id"])
rb = fit_clust(ea, "y ~ age + male + apache + fi10", ea["subject_id"])
pa, pb = ra.predict(ea), rb.predict(ea)
pav, pbv = pa.values, pb.values
aa = roc_auc_score(ea["y"], pav); ab = roc_auc_score(ea["y"], pbv)
ug, idx = boot_idx(ea["subject_id"]); dA = []
for _ in range(1000):
    pick = rng.choice(ug, size=len(ug), replace=True)
    ii = np.concatenate([idx[u] for u in pick])
    if ea["y"].values[ii].min() == ea["y"].values[ii].max(): continue
    dA.append(roc_auc_score(ea["y"].values[ii], pbv[ii]) - roc_auc_score(ea["y"].values[ii], pav[ii]))
RES["eicu_apache"] = dict(n=int(len(ea)), events=int(ea["y"].sum()), apache_missing_pct=round(miss_apache,1),
                          auc_base=round(float(aa),3), auc_full=round(float(ab),3),
                          dAUC=round(float(ab-aa),4),
                          dAUC_lo=round(float(np.nanpercentile(dA,2.5)),4),
                          dAUC_hi=round(float(np.nanpercentile(dA,97.5)),4),
                          fi_or=orrow(rb,"fi10"), apache_or=orrow(rb,"apache"))
print("[eICU APACHE]", RES["eicu_apache"], flush=True)

json.dump(RES, open(OUT,"w"), ensure_ascii=False, indent=2)
print("WROTE", OUT)
