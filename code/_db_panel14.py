# -*- coding: utf-8 -*-
"""第五轮c：按**当前**缺失率规则重新推导 harmonized 面板
规则：三库 FI-Lab 可评估队列中缺失率均 ≤20% 的项目 -> 14 项
（sodium, potassium, chloride, bicarbonate, BUN, creatinine, glucose, calcium,
  hematocrit, hemoglobin, platelet, anion gap, RDW, MCV）
覆盖规则：≥12/14（≈86%，与 11 项时 ≥9/11≈82% 同量级）
输出 analysis/db_audit/panel14.json + fixed_fi14_{db}.csv
"""
import os, json, time
import numpy as np, pandas as pd, psycopg2
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
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



ROOT = ROOT; ANA = os.path.join(ROOT, "analysis")
OUTJ = os.path.join(ANA, "db_audit", "panel14.json")
PG = dict(**_DB_AUTH)
ITEMS = ["sodium","potassium","chloride","bicarbonate","bun","creatinine","glucose",
         "calcium","hematocrit","hemoglobin","platelet","aniongap","rdw","mcv"]
MINP = 12
rng = np.random.default_rng(20261025)
RES = {"items": ITEMS, "min_assessed": MINP}

def conn(db):
    c = dict(PG); c["dbname"] = db; return psycopg2.connect(**c)

def fi_of(L, items, minn):
    cols = []
    for a in items:
        lo, hi = ba.D[a][3], ba.D[a][4]
        mn, mx = L.get(f"{a}_min"), L.get(f"{a}_max")
        if mn is None: continue
        assessed = mn.notna() | mx.notna()
        cols.append(((mn < lo) | (mx > hi)).fillna(False).where(assessed))
    M = pd.concat(cols, axis=1)
    an = M.notna().sum(axis=1); dn = M.sum(axis=1, min_count=1).fillna(0)
    fi = np.full(len(L), np.nan); m = (an >= minn).values
    fi[m] = (dn[m] / an[m]).values
    return pd.Series(fi, index=L.index), an

def _logit2(y, x, iters=40):
    X = np.column_stack([np.ones_like(x), x]); b = np.array([0.0, 1.0])
    for _ in range(iters):
        p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); w = np.clip(p*(1-p), 1e-12, None)
        try: step = np.linalg.solve(X.T@(X*w[:, None]) + 1e-8*np.eye(2), X.T@(y-p))
        except np.linalg.LinAlgError: break
        b = b + step
        if np.abs(step).max() < 1e-9: break
    return b[0], b[1]

def perf(y, p, g, B=400):
    y = np.asarray(y, float); p = np.asarray(p, float)
    lp = np.log(np.clip(p, 1e-12, 1-1e-12)/(1-np.clip(p, 1e-12, 1-1e-12)))
    _, slope = _logit2(y, lp)
    gl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    obs, exp = y.sum(), p.sum(); oe = obs/exp
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    bA, bS = [], []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if y[ii].min() == y[ii].max(): continue
        bA.append(roc_auc_score(y[ii], p[ii])); bS.append(_logit2(y[ii], lp[ii])[1])
    q = lambda a, t: round(float(np.nanpercentile(a, t)), 4) if len(a) else None
    return dict(n=int(len(y)), deaths=int(obs), auc=round(float(roc_auc_score(y, p)), 4),
                auc_lo=q(bA, 2.5), auc_hi=q(bA, 97.5), brier=round(float(brier_score_loss(y, p)), 4),
                oe=round(float(oe), 4),
                oe_lo=round(float(oe*np.exp(-1.96/np.sqrt(obs))), 4),
                oe_hi=round(float(oe*np.exp(1.96/np.sqrt(obs))), 4),
                slope=round(slope, 4), slope_lo=q(bS, 2.5), slope_hi=q(bS, 97.5),
                citl=round(float(gl.params[0]), 4),
                citl_lo=round(float(gl.params[0]-1.96*gl.bse[0]), 4),
                citl_hi=round(float(gl.params[0]+1.96*gl.bse[0]), 4))

def main():
    data = {}
    for db, fn in [("mimiciv", ba.fi_mimic), ("nwicu", ba.fi_nwicu), ("eicu", ba.fi_eicu)]:
        t0 = time.time()
        a = pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
        a["stay_id"] = a["stay_id"].astype(str)
        L = fn(tuple(a["stay_id"].tolist())); L["stay_id"] = L["stay_id"].astype(str)
        fi, an = fi_of(L, ITEMS, MINP)
        L = L.copy(); L["fi14"] = fi.values; L["n14"] = an.values
        d = a.merge(L[["stay_id", "fi14", "n14"]], on="stay_id", how="left")
        if db == "eicu":
            d = d.drop(columns=[c for c in ["subject_id"] if c in d.columns])
            with conn("eicu") as c:
                pid = pd.read_sql_query("SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", c)
            d = d.merge(pid, on="stay_id", how="left").rename(columns={"uniquepid": "subject_id"})
        d.to_csv(f"{ANA}/db_audit/fixed_fi14_{db}.csv", index=False)
        data[db] = d
        ok = d["fi14"].notna()
        RES[db] = dict(n_cohort=int(len(d)), n14=int(ok.sum()), pct=round(float(ok.mean()*100), 1),
                       n14_median=float(d.loc[ok, "n14"].median()),
                       fi_median=round(float(d.loc[ok, "fi14"].median()), 4),
                       fi_iqr=[round(float(d.loc[ok, "fi14"].quantile(.25)), 4),
                               round(float(d.loc[ok, "fi14"].quantile(.75)), 4)])
        print(f"[{db}] {RES[db]} ({time.time()-t0:.0f}s)", flush=True)

    # 派生系数（MIMIC, 14 项）
    def fit(d, col):
        dd = d.dropna(subset=[col, "died", "age", "male"]).copy()
        dd["fi10"] = dd[col]*10
        m = smf.logit("died ~ fi10 + age + male", data=dd).fit(disp=0)
        return dd, m
    d14, m14 = fit(data["mimiciv"], "fi14")
    d30, m30 = fit(data["mimiciv"], "fi_lab")
    RES["coef14"] = {k: round(float(v), 6) for k, v in m14.params.items()}
    RES["coef14_se"] = {k: round(float(v), 6) for k, v in m14.bse.items()}
    b, se = m14.params["fi10"], m14.bse["fi10"]
    RES["mimic_local_OR"] = dict(OR=round(float(np.exp(b)), 3),
                                lo=round(float(np.exp(b-1.96*se)), 3),
                                hi=round(float(np.exp(b+1.96*se)), 3), n=int(m14.nobs))
    RES["corr"] = {}
    RES["common"] = {}
    for db, d in data.items():
        dc = d.dropna(subset=["fi_lab", "fi14", "died", "age", "male"]).copy()
        RES["corr"][db] = dict(n=int(len(dc)),
                               pearson=round(float(dc[["fi_lab", "fi14"]].corr().iloc[0, 1]), 4),
                               spearman=round(float(dc[["fi_lab", "fi14"]].corr(method="spearman").iloc[0, 1]), 4),
                               mad=round(float((dc["fi14"]-dc["fi_lab"]).abs().mean()), 3))
        # 本地重拟合（14 项）
        mm = smf.logit("died ~ fi10 + age + male", data=dc.assign(fi10=dc["fi14"]*10)).fit(disp=0)
        bb, ss = mm.params["fi10"], mm.bse["fi10"]
        p30 = np.asarray((lambda lp: 1/(1+np.exp(-lp)))(
            m30.params["Intercept"] + m30.params["fi10"]*dc["fi_lab"]*10
            + m30.params["age"]*dc["age"] + m30.params["male"]*dc["male"]))
        p14 = np.asarray((lambda lp: 1/(1+np.exp(-lp)))(
            RES["coef14"]["Intercept"] + RES["coef14"]["fi10"]*dc["fi14"]*10
            + RES["coef14"]["age"]*dc["age"] + RES["coef14"]["male"]*dc["male"]))
        g = dc["subject_id"].values if "subject_id" in dc.columns else np.arange(len(dc))
        y = dc["died"].values
        ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
        dA = []
        for _ in range(400):
            pick = rng.choice(ug, size=len(ug), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            if y[ii].min() == y[ii].max(): continue
            dA.append(roc_auc_score(y[ii], p30[ii]) - roc_auc_score(y[ii], p14[ii]))
        RES["common"][db] = dict(
            n=int(len(dc)), deaths=int(y.sum()),
            local14=dict(OR=round(float(np.exp(bb)), 3), lo=round(float(np.exp(bb-1.96*ss)), 3),
                         hi=round(float(np.exp(bb+1.96*ss)), 3)),
            ext30=perf(y, p30, g), ext14=perf(y, p14, g),
            dAUC=round(float(roc_auc_score(y, p30)-roc_auc_score(y, p14)), 4),
            dAUC_lo=round(float(np.nanpercentile(dA, 2.5)), 4),
            dAUC_hi=round(float(np.nanpercentile(dA, 97.5)), 4))
        print(f"[common {db}] n={len(dc)} dAUC={RES['common'][db]['dAUC']}", flush=True)
    json.dump(RES, open(OUTJ, "w"), ensure_ascii=False, indent=2)
    print("WROTE", OUTJ)

if __name__ == "__main__":
    main()
