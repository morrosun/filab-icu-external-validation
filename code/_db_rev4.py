# -*- coding: utf-8 -*-
"""Round-4 核查：11 项面板共同样本比较 + 完整指标 CI + 患者/重复住院计数 + M1-M4 权威数值
输出: analysis/db_audit/rev4.json
"""
import os, json, time
import numpy as np, pandas as pd
import psycopg2, statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


OUT = "analysis/db_audit/rev4.json"
os.makedirs("analysis/db_audit", exist_ok=True)
CONN = dict(**_DB_AUTH)

FIXED = {
    "sodium":("sodium",100010,"sodium",135.0,145.0),
    "potassium":("potassium",100011,"potassium",3.5,5.5),
    "chloride":("chloride",100012,"chloride",98.0,110.0),
    "bicarbonate":("bicarbonate",100013,"bicarbonate",22.0,29.0),
    "bun":("bun",100004,"BUN",7.0,20.0),
    "creatinine":("creatinine",100002,"creatinine",0.5,1.3),
    "glucose":("glucose",100001,"glucose",70.0,140.0),
    "calcium":("calcium",100003,"calcium",8.5,10.5),
    "hematocrit":("hematocrit",100006,"Hct",36.0,50.0),
    "hemoglobin":("hemoglobin",100007,"Hgb",10.0,17.0),
    "platelet":("platelets",100014,"platelets x 1000",150.0,450.0),
}
ITEMS = list(FIXED.keys()); MIN_FIXED = 9

def conn(db):
    c = dict(CONN); c["dbname"] = db; return psycopg2.connect(**c)

def chunks(seq, n=1500):
    seq = list(seq)
    for i in range(0, len(seq), n): yield seq[i:i+n]

def fi_mimic_fixed(ids):
    cols = ", ".join([f"{FIXED[a][0]}_min AS {a}_min, {FIXED[a][0]}_max AS {a}_max" for a in ITEMS])
    sql = f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {{L}}"
    parts = []
    for ch in chunks(ids):
        lit = "(" + ",".join(str(int(x)) for x in ch) + ")"
        with conn("mimiciv") as c:
            with c.cursor() as cur: cur.execute("SET statement_timeout='600s'")
            parts.append(pd.read_sql_query(sql.format(L=lit), c))
    df = pd.concat(parts, ignore_index=True); df["stay_id"] = df["stay_id"].astype(str); return df

def fi_nwicu_fixed(ids):
    ii = tuple(FIXED[a][1] for a in ITEMS)
    case = ", ".join([f"MAX(CASE WHEN le.itemid={FIXED[a][1]} THEN le.valuenum END) {a}_max, "
                      f"MIN(CASE WHEN le.itemid={FIXED[a][1]} THEN le.valuenum END) {a}_min" for a in ITEMS])
    sql = f"""SELECT i.stay_id, {case} FROM icu.icustays i
        JOIN hosp.labevents le ON le.subject_id=i.subject_id AND le.hadm_id=i.hadm_id
         AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
         AND le.valuenum IS NOT NULL AND le.itemid IN {ii}
        WHERE i.stay_id IN {{L}} GROUP BY i.stay_id"""
    parts = []
    for ch in chunks(ids):
        lit = "(" + ",".join("'%s'" % x for x in ch) + ")"
        with conn("nwicu") as c:
            with c.cursor() as cur: cur.execute("SET statement_timeout='600s'")
            parts.append(pd.read_sql_query(sql.format(L=lit), c))
    df = pd.concat(parts, ignore_index=True); df["stay_id"] = df["stay_id"].astype(str); return df

def fi_eicu_fixed(ids):
    nm = tuple(FIXED[a][2] for a in ITEMS)
    case = ", ".join([f"MAX(CASE WHEN labname='{FIXED[a][2]}' THEN labresult END) {a}_max, "
                      f"MIN(CASE WHEN labname='{FIXED[a][2]}' THEN labresult END) {a}_min" for a in ITEMS])
    nl = "(" + ",".join("'%s'" % n for n in nm) + ")"
    sql = f"""SELECT patientunitstayid AS stay_id, {case} FROM eicu_crd.lab
        WHERE patientunitstayid IN {{L}} AND labresultoffset BETWEEN 0 AND 1440
          AND labresult IS NOT NULL AND labname IN {nl} GROUP BY patientunitstayid"""
    parts = []
    for ch in chunks(ids):
        lit = "(" + ",".join("'%s'" % x for x in ch) + ")"
        with conn("eicu") as c:
            with c.cursor() as cur: cur.execute("SET statement_timeout='600s'")
            parts.append(pd.read_sql_query(sql.format(L=lit), c))
    df = pd.concat(parts, ignore_index=True); df["stay_id"] = df["stay_id"].astype(str); return df

def compute_fixed(L):
    defs = []
    for a in ITEMS:
        lo, hi = FIXED[a][3], FIXED[a][4]
        mn, mx = L.get(f"{a}_min"), L.get(f"{a}_max")
        assessed = mn.notna() | mx.notna()
        defs.append(((mn < lo) | (mx > hi)).fillna(False).where(assessed))
    M = pd.concat(defs, axis=1)
    an = M.notna().sum(axis=1); dn = M.sum(axis=1, min_count=1).fillna(0)
    fi = np.full(len(L), np.nan); m = (an >= MIN_FIXED).values
    fi[m] = (dn[m] / an[m]).values
    return pd.Series(fi, index=L.index), an

# ---------- 指标（含 CI） ----------
def metrics(y, p, groups=None, nboot=1000, seed=20261025):
    y = np.asarray(y, float); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs = []
    for _ in range(nboot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[u] for u in pick])
        yy, pp = y[ii], p[ii]
        if yy.min() == yy.max(): continue
        aucs.append(roc_auc_score(yy, pp))
    auc = roc_auc_score(y, p); brier = brier_score_loss(y, p)
    lp = np.log(np.clip(p, 1e-9, 1-1e-9) / (1 - np.clip(p, 1e-9, 1-1e-9)))
    cm = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    # CITL: offset slope=1
    try:
        cI = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                    offset=lp).fit()
        citl, citl_se = float(cI.params[0]), float(cI.bse[0])
    except Exception:
        citl, citl_se = float("nan"), float("nan")
    obs, exp = y.sum(), p.sum()
    # Poisson CI for O:E: (O/E)*exp(±1.96/sqrt(O))
    if obs > 0 and exp > 0:
        oe = obs/exp
        oe_lo = oe*np.exp(-1.96/np.sqrt(obs)); oe_hi = oe*np.exp(1.96/np.sqrt(obs))
    else:
        oe = oe_lo = oe_hi = np.nan
    return dict(n=int(len(y)), deaths=int(obs),
                auc=round(float(auc), 4),
                auc_lo=round(float(np.nanpercentile(aucs, 2.5)), 4) if aucs else None,
                auc_hi=round(float(np.nanpercentile(aucs, 97.5)), 4) if aucs else None,
                brier=round(float(brier), 4),
                oe=round(float(oe), 4), oe_lo=round(float(oe_lo), 4), oe_hi=round(float(oe_hi), 4),
                citl=round(citl, 4), citl_lo=round(citl-1.96*citl_se, 4), citl_hi=round(citl+1.96*citl_se, 4),
                slope=round(float(cm.params[1]), 4),
                slope_lo=round(float(cm.conf_int()[0][1]), 4),
                slope_hi=round(float(cm.conf_int()[1][1]), 4))

def fit_m1(d, fi_col):
    dd = d.dropna(subset=[fi_col, "died", "age", "male"]).copy()
    dd["fi10"] = dd[fi_col]*10
    m = smf.logit("died ~ fi10 + age + male", data=dd).fit(disp=0)
    b, se = m.params["fi10"], m.bse["fi10"]
    return dd, m, dict(n=int(m.nobs), OR=round(float(np.exp(b)), 4),
                       lo=round(float(np.exp(b-1.96*se)), 4), hi=round(float(np.exp(b+1.96*se)), 4),
                       p=float(m.pvalues["fi10"]),
                       coef={k: round(float(v), 6) for k, v in m.params.items()})

def main():
    res = {}
    data = {}
    for db in ["mimiciv", "nwicu", "eicu"]:
        t0 = time.time()
        base = pd.read_csv(f"analysis/analysis_{db}.csv")
        base["stay_id"] = base["stay_id"].astype(str)
        ids = list(base["stay_id"].unique())
        cache = f"analysis/db_audit/fixed_fi_{db}.csv"
        if os.path.exists(cache) and os.path.getsize(cache) > 1000:
            d = pd.read_csv(cache); d["stay_id"] = d["stay_id"].astype(str)
            if db == "eicu":
                d = d.drop(columns=[c for c in ["subject_id"] if c in d.columns])
                with conn("eicu") as cc:
                    pid = pd.read_sql_query(
                        "SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", cc)
                d = d.merge(pid, on="stay_id", how="left").rename(columns={"uniquepid": "subject_id"})
            data[db] = d
            g = d.dropna(subset=["fi_lab"]).groupby("subject_id")["stay_id"].nunique()
            up, rep = int(g.size), int((g >= 2).sum())
            res[f"cohort_{db}"] = dict(n_stays=int(d["fi_lab"].notna().sum()), unique_patients=up,
                                       patients_ge2=rep, pct_patients_ge2=round(rep/up*100,1))
            print(f"[{db}] cached {res[f'cohort_{db}']}", flush=True)
            continue
        L = {"mimiciv": fi_mimic_fixed, "nwicu": fi_nwicu_fixed, "eicu": fi_eicu_fixed}[db](ids)
        fi, na = compute_fixed(L)
        L = L.copy(); L["fi_fixed"] = fi.values; L["n_fixed"] = na.values
        d = base.merge(L[["stay_id", "fi_fixed", "n_fixed"]], on="stay_id", how="left")
        if db == "eicu":
            d = d.drop(columns=[c for c in ["subject_id"] if c in d.columns])
            with conn("eicu") as c:
                pid = pd.read_sql_query(
                    "SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", c)
            d = d.merge(pid, on="stay_id", how="left").rename(columns={"uniquepid": "subject_id"})
        d.to_csv(f"analysis/db_audit/fixed_fi_{db}.csv", index=False)
        data[db] = d
        # ---- 患者 / 重复住院 ----
        if "subject_id" in d.columns and d["subject_id"].notna().any():
            g = d.dropna(subset=["fi_lab"]).groupby("subject_id")["stay_id"].nunique()
            up, rep = int(g.size), int((g >= 2).sum())
        else:
            up, rep = None, None
        res[f"cohort_{db}"] = dict(n_stays=int((d["fi_lab"].notna()).sum()),
                                   unique_patients=up, patients_ge2=rep,
                                   pct_patients_ge2=(round(rep/up*100, 1) if up else None))
        print(f"[{db}] {res[f'cohort_{db}']} ({time.time()-t0:.0f}s)", flush=True)

    # ---- MIMIC M1-M4 权威数值（Died ~ fi10+age+male [+cci_aafree][+sofa+oasis][+vent+vaso+rrt]）----
    rv = pd.read_csv("analysis/rev/rev_mimiciv.csv")
    rv["stay_id"] = rv["stay_id"].astype(str)
    m0 = data["mimiciv"].merge(rv[["stay_id", "sofa", "oasis", "cci_excl_rheum", "vent", "vaso", "rrt"]],
                               on="stay_id", how="left")
    m0["fi10"] = m0["fi_lab"]*10
    ladder = {}
    specs = [("M1", "died ~ fi10 + age + male"),
             ("M2", "died ~ fi10 + age + male + cci_excl_rheum"),
             ("M3", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
             ("M4", "died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt")]
    for lab, f in specs:
        dd = m0.dropna(subset=["died", "fi10", "age", "male", "cci_excl_rheum", "sofa", "oasis"])
        mm = smf.logit(f, data=dd).fit(disp=0)
        b, se = mm.params["fi10"], mm.bse["fi10"]
        ladder[lab] = dict(n=int(mm.nobs), OR=round(float(np.exp(b)), 4),
                           lo=round(float(np.exp(b-1.96*se)), 4), hi=round(float(np.exp(b+1.96*se)), 4),
                           p=("P<0.001" if mm.pvalues["fi10"] < 0.001 else f"{mm.pvalues['fi10']:.4g}"))
        print(f"[MIMIC {lab}] n={ladder[lab]['n']} OR={ladder[lab]['OR']} "
              f"({ladder[lab]['lo']}-{ladder[lab]['hi']})", flush=True)
    res["mimic_ladder"] = ladder

    # ---- 11 项派生系数 ----
    d11, m11, s11 = fit_m1(data["mimiciv"], "fi_fixed")
    res["fixed_derivation"] = s11
    print("[fixed deriv]", s11, flush=True)

    # ---- 30 项派生系数（同口径复核）----
    d30, m30, s30 = fit_m1(data["mimiciv"], "fi_lab")
    res["full_derivation"] = s30

    # ---- A. 各自可评估队列（原口径）----
    res["separate"] = {}
    for db in data:
        d = data[db]
        dd, mm, s = fit_m1(d, "fi_fixed")
        c = s11["coef"]
        p = 1/(1+np.exp(-(c["Intercept"] + c["fi10"]*dd["fi_fixed"]*10 + c["age"]*dd["age"] + c["male"]*dd["male"])))
        res["separate"][db] = dict(local_fit=s, ext=metrics(dd["died"], p, dd.get("subject_id")))
        print(f"[sep {db}] ext {res['separate'][db]['ext']}", flush=True)

    # ---- B. 共同样本（同时满足 30 项 >=15 与 11 项 >=9）----
    res["common"] = {}
    for db in data:
        d = data[db].dropna(subset=["fi_lab", "fi_fixed", "died", "age", "male"]).copy()
        dd30, mm30, s30db = fit_m1(d, "fi_lab")
        dd11, mm11, s11db = fit_m1(d, "fi_fixed")
        # 冻结：各自用 MIMIC 派生系数
        out = {"n": int(len(d)), "deaths": int(d["died"].sum()), "local_or": {
                    "full": s30db, "fixed": s11db}}
        for tag, coef, col in [("full", s30["coef"], "fi_lab"), ("fixed", s11["coef"], "fi_fixed")]:
            p = 1/(1+np.exp(-(coef["Intercept"] + coef["fi10"]*d[col]*10 + coef["age"]*d["age"] + coef["male"]*d["male"])))
            out[f"ext_{tag}"] = metrics(d["died"], p, d.get("subject_id"))
        # 同一样本内 AUC 差值（bootstrap 配对）
        p30 = 1/(1+np.exp(-(s30["coef"]["Intercept"] + s30["coef"]["fi10"]*d["fi_lab"]*10 + s30["coef"]["age"]*d["age"] + s30["coef"]["male"]*d["male"])))
        p11 = 1/(1+np.exp(-(s11["coef"]["Intercept"] + s11["coef"]["fi10"]*d["fi_fixed"]*10 + s11["coef"]["age"]*d["age"] + s11["coef"]["male"]*d["male"])))
        rng = np.random.default_rng(7); y = d["died"].values; diffs = []
        g = d["subject_id"].values if "subject_id" in d.columns else np.arange(len(d))
        ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
        for _ in range(1000):
            pick = rng.choice(ug, size=len(ug), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            if y[ii].min() == y[ii].max(): continue
            diffs.append(roc_auc_score(y[ii], p30[ii]) - roc_auc_score(y[ii], p11[ii]))
        out["dAUC_full_minus_fixed"] = round(float(roc_auc_score(y, p30) - roc_auc_score(y, p11)), 4)
        out["dAUC_lo"] = round(float(np.nanpercentile(diffs, 2.5)), 4)
        out["dAUC_hi"] = round(float(np.nanpercentile(diffs, 97.5)), 4)
        res["common"][db] = out
        print(f"[common {db}] n={out['n']} dAUC={out['dAUC_full_minus_fixed']} "
              f"({out['dAUC_lo']}–{out['dAUC_hi']})", flush=True)

    # ---- 30 项 vs 11 项 在同一样本的相关 ----
    res["corr_common"] = {}
    for db in data:
        d = data[db].dropna(subset=["fi_lab", "fi_fixed"])
        res["corr_common"][db] = dict(n=int(len(d)),
            pearson=round(float(d[["fi_lab", "fi_fixed"]].corr().iloc[0, 1]), 4),
            spearman=round(float(d[["fi_lab", "fi_fixed"]].corr(method="spearman").iloc[0, 1]), 4))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nWROTE", OUT)

if __name__ == "__main__":
    main()
