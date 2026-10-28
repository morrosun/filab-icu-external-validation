# -*- coding: utf-8 -*-
"""Rev4-B (fast): 共同样本 30 vs 11 项对比 + 斜率/CITL 患者层面 bootstrap CI
+ 标准无年龄 Charlson 的 M1-M4 权威数值与增量指标
输出 analysis/db_audit/rev4b.json
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
import psycopg2

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


OUT = "analysis/db_audit/rev4b.json"
CONN = dict(**_DB_AUTH)
BOOT = 400
rng = np.random.default_rng(20261025)

def _logit2(y, x, iters=40):
    """2 参数 logistic IRLS -> (intercept, slope)"""
    X = np.column_stack([np.ones_like(x), x]); b = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-12, None)
        H = X.T @ (X * w[:, None]); g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return float(b[0]), float(b[1])

def logitp(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def load(db):
    a = pd.read_csv(f"analysis/analysis_{db}.csv"); a["stay_id"] = a["stay_id"].astype(str)
    f = pd.read_csv(f"analysis/db_audit/fixed_fi_{db}.csv")[["stay_id", "fi_fixed", "n_fixed"]]
    f["stay_id"] = f["stay_id"].astype(str)
    d = a.merge(f, on="stay_id", how="left")
    if db == "eicu":
        d = d.drop(columns=[c for c in ["subject_id"] if c in d.columns])
        with psycopg2.connect(**CONN, dbname="eicu") as c:
            pid = pd.read_sql_query(
                "SELECT patientunitstayid::text AS stay_id, uniquepid FROM eicu_crd.patient", c)
        d = d.merge(pid, on="stay_id", how="left").rename(columns={"uniquepid": "subject_id"})
    return d

def fitm(d, col):
    dd = d.dropna(subset=[col, "died", "age", "male"]).copy()
    dd["fi10"] = dd[col] * 10
    m = smf.logit("died ~ fi10 + age + male", data=dd).fit(disp=0)
    b, se = m.params["fi10"], m.bse["fi10"]
    return dd, dict(coef={k: float(v) for k, v in m.params.items()}, OR=float(np.exp(b)),
                    lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                    p=float(m.pvalues["fi10"]), n=int(m.nobs))

def pred(d, coef, col):
    lp = (coef["Intercept"] + coef["fi10"] * d[col].values * 10
          + coef["age"] * d["age"].values + coef["male"] * d["male"].values)
    return 1 / (1 + np.exp(-lp))

def boot_idx(g):
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    return ug, idx

def perf(y, p, g):
    y = np.asarray(y, float); p = np.asarray(p, float); g = np.asarray(g)
    lp = logitp(p)
    _, slope = _logit2(y, lp)
    gl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    citl, citl_se = float(gl.params[0]), float(gl.bse[0])
    obs, exp = y.sum(), p.sum(); oe = obs / exp
    ug, idx = boot_idx(g)
    bA, bS, bO = [], [], []
    for _ in range(BOOT):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        yy, pp = y[ii], p[ii]
        if yy.min() == yy.max():
            continue
        bA.append(roc_auc_score(yy, pp)); bO.append(yy.sum() / pp.sum())
        bS.append(_logit2(yy, logitp(pp))[1])
    q = lambda a, t: round(float(np.nanpercentile(a, t)), 4) if len(a) else None
    return dict(n=int(len(y)), deaths=int(obs), auc=round(float(roc_auc_score(y, p)), 4),
                auc_lo=q(bA, 2.5), auc_hi=q(bA, 97.5), brier=round(float(brier_score_loss(y, p)), 4),
                oe=round(float(oe), 4), oe_lo=q(bO, 2.5), oe_hi=q(bO, 97.5),
                slope=round(slope, 4), slope_lo=q(bS, 2.5), slope_hi=q(bS, 97.5),
                citl=round(citl, 4), citl_lo=round(citl - 1.96 * citl_se, 4),
                citl_hi=round(citl + 1.96 * citl_se, 4))

def main():
    res = {}
    data = {db: load(db) for db in ["mimiciv", "nwicu", "eicu"]}
    _, f30 = fitm(data["mimiciv"], "fi_lab")
    _, f11 = fitm(data["mimiciv"], "fi_fixed")
    res["derivation_full"], res["derivation_fixed"] = f30, f11

    res["separate"] = {}
    for db, d in data.items():
        dd, _l = fitm(d, "fi_fixed")
        p = pred(dd, f11["coef"], "fi_fixed")
        g = dd["subject_id"].values if "subject_id" in dd.columns else np.arange(len(dd))
        res["separate"][db] = perf(dd["died"].values, p, g)
        print(f"[sep {db}] {res['separate'][db]}", flush=True)

    res["common"] = {}
    for db, d in data.items():
        dc = d.dropna(subset=["fi_lab", "fi_fixed", "died", "age", "male"]).copy()
        _, l30 = fitm(dc, "fi_lab")
        _, l11 = fitm(dc, "fi_fixed")
        p30 = pred(dc, f30["coef"], "fi_lab"); p11 = pred(dc, f11["coef"], "fi_fixed")
        y = dc["died"].values
        g = dc["subject_id"].values if "subject_id" in dc.columns else np.arange(len(dc))
        ug, idx = boot_idx(g)
        dA = []
        for _ in range(BOOT):
            pick = rng.choice(ug, size=len(ug), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            if y[ii].min() == y[ii].max():
                continue
            dA.append(roc_auc_score(y[ii], p30[ii]) - roc_auc_score(y[ii], p11[ii]))
        res["common"][db] = dict(
            n=int(len(dc)), deaths=int(y.sum()),
            local_full=dict(OR=round(l30["OR"], 3), lo=round(l30["lo"], 3), hi=round(l30["hi"], 3)),
            local_fixed=dict(OR=round(l11["OR"], 3), lo=round(l11["lo"], 3), hi=round(l11["hi"], 3)),
            ext_full=perf(y, p30, g), ext_fixed=perf(y, p11, g),
            dAUC=round(float(roc_auc_score(y, p30) - roc_auc_score(y, p11)), 4),
            dAUC_lo=round(float(np.nanpercentile(dA, 2.5)), 4),
            dAUC_hi=round(float(np.nanpercentile(dA, 97.5)), 4))
        c = res["common"][db]
        print(f"[common {db}] n={c['n']} local_full={c['local_full']} local_fixed={c['local_fixed']} "
              f"dAUC={c['dAUC']} ({c['dAUC_lo']}~{c['dAUC_hi']})", flush=True)
        print(f"   ext_full={c['ext_full']}", flush=True)
        print(f"   ext_fixed={c['ext_fixed']}", flush=True)

    res["corr"] = {}
    for db, d in data.items():
        dc = d.dropna(subset=["fi_lab", "fi_fixed"])
        res["corr"][db] = dict(n=int(len(dc)),
                               pearson=round(float(dc[["fi_lab", "fi_fixed"]].corr().iloc[0, 1]), 4),
                               spearman=round(float(dc[["fi_lab", "fi_fixed"]].corr(method="spearman").iloc[0, 1]), 4))

    # ---- 权威 Table 2：标准无年龄 Charlson ----
    a = data["mimiciv"]
    r = pd.read_csv("analysis/rev/rev_mimiciv.csv"); r["stay_id"] = r["stay_id"].astype(str)
    m0 = a.merge(r[["stay_id", "sofa", "oasis", "charlson", "vent", "vaso", "rrt"]], on="stay_id", how="left", suffixes=("", "_rev"))
    m0["fi10"] = m0["fi_lab"] * 10
    dd = m0.dropna(subset=["died", "fi10", "age", "male", "charlson", "sofa", "oasis", "vent", "vaso", "rrt"])
    g = dd["subject_id"].values if "subject_id" in dd.columns else np.arange(len(dd))
    ug, idx = boot_idx(g)
    specs = [("M1", "died ~ fi10 + age + male"),
             ("M2", "died ~ fi10 + age + male + charlson"),
             ("M3", "died ~ fi10 + age + male + charlson + sofa + oasis"),
             ("M4", "died ~ fi10 + age + male + charlson + sofa + oasis + vent + vaso + rrt")]
    tab = {}
    yv = dd["died"].values
    for lab, f in specs:
        mm = smf.logit(f, data=dd).fit(disp=0)
        b, se, pv = mm.params["fi10"], mm.bse["fi10"], mm.pvalues["fi10"]
        pr = mm.predict(dd).values
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(ug, size=len(ug), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            if yv[ii].min() == yv[ii].max():
                continue
            bs.append(roc_auc_score(yv[ii], pr[ii]))
        tab[lab] = dict(n=int(mm.nobs), OR=round(float(np.exp(b)), 3),
                        lo=round(float(np.exp(b - 1.96 * se)), 3), hi=round(float(np.exp(b + 1.96 * se)), 3),
                        p=("<0.001" if pv < 0.001 else f"{pv:.3g}"),
                        auc=round(float(roc_auc_score(yv, pr)), 3),
                        auc_lo=round(float(np.nanpercentile(bs, 2.5)), 3),
                        auc_hi=round(float(np.nanpercentile(bs, 97.5)), 3))
        print(f"[{lab}] {tab[lab]}", flush=True)
    res["table2"] = tab

    base = "age + male + charlson + sofa + oasis + vent + vaso + rrt"
    mb = smf.logit(f"died ~ {base}", data=dd).fit(disp=0)
    mf = smf.logit(f"died ~ {base} + fi10", data=dd).fit(disp=0)
    pb, pf = mb.predict(dd).values, mf.predict(dd).values
    lr = 2 * (mf.llf - mb.llf)
    from scipy import stats as st
    dA = []
    for _ in range(BOOT):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if yv[ii].min() == yv[ii].max():
            continue
        dA.append(roc_auc_score(yv[ii], pf[ii]) - roc_auc_score(yv[ii], pb[ii]))
    res["incremental"] = dict(
        baseline_auc=round(float(roc_auc_score(yv, pb)), 3), full_auc=round(float(roc_auc_score(yv, pf)), 3),
        dAUC=round(float(roc_auc_score(yv, pf) - roc_auc_score(yv, pb)), 4),
        dAUC_lo=round(float(np.nanpercentile(dA, 2.5)), 4),
        dAUC_hi=round(float(np.nanpercentile(dA, 97.5)), 4),
        lr_chi2=round(float(lr), 1), lr_p=f"{st.chi2.sf(lr, 1):.2e}",
        brier_base=round(float(brier_score_loss(yv, pb)), 4),
        brier_full=round(float(brier_score_loss(yv, pf)), 4),
        n=int(mf.nobs), fi_or=round(float(np.exp(mf.params["fi10"])), 3))
    print("[incremental]", res["incremental"], flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
