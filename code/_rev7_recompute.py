# -*- coding: utf-8 -*-
"""第七轮：以「日期层面清洁规则」为 NWICU 主结局，重算其全部结果。
主规则：dod 的日历日期 ∈ [入院日期, 出院日期]（NWICU 只有日期，故按日期比较）
敏感性：无下界（≤出院日期）、出院后 1 天宽限。
"""
import json
import os
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


PG = dict(**_DB_AUTH)
ANA = "analysis"
RES = {}

# ---------- 1. 取 intime，构造最终标签 ----------
f = pd.read_csv(f"{ANA}/db_audit/nwicu_dod_flags.csv")
f["stay_id"] = f["stay_id"].astype(str)
with psycopg2.connect(dbname="nwicu", **PG) as c:
    it = pd.read_sql_query("SELECT stay_id::text AS stay_id, intime, outtime FROM icu.icustays", c)
it["stay_id"] = it["stay_id"].astype(str)
for col in ["intime", "outtime"]:
    it[col] = pd.to_datetime(it[col], errors="coerce")
f = f.merge(it, on="stay_id", how="left")
for col in ["admittime", "dischtime", "dod"]:
    f[col] = pd.to_datetime(f[col], errors="coerce")

adm_d, dis_d, dod_d, int_d = (f[c].dt.normalize() for c in ["admittime", "dischtime", "dod", "intime"])
f["died_clean"] = (dod_d.notna() & (dod_d >= adm_d) & (dod_d <= dis_d)).astype(float)
f["died_nolower"] = (dod_d.notna() & (dod_d <= dis_d)).astype(int)
f["died_nolower_1d"] = (dod_d.notna() & (dod_d <= dis_d + pd.Timedelta(days=1))).astype(int)
f["died_clean_1d"] = (dod_d.notna() & (dod_d >= adm_d) & (dod_d <= dis_d + pd.Timedelta(days=1))).astype(int)
f["dod_before_adm_date"] = (dod_d.notna() & (dod_d < adm_d)).astype(int)
# 30 天：以 ICU 入科日期起算，并要求死亡日期不早于入科日期
f["mort30_clean"] = (dod_d.notna() & (dod_d >= int_d) & (dod_d <= int_d + pd.Timedelta(days=30))).astype(int)
f["mort30_nolower"] = (dod_d.notna() & (dod_d <= int_d + pd.Timedelta(days=30))).astype(int)
# 死亡日期早于入院日的记录：结局设为不确定（NaN），从该结局分析中剔除，而非计为存活
_bad = dod_d.notna() & (dod_d < adm_d)
f.loc[_bad, "died_clean"] = np.nan
f.loc[_bad, "mort30_clean"] = np.nan
print("[uncertain]", int(_bad.sum()), "records set to unknown outcome", flush=True)

RES["label_counts"] = {k: int(f[k].sum()) for k in
                       ["died_clean", "died_nolower", "died_nolower_1d", "died_clean_1d",
                        "dod_before_adm_date", "mort30_clean", "mort30_nolower"]}
print("[labels]", RES["label_counts"], flush=True)

# ---------- 2. 合并分析数据 ----------
a = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
a["stay_id"] = a["stay_id"].astype(str)
a = a.merge(f[["stay_id", "died_clean", "died_nolower", "mort30_clean"]], on="stay_id", how="left")
a = a.dropna(subset=["fi_lab"]).copy()
a["fi10"] = a["fi_lab"] * 10
nwc = a.copy()
print("[nwicu] n=%d  deaths(clean)=%d  deaths(nolower)=%d" %
      (len(nwc), int(nwc.died_clean.sum()), int(nwc.died_nolower.sum())), flush=True)


def logit_cluster(df, formula, y):
    d = df.dropna(subset=[c for c in ["fi10", "age", "male", y]]).copy()
    m = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
    b, se = m.params["fi10"], m.bse["fi10"]
    return dict(n=int(len(d)), deaths=int(d[y].sum()),
                OR=round(float(np.exp(b)), 3),
                lo=round(float(np.exp(b - 1.96 * se)), 3),
                hi=round(float(np.exp(b + 1.96 * se)), 3),
                p=float(m.pvalues["fi10"]))


# ---------- 3. MIMIC 冻结系数（30 项） ----------
mm = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
mm["fi10"] = mm["fi_lab"] * 10
mder = smf.logit("died ~ fi10 + age + male", data=mm).fit(disp=0)
coef30 = {k: float(v) for k, v in mder.params.items()}
RES["coef30_mimic"] = {k: round(v, 4) for k, v in coef30.items()}
RES["coef30_se"] = {k: round(float(v), 4) for k, v in mder.bse.items()}
print("[coef30]", RES["coef30_mimic"], flush=True)


def calib(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    oe = y.sum() / p.sum()
    obs = y.sum()
    lo = oe * np.exp(-1.96 / np.sqrt(obs)); hi = oe * np.exp(1.96 / np.sqrt(obs))
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    cs = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                  offset=lp).fit()
    return dict(n=int(len(y)), deaths=int(obs), auc=round(float(auc), 4), brier=round(float(brier), 4),
                oe=round(float(oe), 3), oe_lo=round(float(lo), 3), oe_hi=round(float(hi), 3),
                slope=round(float(cs.params[1]), 3), citl=round(float(citl.params[0]), 3),
                citl_se=round(float(citl.bse[0]), 3))


def boot_ci(y, p, groups, nboot=1000, seed=11):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx = {u: np.where(groups == u)[0] for u in ug}
    a_, s_ = [], []
    for _ in range(nboot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        yy, pp = y[ii], p[ii]
        if yy.min() == yy.max():
            continue
        a_.append(roc_auc_score(yy, pp))
        lp = np.log(np.clip(pp, 1e-9, 1 - 1e-9) / (1 - np.clip(pp, 1e-9, 1 - 1e-9)))
        try:
            s_.append(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1])
        except Exception:
            pass
    q = lambda v: (round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3))
    return q(a_), q(s_)


# ---------- 4. NWICU 外部验证（清洁 vs 无下界） ----------
nwc["lp"] = (coef30["Intercept"] + coef30["fi10"] * nwc["fi10"]
             + coef30["age"] * nwc["age"] + coef30["male"] * nwc["male"])
nwc["p"] = 1 / (1 + np.exp(-nwc["lp"]))
for lab, ycol in [("clean", "died_clean"), ("nolower", "died_nolower"), ("nolower_1d", None)]:
    if ycol is None:
        d = f.set_index("stay_id").loc[nwc["stay_id"], "died_nolower_1d"].values
        y = np.asarray(d, float)
        ok = ~np.isnan(y)
    else:
        y = nwc[ycol].values.astype(float); ok = ~np.isnan(y)
    r = calib(y[ok], nwc["p"].values[ok])
    ac, sc = boot_ci(y[ok], nwc["p"].values[ok], nwc["subject_id"].values[ok])
    r["auc_lo"], r["auc_hi"] = ac
    r["slope_lo"], r["slope_hi"] = sc
    r["citl_lo"] = round(r["citl"] - 1.96 * r["citl_se"], 3)
    r["citl_hi"] = round(r["citl"] + 1.96 * r["citl_se"], 3)
    RES[f"ext_nwicu_{lab}"] = r
    print(f"[ext {lab}]", r, flush=True)

# ---------- 5. NWICU M1 OR ----------
RES["m1_nwicu_clean"] = logit_cluster(nwc, "died_clean ~ fi10 + age + male", "died_clean")
RES["m1_nwicu_nolower"] = logit_cluster(nwc, "died_nolower ~ fi10 + age + male", "died_nolower")
print("[m1 clean]", RES["m1_nwicu_clean"], flush=True)

# ---------- 6. 次要结局（清洁标签） ----------
o = pd.read_csv(f"{ANA}/outcomes_nwicu.csv")
o["stay_id"] = o["stay_id"].astype(str)
o["stay_id"] = o["stay_id"].astype(str)
o = o.merge(f[["stay_id", "died_clean", "mort30_clean"]], on="stay_id", how="left")
o["fi10"] = o["fi_lab"] * 10
print("[outcomes] rows=%d  in-cohort=%d" % (len(o), int(o.fi10.notna().sum())), flush=True)
sec = {}
for lab, ycol, extra in [("inhosp", "died_clean", None), ("mort30", "mort30_clean", None),
                         ("readmit", "readmit30", "alive")]:
    d = o.dropna(subset=[ycol, "fi10", "age", "male"]).copy()
    if extra == "alive":
        d = d[d["died_clean"] == 0].copy()
    m = smf.glm(f"{ycol} ~ fi10 + age + male", data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject_id"].values})
    b, se = m.params["fi10"], m.bse["fi10"]
    sec[lab] = dict(n=int(len(d)), events=int(d[ycol].sum()),
                    rate=round(float(d[ycol].mean() * 100), 1),
                    OR=round(float(np.exp(b)), 3),
                    lo=round(float(np.exp(b - 1.96 * se)), 3),
                    hi=round(float(np.exp(b + 1.96 * se)), 3),
                    auc=round(float(roc_auc_score(d[ycol], m.predict(d))), 3))
    print(f"[sec {lab}]", sec[lab], flush=True)
RES["secondary_nwicu_clean"] = sec

# ---------- 7. S7 面板 B：NWICU 共同样本（30 vs 14，清洁标签） ----------
f14 = pd.read_csv(f"{ANA}/db_audit/fixed_fi14_nwicu.csv")
f14["stay_id"] = f14["stay_id"].astype(str)
c = nwc.merge(f14[["stay_id", "fi14"]], on="stay_id", how="left")
dc = c.dropna(subset=["fi_lab", "fi14", "died_clean"]).copy()
P14 = json.load(open(f"{ANA}/db_audit/panel14.json", encoding="utf-8"))
c14 = P14["coef14"]
dc["p30"] = 1 / (1 + np.exp(-(coef30["Intercept"] + coef30["fi10"] * dc["fi_lab"] * 10
                              + coef30["age"] * dc["age"] + coef30["male"] * dc["male"])))
dc["p14"] = 1 / (1 + np.exp(-(c14["Intercept"] + c14["fi10"] * dc["fi14"] * 10
                              + c14["age"] * dc["age"] + c14["male"] * dc["male"])))
y = dc["died_clean"].values.astype(float)
r30 = calib(y, dc["p30"].values); r14 = calib(y, dc["p14"].values)
b30, b14 = boot_ci(y, dc["p30"].values, dc["subject_id"].values, seed=13)
c30, c14b = boot_ci(y, dc["p14"].values, dc["subject_id"].values, seed=17)
d_auc = r30["auc"] - r14["auc"]
rng = np.random.default_rng(19)
ug = np.unique(dc["subject_id"].values); idx = {u: np.where(dc["subject_id"].values == u)[0] for u in ug}
dd = []
for _ in range(600):
    pick = rng.choice(ug, size=len(ug), replace=True)
    ii = np.concatenate([idx[u] for u in pick])
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    dd.append(roc_auc_score(yy, dc["p30"].values[ii]) - roc_auc_score(yy, dc["p14"].values[ii]))
RES["s7_nwicu_clean"] = dict(n=len(dc), deaths=int(y.sum()),
                             ext30=dict(r30, auc_lo=b30[0], auc_hi=b30[1], slope_lo=c30[0], slope_hi=c30[1]),
                             ext14=dict(r14, auc_lo=b14[0], auc_hi=b14[1], slope_lo=c14b[0], slope_hi=c14b[1]),
                             dAUC=round(float(d_auc), 4),
                             dAUC_lo=round(float(np.nanpercentile(dd, 2.5)), 4),
                             dAUC_hi=round(float(np.nanpercentile(dd, 97.5)), 4))
print("[s7 nwicu clean]", RES["s7_nwicu_clean"], flush=True)

# ---------- 8. 敏感性：无下界与 +1 天下的外验 ----------
json.dump(RES, open(f"{ANA}/db_audit/rev7_nwicu.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
f[["stay_id", "died_clean", "died_nolower", "died_clean_1d", "died_nolower_1d",
   "mort30_clean", "dod_before_adm_date"]].to_csv(f"{ANA}/db_audit/nwicu_final_labels.csv", index=False)
print("WROTE rev7_nwicu.json + nwicu_final_labels.csv")
