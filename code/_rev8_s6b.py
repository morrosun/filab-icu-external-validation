# -*- coding: utf-8 -*-
"""第八轮：S6B（精简 26 变量）NWICU 行按最终结局标签重算。"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ANA = "analysis"

# MIMIC 精简指数开发模型（冻结）
mm = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
mr = pd.read_csv("filab/filab_reduced_mimiciv.csv")
mr["stay_id"] = mr["stay_id"].astype(str)
mm["stay_id"] = mm["stay_id"].astype(str)
mm = mm.merge(mr[["stay_id", "fi_lab_reduced"]], on="stay_id", how="left")
md = mm.dropna(subset=["fi_lab_reduced", "died", "age", "male"]).copy()
md["fi10r"] = md["fi_lab_reduced"] * 10
mod = smf.logit("died ~ fi10r + age + male", data=md).fit(disp=0)
coef = {k: float(v) for k, v in mod.params.items()}
se = {k: float(v) for k, v in mod.bse.items()}
print("[coef reduced]", {k: round(v, 4) for k, v in coef.items()}, "n=", int(mod.nobs))

# NWICU 精简指数（最终标签）
a = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
a["stay_id"] = a["stay_id"].astype(str)
a = a.dropna(subset=["fi_lab"]).copy()
nr = pd.read_csv("filab/filab_reduced_nwicu.csv")
nr["stay_id"] = nr["stay_id"].astype(str)
lab = pd.read_csv(f"{ANA}/db_audit/nwicu_final_labels.csv")
lab["stay_id"] = lab["stay_id"].astype(str)
a = a.merge(nr[["stay_id", "fi_lab_reduced"]], on="stay_id", how="left")
a = a.merge(lab[["stay_id", "died_clean"]], on="stay_id", how="left")
a["fi10r"] = a["fi_lab_reduced"] * 10
d = a.dropna(subset=["fi_lab_reduced", "died_clean", "age", "male"]).copy()
print("[nwicu reduced] n=", len(d), "deaths=", int(d.died_clean.sum()))

# 冻结原（30 项）模型与精简模型
m30 = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
m30["fi10"] = m30["fi_lab"] * 10
c30 = {k: float(v) for k, v in smf.logit("died ~ fi10 + age + male", data=m30).fit(disp=0).params.items()}

d["p_orig"] = 1 / (1 + np.exp(-(c30["Intercept"] + c30["fi10"] * d["fi_lab"] * 10
                                + c30["age"] * d["age"] + c30["male"] * d["male"])))
d["p_red"] = 1 / (1 + np.exp(-(coef["Intercept"] + coef["fi10r"] * d["fi10r"]
                               + coef["age"] * d["age"] + coef["male"] * d["male"])))
y = d["died_clean"].values.astype(float)
g = d["subject_id"].values

def metrics(p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    cs = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    cit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    obs, exp = y.sum(), p.sum()
    return dict(auc=round(float(roc_auc_score(y, p)), 4), brier=round(float(brier_score_loss(y, p)), 4),
                oe=round(float(obs / exp), 3),
                oe_lo=round(float(obs / exp * np.exp(-1.96 / np.sqrt(obs))), 3),
                oe_hi=round(float(obs / exp * np.exp(1.96 / np.sqrt(obs))), 3),
                slope=round(float(cs.params[1]), 3),
                citl=round(float(cit.params[0]), 3),
                citl_se=round(float(cit.bse[0]), 3))

res = {"n": int(len(d)), "deaths": int(y.sum())}
for tag, p in [("orig", d["p_orig"].values), ("red", d["p_red"].values)]:
    r = metrics(p)
    # bootstrap CI（患者层面）
    rng = np.random.default_rng(31)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    A, S = [], []
    for _ in range(800):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        yy, pp = y[ii], p[ii]
        if yy.min() == yy.max():
            continue
        A.append(roc_auc_score(yy, pp))
        lp = np.log(np.clip(pp, 1e-9, 1 - 1e-9) / (1 - np.clip(pp, 1e-9, 1 - 1e-9)))
        try:
            S.append(sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1])
        except Exception:
            pass
    r["auc_lo"], r["auc_hi"] = round(float(np.nanpercentile(A, 2.5)), 3), round(float(np.nanpercentile(A, 97.5)), 3)
    r["slope_lo"], r["slope_hi"] = round(float(np.nanpercentile(S, 2.5)), 3), round(float(np.nanpercentile(S, 97.5)), 3)
    r["citl_lo"] = round(r["citl"] - 1.96 * r["citl_se"], 3)
    r["citl_hi"] = round(r["citl"] + 1.96 * r["citl_se"], 3)
    res[tag] = r
    print(f"[{tag}]", r)

json.dump(res, open(f"{ANA}/db_audit/rev8_s6b_nwicu.json", "w"), indent=2)
print("WROTE rev8_s6b_nwicu.json")
