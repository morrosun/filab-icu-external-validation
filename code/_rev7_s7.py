# -*- coding: utf-8 -*-
"""第七轮补2：S7 中 NWICU 面板在清洁死亡标签下的共同样本比较（修正 CI 索引）。"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ANA = "analysis"
R = json.load(open(f"{ANA}/db_audit/rev7_nwicu.json", encoding="utf-8"))
P14 = json.load(open(f"{ANA}/db_audit/panel14.json", encoding="utf-8"))

lab = pd.read_csv(f"{ANA}/db_audit/nwicu_final_labels.csv")
lab["stay_id"] = lab["stay_id"].astype(str)
a = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
a["stay_id"] = a["stay_id"].astype(str)
a = a.dropna(subset=["fi_lab"]).copy().merge(lab[["stay_id", "died_clean"]], on="stay_id", how="left")
f14 = pd.read_csv(f"{ANA}/db_audit/fixed_fi14_nwicu.csv")
f14["stay_id"] = f14["stay_id"].astype(str)
c = a.merge(f14[["stay_id", "fi14"]], on="stay_id", how="left")
dc = c.dropna(subset=["fi_lab", "fi14", "died_clean", "age", "male"]).copy()

mm = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
mm["fi10"] = mm["fi_lab"] * 10
c30 = {k: float(v) for k, v in smf.logit("died ~ fi10 + age + male", data=mm).fit(disp=0).params.items()}
c14 = P14["coef14"]

dc["p30"] = 1 / (1 + np.exp(-(c30["Intercept"] + c30["fi10"] * dc["fi_lab"] * 10
                              + c30["age"] * dc["age"] + c30["male"] * dc["male"])))
dc["p14"] = 1 / (1 + np.exp(-(c14["Intercept"] + c14["fi10"] * dc["fi14"] * 10
                              + c14["age"] * dc["age"] + c14["male"] * dc["male"])))
y = dc["died_clean"].values.astype(float)
g = dc["subject_id"].values


def point(p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    cs = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    cit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    obs, exp = y.sum(), p.sum()
    return dict(n=int(len(y)), deaths=int(obs), auc=round(float(roc_auc_score(y, p)), 4),
                brier=round(float(brier_score_loss(y, p)), 4),
                oe=round(float(obs / exp), 3),
                oe_lo=round(float(obs / exp * np.exp(-1.96 / np.sqrt(obs))), 3),
                oe_hi=round(float(obs / exp * np.exp(1.96 / np.sqrt(obs))), 3),
                slope=round(float(cs.params[1]), 3), citl=round(float(cit.params[0]), 3))


def boot(p, nboot=1000, seed=23):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
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


res = {}
for tag, p in [("ext30", dc["p30"].values), ("ext14", dc["p14"].values)]:
    r = point(p)
    ac, sc = boot(p)
    r["auc_lo"], r["auc_hi"] = ac
    r["slope_lo"], r["slope_hi"] = sc
    res[tag] = r
    print(tag, r, flush=True)

rng = np.random.default_rng(29)
ug = np.unique(g)
idx = {u: np.where(g == u)[0] for u in ug}
dd = []
for _ in range(800):
    pick = rng.choice(ug, size=len(ug), replace=True)
    ii = np.concatenate([idx[u] for u in pick])
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    dd.append(roc_auc_score(yy, dc["p30"].values[ii]) - roc_auc_score(yy, dc["p14"].values[ii]))
res["dAUC"] = round(res["ext30"]["auc"] - res["ext14"]["auc"], 4)
res["dAUC_lo"] = round(float(np.nanpercentile(dd, 2.5)), 4)
res["dAUC_hi"] = round(float(np.nanpercentile(dd, 97.5)), 4)
print("dAUC", res["dAUC"], res["dAUC_lo"], res["dAUC_hi"], flush=True)

R["s7_nwicu_clean"] = res
json.dump(R, open(f"{ANA}/db_audit/rev7_nwicu.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("patched s7_nwicu_clean")
