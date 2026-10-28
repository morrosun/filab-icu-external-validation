# -*- coding: utf-8 -*-
"""Table 2 权威数值（标准无年龄 Charlson）+ 增量指标；轻量（AUC bootstrap 仅 MIMIC）"""
import json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats as st

rng = np.random.default_rng(20261025)
a = pd.read_csv("analysis/analysis_mimiciv.csv"); a["stay_id"] = a["stay_id"].astype(str)
r = pd.read_csv("analysis/rev/rev_mimiciv.csv"); r["stay_id"] = r["stay_id"].astype(str)
d = a.merge(r[["stay_id", "sofa", "oasis", "charlson", "vent", "vaso", "rrt"]],
            on="stay_id", how="left", suffixes=("", "_rev"))
d["fi10"] = d["fi_lab"] * 10
dd = d.dropna(subset=["died", "fi10", "age", "male", "charlson", "sofa", "oasis", "vent", "vaso", "rrt"]).copy()
y = dd["died"].values
g = dd["subject_id"].values
ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}

def boot_auc(pr, B=400):
    out = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if y[ii].min() == y[ii].max():
            continue
        out.append(roc_auc_score(y[ii], pr[ii]))
    return round(float(np.nanpercentile(out, 2.5)), 3), round(float(np.nanpercentile(out, 97.5)), 3)

specs = [("M1", "died ~ fi10 + age + male"),
         ("M2", "died ~ fi10 + age + male + charlson"),
         ("M3", "died ~ fi10 + age + male + charlson + sofa + oasis"),
         ("M4", "died ~ fi10 + age + male + charlson + sofa + oasis + vent + vaso + rrt")]
tab = {}
for lab, f in specs:
    m = smf.logit(f, data=dd).fit(disp=0)
    b, se, pv = m.params["fi10"], m.bse["fi10"], m.pvalues["fi10"]
    pr = m.predict(dd).values
    lo, hi = boot_auc(pr)
    tab[lab] = dict(n=int(m.nobs), OR=round(float(np.exp(b)), 3),
                    lo=round(float(np.exp(b - 1.96 * se)), 3), hi=round(float(np.exp(b + 1.96 * se)), 3),
                    p=("<0.001" if pv < 0.001 else f"{pv:.3g}"),
                    auc=round(float(roc_auc_score(y, pr)), 3), auc_lo=lo, auc_hi=hi)
    print(lab, tab[lab], flush=True)

base = "age + male + charlson + sofa + oasis + vent + vaso + rrt"
mb = smf.logit(f"died ~ {base}", data=dd).fit(disp=0)
mf = smf.logit(f"died ~ {base} + fi10", data=dd).fit(disp=0)
pb, pf = mb.predict(dd).values, mf.predict(dd).values
lr = 2 * (mf.llf - mb.llf)
dA = []
for _ in range(400):
    pick = rng.choice(ug, size=len(ug), replace=True)
    ii = np.concatenate([idx[u] for u in pick])
    if y[ii].min() == y[ii].max():
        continue
    dA.append(roc_auc_score(y[ii], pf[ii]) - roc_auc_score(y[ii], pb[ii]))
inc = dict(baseline_auc=round(float(roc_auc_score(y, pb)), 3), full_auc=round(float(roc_auc_score(y, pf)), 3),
           dAUC=round(float(roc_auc_score(y, pf) - roc_auc_score(y, pb)), 4),
           dAUC_lo=round(float(np.nanpercentile(dA, 2.5)), 4), dAUC_hi=round(float(np.nanpercentile(dA, 97.5)), 4),
           lr_chi2=round(float(lr), 1), lr_p=f"{st.chi2.sf(lr, 1):.2e}",
           brier_base=round(float(brier_score_loss(y, pb)), 4), brier_full=round(float(brier_score_loss(y, pf)), 4),
           n=int(mf.nobs), fi_or=round(float(np.exp(mf.params["fi10"])), 3),
           fi_lo=round(float(np.exp(mf.params["fi10"] - 1.96 * mf.bse["fi10"])), 3),
           fi_hi=round(float(np.exp(mf.params["fi10"] + 1.96 * mf.bse["fi10"])), 3))
print("incremental", inc, flush=True)
json.dump({"table2": tab, "incremental": inc, "n": int(len(dd))},
          open("analysis/db_audit/table2_final.json", "w"), indent=2)
print("WROTE analysis/db_audit/table2_final.json")
