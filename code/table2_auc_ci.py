# -*- coding: utf-8 -*-
"""Bootstrap 95% CI for Table 2 (MIMIC-IV hierarchical M1-M5) AUCs.
Method matches existing calib_metrics(): fit on full derivation cohort,
obtain in-sample predictions p, then bootstrap-resample (y,p) pairs 2000x.
"""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")

ROOT = ROOT
ANA = os.path.join(ROOT, "analysis")
rev = pd.read_csv(os.path.join(ANA, "rev", "rev_mimiciv.csv")).dropna(subset=["fi_lab"]).copy()
rev["fi10"] = rev["fi_lab"] * 10
y = rev["died"].values
n = len(rev)
print("n =", n)

models = {
    "M1: age+sex": "died ~ fi10 + age + male",
    "M2: +ARD": "died ~ fi10 + age + male + ard",
    "M3: +Charlson(-rheum)": "died ~ fi10 + age + male + ard + cci_excl_rheum",
    "M4: +SOFA+OASIS": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis",
    "M5: +vent+vaso+rrt": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis + vent + vaso + rrt",
}

rng = np.random.default_rng(20240715)
results = []
for name, f in models.items():
    m = smf.logit(f, data=rev).fit(disp=0)
    p = np.asarray(m.predict(rev))
    auc = roc_auc_score(y, p)
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    results.append((name, auc, lo, hi))
    print(f"{name:28s} AUC={auc:.4f}  95% CI ({lo:.4f}-{hi:.4f})  ->  {auc:.3f} ({lo:.3f}-{hi:.3f})")

print("\nFormatted for Table 2:")
for name, auc, lo, hi in results:
    print(f"  {name}: {auc:.3f} ({lo:.3f}-{hi:.3f})")
