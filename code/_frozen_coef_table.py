"""导出冻结（DER）系数模型的完整参数表：β、SE、OR、95%CI（修改意见3 一.7）。
在 MIMIC-IV 表观队列上拟合 died ~ fi10 + age + male + ard，冻结全部系数（含截距）。
外部队列（NWICU/eICU）不重估任何参数。"""
import warnings, numpy as np, pandas as pd, statsmodels.api as sm
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")
ANA = os.path.join(ROOT, 'analysis')
OUT = os.path.join(ROOT, 'analysis', 'revision')

a = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab","died","age","male","ard"]).copy()
a["fi10"] = a["fi_lab"] * 10
y = a["died"].astype(float).values
X = sm.add_constant(a[["fi10", "age", "male", "ard"]].astype(float).values)  # const,fi10,age,male,ard
m = sm.Logit(y, X).fit(disp=0)

names = ["Intercept (截距)", "fi10 (FI-Lab 每0.1)", "age (年龄, 每岁)", "male (男性)", "ard (自身免疫风湿病)"]
rows = []
ci = m.conf_int()
for i, nm in enumerate(names):
    b = m.params[i]; se = m.bse[i]; lo, hi = ci[i]
    is_int = (i == 0)
    rows.append(dict(
        term=nm,
        beta=round(b, 4),
        se=round(se, 4),
        OR=("—" if is_int else round(np.exp(b), 3)),
        OR_lo=("—" if is_int else round(np.exp(lo), 3)),
        OR_hi=("—" if is_int else round(np.exp(hi), 3)),
        p=("<0.001" if m.pvalues[i] < 0.001 else round(m.pvalues[i], 3)),
    ))
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/frozen_coefficients_full.csv", index=False)
print("n =", len(a), "deaths =", int(y.sum()))
print(out.to_string(index=False))
print("DONE -> frozen_coefficients_full.csv")
