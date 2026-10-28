# -*- coding: utf-8 -*-
"""
敏感性分析（审稿 Issue 3）：用「排除 >70% 缺失变量后的 FI-Lab」重跑核心模型
  A) MIMIC-IV 分层模型 M1-M5（与 revision_analysis.py 完全同口径，仅换 fi_lab）
  B) 冻结系数外部验证（MIMIC 重新推导 -> 三库应用）
  C) ARD vs 非ARD 亚组性能
  D) 精简 FI-Lab 的描述性汇总（中位、已评估中位、死亡梯度）
输出：analysis/revision/sens_lowmissing_*.csv
"""
import os, warnings
import numpy as np, pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")

ROOT = ROOT
ANA = os.path.join(ROOT, "analysis")
OUT = os.path.join(ANA, "revision")
os.makedirs(OUT, exist_ok=True)

def load_reduced(db):
    base = pd.read_csv(f"{ANA}/analysis_{db}.csv")
    red = pd.read_csv(f"{ROOT}/filab/filab_reduced_{db}.csv")
    base = base.merge(red[["stay_id", "fi_lab_reduced", "n_assessed_reduced"]], on="stay_id", how="left")
    base["fi_lab"] = base["fi_lab_reduced"]
    base["n_assessed"] = base["n_assessed_reduced"]
    return base.dropna(subset=["fi_lab"]).copy()

mim = load_reduced("mimiciv")
nwi = load_reduced("nwicu")
eic = load_reduced("eicu")
print(f"loaded reduced: MIMIC {len(mim):,}, NWICU {len(nwi):,}, eICU {len(eic):,}")

# ---------------- A) MIMIC 分层模型 M1-M5 ----------------
rev = pd.read_csv(f"{OUT}/../rev/rev_mimiciv.csv") if os.path.exists(f"{OUT}/../rev/rev_mimiciv.csv") else pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv")
red_map = pd.read_csv(f"{ROOT}/filab/filab_reduced_mimiciv.csv").set_index("stay_id")["fi_lab_reduced"]
rev["fi_lab"] = rev["stay_id"].map(red_map)
rev = rev.dropna(subset=["fi_lab"]).copy()
mim_cc = rev.dropna(subset=["sofa", "oasis", "sapsii", "cci_excl_rheum"]).copy()
mim_cc["fi10"] = mim_cc["fi_lab"] * 10
print(f"MIMIC complete-case w/ severity (reduced FI): {len(mim_cc):,} / {len(rev):,}")

def fit_or(df, formula):
    m = smf.logit(formula, data=df).fit(disp=0)
    b = m.params.get("fi10", np.nan); se = m.bse.get("fi10", np.nan); p = m.pvalues.get("fi10", np.nan)
    orr = np.exp(b); lo = np.exp(b - 1.96 * se); hi = np.exp(b + 1.96 * se)
    auc = roc_auc_score(df["died"], m.predict(df))
    return dict(OR=orr, lo=lo, hi=hi, p=p, auc=auc, llf=m.llf)

models = {
    "M1: age+sex": "died ~ fi10 + age + male",
    "M2: +ARD": "died ~ fi10 + age + male + ard",
    "M3: +Charlson(-rheum)": "died ~ fi10 + age + male + ard + cci_excl_rheum",
    "M4: +SOFA+OASIS": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis",
    "M5: +vent+vaso+rrt": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis + vent + vaso + rrt",
}
rows = []
for name, f in models.items():
    r = fit_or(mim_cc, f)
    rows.append(dict(model=name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
hier = pd.DataFrame(rows)
hier.to_csv(f"{OUT}/sens_lowmissing_hierarchical.csv", index=False)
print("\n=== A) MIMIC 分层模型（精简 FI-Lab, 每0.1 OR）===")
print(hier[["model", "OR", "lo", "hi", "p", "auc"]].to_string(index=False))

# ---------------- B) 冻结系数外部验证（精简 FI） ----------------
for d in (mim, nwi, eic):
    d["fi10"] = d["fi_lab"] * 10
derive_f = "died ~ age + male + ard + fi10"
m_der = smf.logit(derive_f, data=mim).fit(disp=0)
coef = m_der.params
print("\n推导系数（精简 FI, MIMIC）：\n", coef.round(4).to_string())

def ci95(x, lo, hi):
    return f"{x:.3f} ({lo:.3f}–{hi:.3f})"

def calib_metrics(df):
    X = df.copy()
    lp = (coef["Intercept"] + coef["age"] * X["age"] + coef["male"] * X["male"]
          + coef["ard"] * X["ard"] + coef["fi10"] * X["fi10"])
    p = np.asarray(1 / (1 + np.exp(-lp)))
    y = X["died"].values
    auc = roc_auc_score(y, p); brier = brier_score_loss(y, p); oe = y.mean() / p.mean()
    # AUC bootstrap 95% CI
    rng = np.random.default_rng(20240715)
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(y[idx])):
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    # O:E Poisson 95% CI
    O = y.sum(); oe_lo = np.exp(np.log(oe) - 1.96 / np.sqrt(O)); oe_hi = np.exp(np.log(oe) + 1.96 / np.sqrt(O))
    # CITL = intercept of offset model logit(y) = alpha + offset(lp) (fixed slope=1)
    Xo = np.ones((len(y), 1))
    cm_o = sm.Logit(y, Xo, offset=lp.values).fit(disp=0)
    citl = cm_o.params[0]; citl_lo = citl - 1.96 * cm_o.bse[0]; citl_hi = citl + 1.96 * cm_o.bse[0]
    # Slope = free-slope regression logit(y) = alpha + beta*lp
    Xc = sm.add_constant(lp.values); cm = sm.Logit(y, Xc).fit(disp=0)
    slope = cm.params[1]; slope_lo = slope - 1.96 * cm.bse[1]; slope_hi = slope + 1.96 * cm.bse[1]
    return dict(n=len(y), deaths=int(O),
                auc=round(auc, 3), auc_ci=ci95(auc, auc_lo, auc_hi),
                brier=round(brier, 3),
                oe=round(oe, 3), oe_ci=ci95(oe, oe_lo, oe_hi),
                citl=round(citl, 3), citl_ci=ci95(citl, citl_lo, citl_hi),
                slope=round(slope, 3), slope_ci=ci95(slope, slope_lo, slope_hi))

ext = []
for name, df in [("MIMIC-IV (apparent)", mim), ("NWICU (frozen external)", nwi), ("eICU (frozen external)", eic)]:
    ext.append(dict(cohort=name, **calib_metrics(df)))
ext = pd.DataFrame(ext); ext.to_csv(f"{OUT}/sens_lowmissing_external.csv", index=False)
print("\n=== B) 冻结系数外部验证（精简 FI）===")
print(ext.to_string(index=False))

# ---------------- C) ARD vs 非ARD 亚组 ----------------
sub = []
for name, df in [("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]:
    for grp, mask in [("ARD", df["ard"] == 1), ("non-ARD", df["ard"] == 0)]:
        d = df[mask]
        if d["died"].sum() >= 5 and len(d) >= 30:
            try:
                sub.append(dict(cohort=name, group=grp, **calib_metrics(d)))
            except Exception as e:
                sub.append(dict(cohort=name, group=grp, n=len(d), deaths=int(d["died"].sum()), note=str(e)[:40]))
        else:
            sub.append(dict(cohort=name, group=grp, n=len(d), deaths=int(d["died"].sum()), note="too few"))
subp = pd.DataFrame(sub); subp.to_csv(f"{OUT}/sens_lowmissing_ard.csv", index=False)
print("\n=== C) ARD vs 非ARD 亚组（精简 FI）===")
print(subp.to_string(index=False))

# ---------------- D) 精简 FI 描述性汇总 ----------------
summ = []
for db, lab in [("MIMIC-IV", "MIMIC"), ("NWICU", "NWICU"), ("eICU", "eICU")]:
    src = {"MIMIC-IV": mim, "NWICU": nwi, "eICU": eic}[db]
    d = src.dropna(subset=["fi_lab"]).copy()
    d["q"] = pd.qcut(d["fi_lab"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grad = d.groupby("q", observed=True).agg(n=("died", "size"), deaths=("died", "sum"))
    grad["mort"] = (grad["deaths"] / grad["n"] * 100).round(1)
    summ.append(dict(database=db, n_fi=len(d), fi_median=round(d["fi_lab"].median(), 3),
                     n_assessed_median=int(d["n_assessed"].median()),
                     q1_mort=grad.loc["Q1", "mort"], q4_mort=grad.loc["Q4", "mort"],
                     corr=round(d["fi_lab"].corr(d["died"]), 3)))
summp = pd.DataFrame(summ); summp.to_csv(f"{OUT}/sens_lowmissing_summary.csv", index=False)
print("\n=== D) 精简 FI 描述性汇总 ===")
print(summp.to_string(index=False))

print("\nAll sensitivity (low-missingness) outputs saved to", OUT)
