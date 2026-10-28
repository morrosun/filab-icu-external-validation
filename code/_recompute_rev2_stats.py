# -*- coding: utf-8 -*-
"""
修订意见2 重算（无需数据库）：
 (P4) 校准指标重算
      - Calibration-in-the-large (校准截距): logit(y)=alpha + offset(lp), 固定斜率=1
      - Calibration slope (校准斜率): logit(y)=alpha + beta*lp (自由斜率)  + 95%CI
      - O:E 比 + 95%CI (bootstrap)
      - AUC + 95%CI (bootstrap)
 (P7) ARD 亚组补充
      - ARD 与非 ARD 各自的 FI-Lab OR(每0.1)
      - FI x ARD 交互项 P 值 (died ~ fi10+age+male+ard+fi10:ard)
      - ARD 亚组 AUC 95%CI (已含于校准块)
输出 analysis/revision/rev2_calibration.csv, rev2_ard_analysis.csv
"""
import os, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")
np.random.seed(42)

ANA = os.path.join(ROOT, 'analysis')
OUT = os.path.join(ROOT, 'analysis', 'revision')
os.makedirs(OUT, exist_ok=True)

mim = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
nwi = pd.read_csv(f"{ANA}/analysis_nwicu.csv").dropna(subset=["fi_lab"]).copy()
eic = pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
for d in (mim, nwi, eic):
    d["fi10"] = d["fi_lab"] * 10

# ---- 推导模型(冻结系数): MIMIC 全量 age+male+ard+fi10 ----
m_der = smf.logit("died ~ age + male + ard + fi10", data=mim).fit(disp=0)
coef = m_der.params

def lp_of(df):
    return (coef["Intercept"] + coef["age"]*df["age"] + coef["male"]*df["male"]
            + coef["ard"]*df["ard"] + coef["fi10"]*df["fi10"])

def auc_ci(y, p, B=1000):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    if y.sum() < 2 or len(np.unique(y)) < 2: return (np.nan, np.nan, np.nan)
    bs = []
    for _ in range(B):
        idx = np.random.randint(0, n, n)
        try: bs.append(roc_auc_score(y[idx], p[idx]))
        except Exception: pass
    if not bs: return (np.nan, np.nan, np.nan)
    return (roc_auc_score(y, p), np.percentile(bs, 2.5), np.percentile(bs, 97.5))

def oe_ci(y, p):
    """O:E 比 95%CI（Poisson 近似）：CI = exp{ log(O/E) ± 1.96/√O }，O=观察死亡数。
    此前用 bootstrap 仅在分母上重采样，分母均值方差极小导致 CI 过窄，已废弃。"""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    O = y.sum()
    base = y.mean() / p.mean()
    if O <= 0:
        return (base, np.nan, np.nan)
    se = 1.96 / np.sqrt(O)
    lo = np.exp(np.log(base) - se)
    hi = np.exp(np.log(base) + se)
    return (base, lo, hi)

def calib_block(name, df):
    y = df["died"].astype(int).values
    lp = lp_of(df).values
    p = 1/(1+np.exp(-lp))
    auc, a_lo, a_hi = auc_ci(y, p)
    brier = brier_score_loss(y, p)
    oe, oe_lo, oe_hi = oe_ci(y, p)
    # calibration-in-the-large: logit(y)=alpha + offset(lp)
    citl = smf.logit("died ~ 1", data=df, offset=pd.Series(lp, index=df.index)).fit(disp=0)
    citl_int = citl.params["Intercept"]
    citl_lo, citl_hi = citl.conf_int().loc["Intercept"]
    # calibration slope: logit(y)=alpha + beta*lp
    Xc = sm.add_constant(lp)
    cm = sm.Logit(y, Xc).fit(disp=0)
    cal_int = cm.params[0]  # const
    cal_slope = cm.params[1]
    ci = cm.conf_int()
    s_lo, s_hi = ci[1, 0], ci[1, 1]
    return dict(cohort=name, n=int(len(y)), deaths=int(y.sum()),
                auc=round(auc, 3), auc_lo=round(a_lo, 3), auc_hi=round(a_hi, 3),
                brier=round(brier, 3),
                oe=round(oe, 3), oe_lo=round(oe_lo, 3), oe_hi=round(oe_hi, 3),
                cal_int_large=round(citl_int, 3), citl_lo=round(citl_lo, 3), citl_hi=round(citl_hi, 3),
                cal_slope=round(cal_slope, 3), slope_lo=round(s_lo, 3), slope_hi=round(s_hi, 3))

print("推导系数:", dict(coef))

cal_rows = []
# 表5 冻结外验
cal_rows.append(calib_block("MIMIC-IV (表观)", mim))
cal_rows.append(calib_block("NWICU (冻结外部)", nwi))
cal_rows.append(calib_block("eICU (冻结外部)", eic))
# 表6 ARD 亚组
for name, df in [("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]:
    for grp, mask in [("ARD", df["ard"] == 1), ("非ARD", df["ard"] == 0)]:
        if df[mask]["died"].sum() >= 5 and mask.sum() >= 30:
            cal_rows.append(calib_block(f"{name} {grp}", df[mask]))
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(f"{OUT}/rev2_calibration.csv", index=False)
print("\n=== 校准重算 (P4) ===")
print(cal_df.to_string(index=False))

# ---------------- (P7) ARD 分析 ----------------
def fi_or(df, formula):
    m = smf.logit(formula, data=df).fit(disp=0)
    b = m.params["fi10"]; se = m.bse["fi10"]; p = m.pvalues["fi10"]
    return dict(OR=round(np.exp(b), 3), lo=round(np.exp(b-1.96*se), 3), hi=round(np.exp(b+1.96*se), 3), p=p)

ard_rows = []
for name, df in [("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]:
    ard = df[df["ard"] == 1]; non = df[df["ard"] == 0]
    r_a = fi_or(ard, "died ~ fi10 + age + male")
    r_n = fi_or(non, "died ~ fi10 + age + male")
    # 交互项
    mi = smf.logit("died ~ fi10 + age + male + ard + fi10:ard", data=df).fit(disp=0)
    pint = mi.pvalues.get("fi10:ard", np.nan)
    ard_rows.append(dict(db=name,
        ard_n=int(len(ard)), ard_OR=r_a["OR"], ard_lo=r_a["lo"], ard_hi=r_a["hi"], ard_p=r_a["p"],
        nonard_n=int(len(non)), nonard_OR=r_n["OR"], nonard_lo=r_n["lo"], nonard_hi=r_n["hi"], nonard_p=r_n["p"],
        interaction_p=pint))
ard_df = pd.DataFrame(ard_rows)
ard_df["interaction_p_fmt"] = ard_df["interaction_p"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
ard_df.to_csv(f"{OUT}/rev2_ard_analysis.csv", index=False)
print("\n=== ARD 亚组 FI OR 与交互 (P7) ===")
print(ard_df.to_string(index=False))

print("\nDONE -> rev2_calibration.csv, rev2_ard_analysis.csv")
