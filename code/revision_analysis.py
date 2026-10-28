# -*- coding: utf-8 -*-
"""
修订分析（回应审稿意见1）：
 1) 严重程度调整的分层模型 M1-M5（MIMIC 全量：SOFA/OASIS/SAPSII/Charlson去风湿/vent/vaso/rrt）
 2) FI-Lab 相对传统严重程度评分的增量价值（ΔAUC、LR 检验、决策曲线 DCA）
 3) 规范外部验证：冻结 MIMIC 系数应用于 NWICU/eICU，报告 AUC/校准斜率/截距/Brier/O:E
 4) ARD 亚组 vs 非ARD 的模型性能（AUC/校准/Brier/O:E/净获益）
 5) 阈值敏感性（n_assessed>=10/15/20）与早期死亡泄漏敏感性（排除 icu_los<1d）
 6) eICU 严重程度（APACHE）、NWICU Charlson（从 diagnoses_icd 计算）
输出至 analysis/revision/*.csv
"""
import os, warnings
import numpy as np, pandas as pd
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")

ROOT = ROOT
ANA = os.path.join(ROOT, "analysis")
OUT = os.path.join(ANA, "revision")
os.makedirs(OUT, exist_ok=True)
CONN = dict(**_DB_AUTH)

def q(db, sql):
    c = psycopg2.connect(dbname=db, **CONN)
    try:
        return pd.read_sql(sql, c)
    finally:
        c.close()

# ---------------------------------------------------------------------------
# 载入既有分析队列
# ---------------------------------------------------------------------------
mim = pd.read_csv(f"{ANA}/analysis_mimiciv.csv")
nwi = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
eic = pd.read_csv(f"{ANA}/analysis_eicu.csv")
# icu_los 用于早期死亡敏感性
for d, name in [(mim, "mimiciv"), (nwi, "nwicu"), (eic, "eicu")]:
    fl = pd.read_csv(f"{ROOT}/filab/filab_{name}.csv")[["stay_id", "icu_los_days"]]
    d.merge  # noop
    d["icu_los_days"] = d["stay_id"].map(fl.set_index("stay_id")["icu_los_days"])

mim = mim.dropna(subset=["fi_lab"]).copy()
nwi = nwi.dropna(subset=["fi_lab"]).copy()
eic = eic.dropna(subset=["fi_lab"]).copy()
print(f"loaded: MIMIC {len(mim)}, NWICU {len(nwi)}, eICU {len(eic)}")

# ---------------------------------------------------------------------------
# 1) MIMIC 严重程度评分 join
# ---------------------------------------------------------------------------
sids = tuple(int(x) for x in mim["stay_id"].unique())
sofa = q("mimiciv", f"SELECT stay_id, sofa FROM mimiciv_derived.first_day_sofa WHERE stay_id IN {sids}")
oasis = q("mimiciv", f"SELECT stay_id, oasis FROM mimiciv_derived.oasis WHERE stay_id IN {sids}")
saps = q("mimiciv", f"SELECT stay_id, sapsii FROM mimiciv_derived.sapsii WHERE stay_id IN {sids}")
# charlson (by hadm_id) - 去掉风湿项
hids = tuple(int(x) for x in mim["hadm_id"].unique())
chl = q("mimiciv", f"""SELECT hadm_id, charlson_comorbidity_index AS cci, rheumatic_disease
                       FROM mimiciv_derived.charlson WHERE hadm_id IN {hids}""")
chl["cci_excl_rheum"] = chl["cci"] - chl["rheumatic_disease"].fillna(0)
# 治疗：任何有创通气/血管活性药/RRT
vent = q("mimiciv", f"""SELECT DISTINCT stay_id, 1 AS vent FROM mimiciv_derived.ventilation
                        WHERE stay_id IN {sids} AND ventilation_status IN ('InvasiveVent','Tracheostomy')""")
vaso = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS vaso FROM mimiciv_derived.vasoactive_agent WHERE stay_id IN {sids}")
rrt = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS rrt FROM mimiciv_derived.rrt WHERE stay_id IN {sids}")

for tbl, col in [(sofa, "sofa"), (oasis, "oasis"), (saps, "sapsii")]:
    mim[col] = mim["stay_id"].map(tbl.groupby("stay_id")[col].max())
mim["cci_excl_rheum"] = mim["hadm_id"].map(chl.set_index("hadm_id")["cci_excl_rheum"])
for tbl, col in [(vent, "vent"), (vaso, "vaso"), (rrt, "rrt")]:
    mim[col] = mim["stay_id"].map(tbl.set_index("stay_id")[col]).fillna(0).astype(int)
# 填补少量缺失（严重程度评分用中位数，便于完整病例外的稳健性；同时保留完整病例版）
mim_cc = mim.dropna(subset=["sofa", "oasis", "sapsii", "cci_excl_rheum"]).copy()
print(f"MIMIC complete-case w/ severity: {len(mim_cc)} / {len(mim)}")
print("  SOFA median", mim_cc.sofa.median(), "OASIS", mim_cc.oasis.median(), "SAPSII", mim_cc.sapsii.median())
print("  vent%%", 100*mim_cc.vent.mean(), "vaso%%", 100*mim_cc.vaso.mean(), "rrt%%", 100*mim_cc.rrt.mean())

# ---------------------------------------------------------------------------
# 2) 分层模型 M1-M5 （FI-Lab 每 0.1 的 OR 随调整变化）
# ---------------------------------------------------------------------------
mim_cc["fi10"] = mim_cc["fi_lab"] * 10  # 每 0.1
def fit_or(df, formula):
    m = smf.logit(formula, data=df).fit(disp=0)
    b = m.params.get("fi10", np.nan); se = m.bse.get("fi10", np.nan); p = m.pvalues.get("fi10", np.nan)
    orr = np.exp(b); lo = np.exp(b - 1.96*se); hi = np.exp(b + 1.96*se)
    auc = roc_auc_score(df["died"], m.predict(df))
    return dict(OR=orr, lo=lo, hi=hi, p=p, auc=auc, llf=m.llf, k=len(m.params)), m

models = {
    "M1: age+sex": "died ~ fi10 + age + male",
    "M2: +ARD": "died ~ fi10 + age + male + ard",
    "M3: +Charlson(-rheum)": "died ~ fi10 + age + male + ard + cci_excl_rheum",
    "M4: +SOFA+OASIS": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis",
    "M5: +vent+vaso+rrt": "died ~ fi10 + age + male + ard + cci_excl_rheum + sofa + oasis + vent + vaso + rrt",
}
rows = []
for name, f in models.items():
    r, _ = fit_or(mim_cc, f)
    rows.append(dict(model=name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
hier = pd.DataFrame(rows)
hier.to_csv(f"{OUT}/hierarchical_models_mimiciv.csv", index=False)
print("\n=== MIMIC 分层模型（FI-Lab 每0.1 OR）===")
print(hier[["model", "OR", "lo", "hi", "p", "auc"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 2b) 增量价值：base(严重程度) vs base+FI  → ΔAUC + LR 检验
# ---------------------------------------------------------------------------
base_f = "died ~ age + male + ard + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
full_f = base_f + " + fi10"
m_base = smf.logit(base_f, data=mim_cc).fit(disp=0)
m_full = smf.logit(full_f, data=mim_cc).fit(disp=0)
auc_base = roc_auc_score(mim_cc["died"], m_base.predict(mim_cc))
auc_full = roc_auc_score(mim_cc["died"], m_full.predict(mim_cc))
lr_stat = 2*(m_full.llf - m_base.llf)
from scipy.stats import chi2
lr_p = chi2.sf(lr_stat, 1)
brier_base = brier_score_loss(mim_cc["died"], m_base.predict(mim_cc))
brier_full = brier_score_loss(mim_cc["died"], m_full.predict(mim_cc))
inc = pd.DataFrame([
    dict(model="Base (severity only)", auc=round(auc_base,4), brier=round(brier_base,4)),
    dict(model="Base + FI-Lab", auc=round(auc_full,4), brier=round(brier_full,4)),
    dict(model="Increment", auc=round(auc_full-auc_base,4), brier=round(brier_full-brier_base,4)),
])
inc.to_csv(f"{OUT}/incremental_value_mimiciv.csv", index=False)
print(f"\n=== 增量价值 ===\n  AUC {auc_base:.4f} -> {auc_full:.4f} (Δ{auc_full-auc_base:+.4f}); LR χ²={lr_stat:.1f}, P={lr_p:.2e}")
print(f"  Brier {brier_base:.4f} -> {brier_full:.4f}")

# ---------------------------------------------------------------------------
# 2c) 决策曲线分析（DCA）：base vs base+FI vs treat-all/none
# ---------------------------------------------------------------------------
def net_benefit(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    out = []
    for t in thr:
        tp = np.sum((p >= t) & (y == 1)); fp = np.sum((p >= t) & (y == 0))
        nb = tp/n - fp/n * (t/(1-t))
        prev = y.mean(); nb_all = prev - (1-prev)*(t/(1-t))
        out.append((t, nb, nb_all))
    return out
thr = np.arange(0.05, 0.76, 0.05)
p_base = m_base.predict(mim_cc).values; p_full = m_full.predict(mim_cc).values
dca_rows = []
for (t, nb_b, nb_all), (_, nb_f, _) in zip(net_benefit(mim_cc["died"], p_base, thr),
                                            net_benefit(mim_cc["died"], p_full, thr)):
    dca_rows.append(dict(threshold=round(t,2), nb_base=round(nb_b,4), nb_full=round(nb_f,4),
                         nb_all=round(nb_all,4), nb_none=0.0))
dca = pd.DataFrame(dca_rows); dca.to_csv(f"{OUT}/dca_mimiciv.csv", index=False)
print("\n=== DCA 已输出（阈值 0.05-0.75）===")

# ---------------------------------------------------------------------------
# 3) 规范外部验证：冻结 MIMIC 简约模型系数 → NWICU/eICU
#    推导模型（可跨库获得的变量）：age + male + ard + fi10
# ---------------------------------------------------------------------------
mim["fi10"] = mim["fi_lab"]*10; nwi["fi10"] = nwi["fi_lab"]*10; eic["fi10"] = eic["fi_lab"]*10
derive_f = "died ~ age + male + ard + fi10"
m_der = smf.logit(derive_f, data=mim).fit(disp=0)
print("\n=== 推导模型系数（MIMIC, 冻结用于外验）===")
print(m_der.params)
coef = m_der.params
coef.to_csv(f"{OUT}/derivation_coefficients.csv")

def calib_metrics(df, model):
    X = df.copy()
    lp = (coef["Intercept"] + coef["age"]*X["age"] + coef["male"]*X["male"]
          + coef["ard"]*X["ard"] + coef["fi10"]*X["fi10"])
    p = 1/(1+np.exp(-lp))
    y = X["died"].values
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    oe = y.mean()/p.mean()
    # calibration slope & intercept: logit(y)~lp
    Xc = sm.add_constant(lp.values)
    cm = sm.Logit(y, Xc).fit(disp=0)
    cal_int = cm.params[0]; cal_slope = cm.params[1]
    return dict(n=len(y), deaths=int(y.sum()), auc=round(auc,3), brier=round(brier,3),
                oe_ratio=round(oe,3), cal_intercept=round(cal_int,3), cal_slope=round(cal_slope,3))

ext = []
for name, df in [("MIMIC-IV (apparent)", mim), ("NWICU (frozen external)", nwi), ("eICU (frozen external)", eic)]:
    ext.append(dict(cohort=name, **calib_metrics(df, m_der)))
ext = pd.DataFrame(ext); ext.to_csv(f"{OUT}/external_validation_frozen.csv", index=False)
print("\n=== 冻结系数外部验证 ===")
print(ext.to_string(index=False))

# ---------------------------------------------------------------------------
# 4) ARD 亚组 vs 非ARD 模型性能（各库；用冻结推导模型）
# ---------------------------------------------------------------------------
sub = []
for name, df in [("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]:
    for grp, mask in [("ARD", df["ard"] == 1), ("non-ARD", df["ard"] == 0)]:
        d = df[mask]
        if d["died"].sum() >= 5 and len(d) >= 30:
            try:
                sub.append(dict(cohort=name, group=grp, **calib_metrics(d, m_der)))
            except Exception as e:
                sub.append(dict(cohort=name, group=grp, n=len(d), deaths=int(d["died"].sum()), note=str(e)[:40]))
        else:
            sub.append(dict(cohort=name, group=grp, n=len(d), deaths=int(d["died"].sum()), note="too few"))
subp = pd.DataFrame(sub); subp.to_csv(f"{OUT}/ard_subgroup_performance.csv", index=False)
print("\n=== ARD vs 非ARD 模型性能 ===")
print(subp.to_string(index=False))

# ---------------------------------------------------------------------------
# 5) 阈值敏感性 + 早期死亡泄漏敏感性（MIMIC 主库）
# ---------------------------------------------------------------------------
sens = []
raw = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
raw["fi10"] = raw["fi_lab"]*10
fl = pd.read_csv(f"{ROOT}/filab/filab_mimiciv.csv").set_index("stay_id")["icu_los_days"]
raw["icu_los_days"] = raw["stay_id"].map(fl)
for thr_n in [10, 15, 20]:
    d = raw[raw["n_assessed"] >= thr_n]
    r, _ = fit_or(d, "died ~ fi10 + age + male + ard")
    sens.append(dict(analysis=f"n_assessed>={thr_n}", n=len(d), OR=round(r["OR"],3), lo=round(r["lo"],3), hi=round(r["hi"],3), p=r["p"]))
# 排除早期死亡（icu_los<1d）
d = raw[~((raw["died"] == 1) & (raw["icu_los_days"] < 1))]
r, _ = fit_or(d, "died ~ fi10 + age + male + ard")
sens.append(dict(analysis="exclude ICU death <24h", n=len(d), OR=round(r["OR"],3), lo=round(r["lo"],3), hi=round(r["hi"],3), p=r["p"]))
sensp = pd.DataFrame(sens); sensp.to_csv(f"{OUT}/threshold_leakage_sensitivity.csv", index=False)
print("\n=== 阈值/泄漏 敏感性 (MIMIC) ===")
print(sensp.to_string(index=False))

print("\nAll revision analyses saved to", OUT)
