# -*- coding: utf-8 -*-
"""
方向 B 的第 1 步：临床化呈现 + 论文级可视化
输入：analysis/analysis_{mimiciv,nwicu,eicu}.csv （含 ARD 与非 ARD 对照队列，30 项 FI-Lab）
产出：figures/ 下 5 张出版级 PNG + results_table.csv（临床指标汇总）
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
import patsy
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

# ---- 出版级样式 ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

C_ARD = "#C0392B"      # 风湿病(ARD) 红
C_NON = "#2E86C1"      # 非ARD 蓝
DBS = ["mimiciv", "nwicu", "eicu"]
DBLAB = {"mimiciv": "MIMIC-IV", "nwicu": "NWICU", "eicu": "eICU"}

# ---------- 载入 ----------
dfs = {}
for db in DBS:
    d = pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{db}.csv'))
    d = d.dropna(subset=["fi_lab"]).copy()
    d["db"] = db
    dfs[db] = d
    print(f"{db}: n={len(d):,}  deaths={int(d.died.sum()):,}  ARD={int(d.ard.sum()):,}")
allc = pd.concat(dfs.values(), ignore_index=True)

# ============================================================
# 图1：FI-Lab 分布（按数据库，ARD 分层）
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
data_ard = [dfs[db].loc[dfs[db].ard == 1, "fi_lab"] for db in DBS]
data_non = [dfs[db].loc[dfs[db].ard == 0, "fi_lab"] for db in DBS]
positions = np.arange(1, 4)
bp1 = ax.boxplot(data_ard, positions=positions - 0.18, widths=0.3, patch_artist=True,
                 showfliers=False, medianprops=dict(color="black", linewidth=1.5))
bp2 = ax.boxplot(data_non, positions=positions + 0.18, widths=0.3, patch_artist=True,
                 showfliers=False, medianprops=dict(color="black", linewidth=1.5))
for b in bp1["boxes"]:
    b.set_facecolor(C_ARD); b.set_alpha(0.75)
for b in bp2["boxes"]:
    b.set_facecolor(C_NON); b.set_alpha(0.75)
for b in list(bp1["boxes"]) + list(bp2["boxes"]):
    b.set_edgecolor("black")
ax.set_xticks(positions); ax.set_xticklabels([DBLAB[d] for d in DBS])
ax.set_ylabel("FI-Lab (deficit-accumulation frailty index)")
ax.set_title("Figure 1. Distribution of FI-Lab by database and rheumatic-disease status")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
          ["Autoimmune/rheumatic disease (ARD)", "Non-ARD controls"], loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_filab_distribution.png"); plt.close(fig)

# ============================================================
# 图2：FI 四分位死亡率（ARD 分层），3 面板
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
for j, db in enumerate(DBS):
    d = dfs[db]
    q = pd.qcut(d["fi_lab"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    d = d.assign(q=q)
    rates = []
    for lv in ["Q1", "Q2", "Q3", "Q4"]:
        sub = d[d.q == lv]
        r_ard = sub.loc[sub.ard == 1, "died"].mean()
        r_non = sub.loc[sub.ard == 0, "died"].mean()
        rates.append((r_ard, r_non))
    rates = np.array(rates)
    x = np.arange(4); w = 0.38
    ax = axes[j]
    b1 = ax.bar(x - w/2, rates[:, 0]*100, w, color=C_ARD, alpha=0.85, label="ARD")
    b2 = ax.bar(x + w/2, rates[:, 1]*100, w, color=C_NON, alpha=0.85, label="Non-ARD")
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n(least)\n", "Q2", "Q3", "Q4\n(most)"])
    ax.set_title(DBLAB[db])
    ax.yaxis.set_major_formatter(PercentFormatter())
    if j == 0:
        ax.set_ylabel("In-hospital mortality (%)")
        ax.legend(loc="upper left", frameon=False)
    # 标注 ARD 组内 Q4 vs Q1 倍数
    if rates[3, 0] > 0 and rates[0, 0] > 0:
        ax.annotate(f"{rates[3,0]/rates[0,0]:.1f}×",
                    xy=(3 - w/2, rates[3, 0]*100), xytext=(3 - w/2, rates[3, 0]*100 + 3),
                    ha="center", fontsize=10, fontweight="bold", color=C_ARD)
    ax.set_ylim(0, max(rates.flatten())*100*1.25)
fig.suptitle("Figure 2. In-hospital mortality across FI-Lab quartiles, stratified by rheumatic-disease status",
             fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_mortality_by_quartile.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 图3：限制性立方样条（FI vs 死亡概率，调整年龄/性别/ARD）
# ============================================================
def rcs_predict(db):
    d = dfs[db]
    d = d.copy()
    d["male"] = d["male"].astype(float)
    # reference: median age, female, non-ARD
    med_age = d["age"].median()
    X = patsy.dmatrix("cr(fi_lab, df=4) + age + male + ard", d, return_type="dataframe")
    y = d["died"].astype(int)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    grid = np.linspace(d.fi_lab.quantile(0.02), d.fi_lab.quantile(0.98), 100)
    G = pd.DataFrame({"fi_lab": grid, "age": med_age, "male": 0.0, "ard": 0.0})
    Xg = patsy.build_design_matrices([X.design_info], G, return_type="dataframe")[0]
    pred = fit.get_prediction(Xg).summary_frame(alpha=0.05)
    return grid, pred["mean"].values, pred["mean_ci_lower"].values, pred["mean_ci_upper"].values

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for j, db in enumerate(DBS):
    g, lo, lw, hi = rcs_predict(db)
    ax = axes[j]
    ax.plot(g, lo*100, color="#1A5276", lw=2)
    ax.fill_between(g, lw*100, hi*100, color="#AED6F1", alpha=0.5)
    ax.set_title(DBLAB[db]); ax.set_xlabel("FI-Lab")
    ax.yaxis.set_major_formatter(PercentFormatter())
    if j == 0:
        ax.set_ylabel("Predicted in-hospital mortality (%)")
    ax.grid(alpha=0.25)
fig.suptitle("Figure 3. Adjusted restricted cubic spline of FI-Lab vs in-hospital mortality\n"
             "(reference: median age, female, non-ARD; shaded = 95% CI)", fontsize=12.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rcs_spline.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 图4：森林图 —— 每 0.1 FI 增量的死亡 OR（外部验证）
# ============================================================
def fit_model(db, formula, use_charlson=False):
    d = dfs[db].copy()
    d["male"] = d["male"].astype(float)
    if use_charlson:
        d = d.dropna(subset=["charlson"]).copy()
        d["charlson"] = d["charlson"].astype(float)
    res = smf.logit(formula, d).fit(disp=0)
    return res

models = {
    "MIMIC-IV (primary)": fit_model("mimiciv", "died ~ ard*fi_lab + age + male"),
    "MIMIC-IV (+Charlson)": fit_model("mimiciv", "died ~ ard*fi_lab + age + male + charlson", True),
    "NWICU (external)": fit_model("nwicu", "died ~ ard*fi_lab + age + male"),
    "eICU (external)": fit_model("eicu", "died ~ ard*fi_lab + age + male"),
}
rows = []
for name, res in models.items():
    b = res.params["fi_lab"]; se = res.bse["fi_lab"]
    # OR per 0.1 FI
    orr = np.exp(0.1 * b)
    lo = np.exp(0.1 * (b - 1.96*se))
    hi = np.exp(0.1 * (b + 1.96*se))
    p = res.pvalues["fi_lab"]
    rows.append((name, orr, lo, hi, p))
forest = pd.DataFrame(rows, columns=["model", "OR_per0.1", "lo", "hi", "p"])

fig, ax = plt.subplots(figsize=(8, 4.2))
yp = np.arange(len(forest))[::-1]
for i, (_, r) in enumerate(forest.iterrows()):
    ax.plot([r.lo, r.hi], [yp[i], yp[i]], color="#34495E", lw=1.5)
    ax.plot([r["OR_per0.1"]], [yp[i]], "o", color="#C0392B", ms=7)
    ax.annotate(f"{r['OR_per0.1']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                xy=(r.hi, yp[i]), xytext=(r.hi + 0.04, yp[i]),
                va="center", fontsize=10)
ax.axvline(1.0, color="gray", ls="--", lw=1)
ax.set_yticks(yp); ax.set_yticklabels(forest["model"])
ax.set_xlabel("Odds ratio for in-hospital death per 0.1 increase in FI-Lab (95% CI)")
ax.set_title("Figure 4. Association of FI-Lab with mortality across databases")
ax.set_xlim(1.0, max(forest.hi) * 1.25)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_forest.png"); plt.close(fig)

# ============================================================
# 图5：校准图（MIMIC 推导模型 → 三库外部验证）
# ============================================================
base = dfs["mimiciv"].copy(); base["male"] = base["male"].astype(float)
m = smf.logit("died ~ ard*fi_lab + age + male", base).fit(disp=0)
beta = m.params

def predict_prob(d):
    d = d.copy(); d["male"] = d["male"].astype(float)
    lp = (beta["Intercept"]
          + beta["ard"]*d["ard"] + beta["fi_lab"]*d["fi_lab"]
          + beta["ard:fi_lab"]*d["ard"]*d["fi_lab"]
          + beta["age"]*d["age"] + beta["male"]*d["male"])
    return 1/(1+np.exp(-lp))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True, sharex=True)
calib_rows = []
for j, db in enumerate(DBS):
    d = dfs[db]
    d = d.copy(); d["pred"] = predict_prob(d)
    # 十分位数分组
    d["dec"] = pd.qcut(d["pred"], 10, duplicates="drop")
    g = d.groupby("dec", observed=True).agg(obs=("died", "mean"),
                                            pred=("pred", "mean"),
                                            n=("died", "size"))
    ax = axes[j]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(g["pred"], g["obs"], "o-", color="#1A5276", lw=1.8, ms=7)
    ax.set_title(f"{DBLAB[db]}\n(AUC={roc_auc_score(d.died.astype(int), d.pred):.2f})")
    ax.set_xlabel("Predicted mortality probability")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if j == 0:
        ax.set_ylabel("Observed mortality (%)")
    ax.grid(alpha=0.25)
    # 校准斜率/截距
    d2 = d.assign(logit=np.log(d.pred/(1-d.pred)))
    cr = smf.logit("died ~ logit", d2).fit(disp=0)
    calib_rows.append((DBLAB[db], cr.params["Intercept"], cr.params["logit"],
                       roc_auc_score(d.died.astype(int), d.pred)))
calib = pd.DataFrame(calib_rows, columns=["database", "calib_intercept", "calib_slope", "auc"])

fig.suptitle("Figure 5. Calibration of the MIMIC-IV–derived model across databases\n"
             "(points = deciles of predicted risk; dashed = perfect calibration)",
             fontsize=12.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_calibration.png", bbox_inches="tight"); plt.close(fig)

# ============================================================
# 汇总表（可进论文）
# ============================================================
# 每 0.1 FI 的 OR/CI/p，及 ARD 主效应、交互项
summ = []
for name, res in models.items():
    b_fi = res.params["fi_lab"]; se_fi = res.bse["fi_lab"]
    b_ard = res.params["ard"]; se_ard = res.bse["ard"]
    b_int = res.params.get("ard:fi_lab", np.nan); se_int = res.bse.get("ard:fi_lab", np.nan)
    summ.append({
        "model": name,
        "OR_per0.1_FI": round(np.exp(0.1*b_fi), 3),
        "OR_per0.1_FI_lo": round(np.exp(0.1*(b_fi-1.96*se_fi)), 3),
        "OR_per0.1_FI_hi": round(np.exp(0.1*(b_fi+1.96*se_fi)), 3),
        "p_FI": f"{res.pvalues['fi_lab']:.1e}",
        "OR_ARD": round(np.exp(b_ard), 3),
        "p_ARD": f"{res.pvalues['ard']:.2f}",
        "OR_interaction": (round(np.exp(b_int), 3) if not np.isnan(b_int) else ""),
        "p_interaction": (f"{res.pvalues['ard:fi_lab']:.2f}" if not np.isnan(b_int) else ""),
    })
summ = pd.DataFrame(summ)
summ["db_key"] = summ["model"].str.replace(r" \(.*\)", "", regex=True)
calib2 = calib.rename(columns={"database": "db_key"})
out = pd.merge(summ, calib2, on="db_key", how="left").drop(columns=["db_key"])
out.to_csv(os.path.join(ROOT, 'analysis', 'results_table.csv'), index=False)
print("\n=== results_table.csv ===")
print(out.to_string(index=False))
print("\nAll figures saved to", OUT)
