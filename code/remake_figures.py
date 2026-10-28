# -*- coding: utf-8 -*-
"""
重制全部图片：统一字体为 Times New Roman，并将 11 张原始图按逻辑组织为 5 张组合组图，
导出 PDF（矢量）与 PNG。同时刷新 figures/ 下 11 张独立 PNG 为 TNR 版本。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



# ============ 全局字体：Times New Roman ============
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

ROOT = ROOT
FIG = os.path.join(ROOT, "figures")
COMB = os.path.join(ROOT, "figures", "combined")
os.makedirs(FIG, exist_ok=True)
os.makedirs(COMB, exist_ok=True)

C_ARD = "#C0392B"; C_NON = "#2E86C1"
DBS = ["mimiciv", "nwicu", "eicu"]
DBLAB = {"mimiciv": "MIMIC-IV", "nwicu": "NWICU", "eicu": "eICU"}

# ---------- 载入 ----------
dfs = {}
for db in DBS:
    d = pd.read_csv(f"{ROOT}/analysis/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
    d["db"] = db
    dfs[db] = d
allc = pd.concat(dfs.values(), ignore_index=True)

# ============================================================
# 子图绘制函数（传入 ax）
# ============================================================
def panel_letter(ax, letter):
    ax.text(-0.12, 1.01, letter, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", va="bottom", ha="left")

def plot_fi_dist(ax):
    data_ard = [dfs[db].loc[dfs[db].ard == 1, "fi_lab"] for db in DBS]
    data_non = [dfs[db].loc[dfs[db].ard == 0, "fi_lab"] for db in DBS]
    pos = np.arange(1, 4)
    bp1 = ax.boxplot(data_ard, positions=pos - 0.18, widths=0.3, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    bp2 = ax.boxplot(data_non, positions=pos + 0.18, widths=0.3, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for b in bp1["boxes"]:
        b.set_facecolor(C_ARD); b.set_alpha(0.75)
    for b in bp2["boxes"]:
        b.set_facecolor(C_NON); b.set_alpha(0.75)
    for b in list(bp1["boxes"]) + list(bp2["boxes"]):
        b.set_edgecolor("black")
    ax.set_xticks(pos); ax.set_xticklabels([DBLAB[d] for d in DBS])
    ax.set_ylabel("FI-Lab")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["ARD", "Non-ARD"], loc="upper left", frameon=False)
    ax.set_ylim(0, 0.9)

def plot_quartile(ax, db):
    d = dfs[db]
    q = pd.qcut(d["fi_lab"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    d = d.assign(q=q)
    rates = []
    for lv in ["Q1", "Q2", "Q3", "Q4"]:
        sub = d[d.q == lv]
        rates.append((sub.loc[sub.ard == 1, "died"].mean() * 100,
                      sub.loc[sub.ard == 0, "died"].mean() * 100))
    rates = np.array(rates); x = np.arange(4); w = 0.38
    ax.bar(x - w/2, rates[:, 0], w, color=C_ARD, alpha=0.85, label="ARD")
    ax.bar(x + w/2, rates[:, 1], w, color=C_NON, alpha=0.85, label="Non-ARD")
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(DBLAB[db], fontsize=11.5)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("FI-Lab quartile")
    if rates[3, 0] > 0 and rates[0, 0] > 0:
        ax.annotate(f"{rates[3,0]/rates[0,0]:.1f}×", xy=(3 - w/2, rates[3, 0]),
                    xytext=(3 - w/2, rates[3, 0] + 2), ha="center",
                    fontsize=10, fontweight="bold", color=C_ARD)
    ax.set_ylim(0, max(rates.flatten()) * 1.25)
    ax.grid(axis="y", alpha=0.25)

def rcs_predict(db):
    d = dfs[db].copy(); d["male"] = d["male"].astype(float)
    med_age = d["age"].median()
    X = patsy.dmatrix("cr(fi_lab, df=4) + age + male + ard", d, return_type="dataframe")
    y = d["died"].astype(int)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    grid = np.linspace(d.fi_lab.quantile(0.02), d.fi_lab.quantile(0.98), 100)
    G = pd.DataFrame({"fi_lab": grid, "age": med_age, "male": 0.0, "ard": 0.0})
    Xg = patsy.build_design_matrices([X.design_info], G, return_type="dataframe")[0]
    pred = fit.get_prediction(Xg).summary_frame(alpha=0.05)
    return grid, pred["mean"].values, pred["mean_ci_lower"].values, pred["mean_ci_upper"].values

def plot_rcs(ax, db):
    g, lo, lw, hi = rcs_predict(db)
    ax.plot(g, lo*100, color="#1A5276", lw=2)
    ax.fill_between(g, lw*100, hi*100, color="#AED6F1", alpha=0.5)
    ax.set_title(DBLAB[db], fontsize=11.5); ax.set_xlabel("FI-Lab")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(alpha=0.25)

# 主森林图（来自 results_table.csv）
def plot_forest_main(ax):
    rt = pd.read_csv(f"{ROOT}/analysis/results_table.csv")
    rt = rt[rt["model"].isin(["MIMIC-IV (primary)", "MIMIC-IV (+Charlson)",
                              "NWICU (external)", "eICU (external)"])].reset_index(drop=True)
    yp = np.arange(len(rt))[::-1]
    for i, (_, r) in enumerate(rt.iterrows()):
        ax.plot([r["OR_per0.1_FI_lo"], r["OR_per0.1_FI_hi"]], [yp[i], yp[i]], color="#34495E", lw=1.5)
        ax.plot([r["OR_per0.1_FI"]], [yp[i]], "o", color=C_ARD, ms=7)
        ax.annotate(f"{r['OR_per0.1_FI']:.2f} [{r['OR_per0.1_FI_lo']:.2f}, {r['OR_per0.1_FI_hi']:.2f}]",
                    xy=(r["OR_per0.1_FI_hi"], yp[i]), xytext=(r["OR_per0.1_FI_hi"] + 0.04, yp[i]),
                    va="center", fontsize=9)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yp); ax.set_yticklabels(rt["model"])
    ax.set_xlabel("OR for in-hospital death per 0.1 increase in FI-Lab (95% CI)")
    ax.set_xlim(1.0, max(rt["OR_per0.1_FI_hi"]) * 1.25)
    ax.grid(axis="x", alpha=0.25)

# 校准图（MIMIC 推导模型 -> 三库）
def plot_calib(ax, db):
    base = dfs["mimiciv"].copy(); base["male"] = base["male"].astype(float)
    m = smf.logit("died ~ ard*fi_lab + age + male", base).fit(disp=0)
    beta = m.params
    d = dfs[db].copy(); d["male"] = d["male"].astype(float)
    lp = (beta["Intercept"] + beta["ard"]*d["ard"] + beta["fi_lab"]*d["fi_lab"]
          + beta.get("ard:fi_lab", 0.0)*d["ard"]*d["fi_lab"]
          + beta["age"]*d["age"] + beta["male"]*d["male"])
    d["pred"] = 1/(1+np.exp(-lp))
    d["dec"] = pd.qcut(d["pred"], 10, duplicates="drop")
    g = d.groupby("dec", observed=True).agg(obs=("died", "mean"), pred=("pred", "mean"))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(g["pred"], g["obs"], "o-", color="#1A5276", lw=1.8, ms=7)
    ax.set_title(f"{DBLAB[db]} (AUC={roc_auc_score(d.died.astype(int), d.pred):.2f})", fontsize=11.5)
    ax.set_xlabel("Predicted mortality probability")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.25)

# 结局森林图（来自 results_outcomes.csv）
def plot_forest_outcomes(ax):
    rt = pd.read_csv(f"{ROOT}/analysis/results_outcomes.csv")
    rt = rt.reset_index(drop=True)
    yp = np.arange(len(rt))[::-1]
    for i, (_, r) in enumerate(rt.iterrows()):
        ax.plot([r.lo, r.hi], [yp[i], yp[i]], color="#34495E", lw=1.5)
        ax.plot([r.OR], [yp[i]], "o", color=C_ARD, ms=7)
        ax.annotate(f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]", xy=(r.hi, yp[i]),
                    xytext=(r.hi + 0.03, yp[i]), va="center", fontsize=8.5)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yp); ax.set_yticklabels([f"{r.database}\n{r.outcome}" for _, r in rt.iterrows()])
    ax.set_xlabel("OR per 0.1 increase in FI-Lab (95% CI)")
    ax.set_xlim(1.0, 2.2); ax.grid(axis="x", alpha=0.25)

# 30天结局四分位面板
def plot_quartile_outcome(ax, db, y, title):
    o = pd.read_csv(f"{ROOT}/analysis/outcomes_{db}.csv").dropna(subset=["fi_lab"]).copy()
    q = pd.qcut(o["fi_lab"], 4, labels=["Q1", "Q2", "Q3", "Q4"]); o = o.assign(q=q)
    rates = []
    for lv in ["Q1", "Q2", "Q3", "Q4"]:
        sub = o[o.q == lv]
        rates.append((sub.loc[sub.ard == 1, y].mean()*100, sub.loc[sub.ard == 0, y].mean()*100))
    rates = np.array(rates); x = np.arange(4); w = 0.38
    ax.bar(x - w/2, rates[:, 0], w, color=C_ARD, alpha=0.85, label="ARD")
    ax.bar(x + w/2, rates[:, 1], w, color=C_NON, alpha=0.85, label="Non-ARD")
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title(f"{DBLAB[db]}", fontsize=11)
    ax.set_xlabel("FI-Lab quartile")
    ax.set_ylabel(f"{title} (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

# 年代森林
def plot_era_forest(ax):
    e = pd.read_csv(f"{ROOT}/analysis/results_sensitivity_era.csv")
    e = e.dropna(subset=["OR"]).reset_index(drop=True)
    yp = np.arange(len(e))[::-1]
    for i, (_, r) in enumerate(e.iterrows()):
        ax.plot([r.lo, r.hi], [yp[i], yp[i]], color="#34495E", lw=1.5)
        ax.plot([r.OR], [yp[i]], "o", color=C_NON, ms=6)
        ax.annotate(f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]", xy=(r.hi, yp[i]),
                    xytext=(min(r.hi + 0.03, 2.35), yp[i]), va="center", fontsize=8)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yp); ax.set_yticklabels([f"{r.database} | {r.era} (n={int(r.n):,}, d={int(r.deaths)})" for _, r in e.iterrows()])
    ax.set_xlabel("OR for in-hospital death per 0.1 FI-Lab (95% CI)")
    ax.set_xlim(0.8, 2.4); ax.grid(axis="x", alpha=0.25)

def plot_covid(ax):
    cc = pd.read_csv(f"{ROOT}/analysis/results_sensitivity_covid.csv")
    cc = cc[cc.database.isin(["MIMIC-IV", "NWICU"])].reset_index(drop=True)
    labels = []; k = 0
    for db in ["MIMIC-IV", "NWICU"]:
        for tag, off, col in [("All", 0.16, C_ARD), ("Excl. COVID", -0.16, C_NON)]:
            row = cc[(cc.database == db) & (cc.cohort == tag)].iloc[0]
            y = k + off
            ax.plot([row.lo, row.hi], [y, y], color="#34495E", lw=1.5)
            ax.plot([row.OR], [y], "o", color=col, ms=8, label=tag if k == 0 else "")
            ax.annotate(f"{row.OR:.2f}", xy=(row.hi, y), xytext=(row.hi + 0.02, y), va="center", fontsize=9)
        labels.append(db); k += 1
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("OR for death per 0.1 FI-Lab (95% CI)")
    ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", alpha=0.25)

def plot_subtype_forest(ax):
    s = pd.read_csv(f"{ROOT}/analysis/results_sensitivity_subtype.csv")
    s = s.dropna(subset=["OR"]).reset_index(drop=True)
    yp = np.arange(len(s))[::-1]
    for i, (_, r) in enumerate(s.iterrows()):
        ax.plot([r.lo, r.hi], [yp[i], yp[i]], color="#34495E", lw=1.5)
        ax.plot([r.OR], [yp[i]], "o", color=C_ARD, ms=6)
        ax.annotate(f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]", xy=(r.hi, yp[i]),
                    xytext=(min(r.hi + 0.03, 3.1), yp[i]), va="center", fontsize=8)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yp); ax.set_yticklabels([f"{r.database} | {r.subtype} (n={int(r.n)}, d={int(r.deaths)})" for _, r in s.iterrows()])
    ax.set_xlabel("OR for in-hospital death per 0.1 FI-Lab (95% CI)")
    ax.set_xlim(0.6, 3.2); ax.grid(axis="x", alpha=0.25)

# ============================================================
# A. 刷新 11 张独立 PNG (TNR)
# ============================================================
# fig1
fig, ax = plt.subplots(figsize=(7.5, 5)); plot_fi_dist(ax)
ax.set_title("Figure 1. Distribution of FI-Lab by database and ARD status")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_filab_distribution.png"); plt.close(fig)
# fig2 (3-panel)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
for j, db in enumerate(DBS): plot_quartile(axes[j], db)
axes[0].set_ylabel("In-hospital mortality (%)")
fig.suptitle("Figure 2. In-hospital mortality across FI-Lab quartiles, by ARD status", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_mortality_by_quartile.png", bbox_inches="tight"); plt.close(fig)
# fig3 (3-panel)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for j, db in enumerate(DBS): plot_rcs(axes[j], db)
axes[0].set_ylabel("Predicted in-hospital mortality (%)")
fig.suptitle("Figure 3. Adjusted RCS of FI-Lab vs in-hospital mortality", fontsize=12.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_rcs_spline.png", bbox_inches="tight"); plt.close(fig)
# fig4
fig, ax = plt.subplots(figsize=(8, 4.2)); plot_forest_main(ax)
ax.set_title("Figure 4. Association of FI-Lab with mortality across databases")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_forest.png"); plt.close(fig)
# fig5 (3-panel)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True, sharex=True)
for j, db in enumerate(DBS): plot_calib(axes[j], db)
axes[0].set_ylabel("Observed mortality (%)")
fig.suptitle("Figure 5. Calibration of MIMIC-IV-derived model across databases", fontsize=12.5, y=1.03)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_calibration.png", bbox_inches="tight"); plt.close(fig)
# fig6
fig, ax = plt.subplots(figsize=(9, 5.2)); plot_forest_outcomes(ax)
ax.set_title("Figure 6. FI-Lab association with in-hospital, 30-day mortality and readmission")
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_forest_outcomes.png"); plt.close(fig)
# fig7/8 (quartile outcomes)
for db in ["mimiciv", "nwicu"]:
    o = pd.read_csv(f"{ROOT}/analysis/outcomes_{db}.csv").dropna(subset=["fi_lab"]).copy()
    q = pd.qcut(o["fi_lab"], 4, labels=["Q1", "Q2", "Q3", "Q4"]); o = o.assign(q=q)
    suffix = "" if db == "mimiciv" else "b"
    for y, t, base in [("mort30", "30-day mortality", "fig7"), ("readmit30", "30-day readmission", "fig8")]:
        rates = []
        for lv in ["Q1", "Q2", "Q3", "Q4"]:
            sub = o[o.q == lv]
            rates.append((sub.loc[sub.ard == 1, y].mean()*100, sub.loc[sub.ard == 0, y].mean()*100))
        rates = np.array(rates); x = np.arange(4); w = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(x - w/2, rates[:, 0], w, color=C_ARD, alpha=0.85, label="ARD")
        ax.bar(x + w/2, rates[:, 1], w, color=C_NON, alpha=0.85, label="Non-ARD")
        ax.set_xticks(x); ax.set_xticklabels(["Q1","Q2","Q3","Q4"]); ax.set_xlabel("FI-Lab quartile")
        ax.set_ylabel(f"{t} (%)"); ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_title(DBLAB[db]); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(f"{FIG}/{base}{suffix}_{'30dmort' if 'mortality' in t else '30dreadmit'}_quartile.png"); plt.close(fig)
# fig9/10/11
fig, ax = plt.subplots(figsize=(9, 5)); plot_era_forest(ax)
ax.set_title("Figure 9. FI-Lab and in-hospital mortality across admission eras")
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_era_forest.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.2)); plot_covid(ax)
ax.set_title("Figure 10. FI-Lab effect robust to COVID-19 exclusion")
fig.tight_layout(); fig.savefig(f"{FIG}/fig10_covid_sensitivity.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.5)); plot_subtype_forest(ax)
ax.set_title("Figure 11. FI-Lab and in-hospital mortality within ARD subtypes")
fig.tight_layout(); fig.savefig(f"{FIG}/fig11_subtype_forest.png"); plt.close(fig)
print("Refreshed 11 standalone PNGs (Times New Roman).")

# ============================================================
# B. 5 张组合组图 (PDF + PNG)
# ============================================================
def save_comb(fig, name):
    fig.tight_layout()
    fig.savefig(f"{COMB}/{name}.pdf")
    fig.savefig(f"{COMB}/{name}.png", dpi=300)

# 组图1：FI分布 + 四分位死亡率
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_fi_dist(axes[0, 0]); panel_letter(axes[0, 0], "A")
axes[0, 0].set_title("FI-Lab distribution (ARD vs Non-ARD)", fontsize=11.5)
for j, db in enumerate(DBS):
    ax = axes[0, 1] if j == 0 else (axes[1, 0] if j == 1 else axes[1, 1])
    plot_quartile(ax, db); panel_letter(ax, chr(66 + j))
fig.suptitle("Figure 1. FI-Lab characteristics and mortality gradient across quartiles", fontsize=14, y=0.99)
save_comb(fig, "Combined_Fig1_FI_distribution_quartile")
plt.close(fig)

# 组图2：RCS + 主森林
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
pos = [(0, 0), (0, 1), (1, 0)]
for j, db in enumerate(DBS):
    plot_rcs(axes[pos[j]], db); panel_letter(axes[pos[j]], chr(65 + j))
axes[0, 0].set_ylabel("Predicted in-hospital mortality (%)")
plot_forest_main(axes[1, 1]); panel_letter(axes[1, 1], "D")
fig.suptitle("Figure 2. Dose-response of FI-Lab with mortality and cross-database consistency", fontsize=14, y=0.99)
save_comb(fig, "Combined_Fig2_RCS_forest")
plt.close(fig)

# 组图3：校准 + 结局森林
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for j, db in enumerate(DBS):
    plot_calib(axes[pos[j]], db); panel_letter(axes[pos[j]], chr(65 + j))
axes[0, 0].set_ylabel("Observed mortality (%)")
plot_forest_outcomes(axes[1, 1]); panel_letter(axes[1, 1], "D")
fig.suptitle("Figure 3. Model calibration and extended outcomes (30-day)", fontsize=14, y=0.99)
save_comb(fig, "Combined_Fig3_calibration_outcomes")
plt.close(fig)

# 组图4：30天结局四分位
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_quartile_outcome(axes[0, 0], "mimiciv", "mort30", "30-day mortality"); panel_letter(axes[0, 0], "A")
plot_quartile_outcome(axes[0, 1], "nwicu", "mort30", "30-day mortality"); panel_letter(axes[0, 1], "B")
plot_quartile_outcome(axes[1, 0], "mimiciv", "readmit30", "30-day readmission"); panel_letter(axes[1, 0], "C")
plot_quartile_outcome(axes[1, 1], "nwicu", "readmit30", "30-day readmission"); panel_letter(axes[1, 1], "D")
fig.suptitle("Figure 4. FI-Lab and 30-day mortality / readmission by quartile", fontsize=14, y=0.99)
save_comb(fig, "Combined_Fig4_30day_outcomes")
plt.close(fig)

# 组图5：敏感性分析
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
plot_era_forest(axes[0, 0]); panel_letter(axes[0, 0], "A")
axes[0, 0].set_title("By admission era", fontsize=11.5)
plot_covid(axes[0, 1]); panel_letter(axes[0, 1], "B")
axes[0, 1].set_title("Excluding COVID-19", fontsize=11.5)
plot_subtype_forest(axes[1, 0]); panel_letter(axes[1, 0], "C")
axes[1, 0].set_title("Within ARD subtypes", fontsize=11.5)
axes[1, 1].axis("off")
axes[1, 1].text(0.02, 0.95, "Sensitivity conclusions", fontsize=12, fontweight="bold", va="top")
txt = ("• FI-Lab effect stable across 2008-2022 (OR 1.59-1.68 in MIMIC).\n"
       "• Excluding COVID-19 leaves the effect unchanged (MIMIC 1.64 -> 1.64; NWICU 1.54 -> 1.68).\n"
       "• Within ARD subtypes, RA and vasculitis show strong, significant FI-mortality\n"
       "  associations; SLE is underpowered (n=88, n.s.) and needs larger samples.\n"
       "• No ARD×FI interaction in any database (p=0.26/0.35/0.79) => effect is\n"
       "  universal, not ARD-specific.")
axes[1, 1].text(0.02, 0.82, txt, fontsize=10.5, va="top", linespacing=1.5)
fig.suptitle("Figure 5. Sensitivity and subgroup analyses", fontsize=14, y=0.99)
save_comb(fig, "Combined_Fig5_sensitivity")
plt.close(fig)

# 主合并 PDF（多页）
from matplotlib.backends.backend_pdf import PdfPages
pdf_path = os.path.join(COMB, "All_Figures_TNR.pdf")
with PdfPages(pdf_path) as pdf:
    for name in ["Combined_Fig1_FI_distribution_quartile", "Combined_Fig2_RCS_forest",
                 "Combined_Fig3_calibration_outcomes", "Combined_Fig4_30day_outcomes",
                 "Combined_Fig5_sensitivity"]:
        fig = plt.figure(figsize=(13, 9))
        img = plt.imread(f"{COMB}/{name}.png")
        plt.imshow(img); plt.axis("off")
        pdf.savefig(fig); plt.close(fig)
print("Combined figures + master PDF saved to", COMB)
print("DONE.")
