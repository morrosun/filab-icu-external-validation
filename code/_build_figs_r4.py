# -*- coding: utf-8 -*-
"""Round-4 重绘 Figure 2 / Figure 3（ARD 已移出；模型 = fi10 + age + male）
Figure 2: (A) FI-Lab 分布（三库）; (B) MIMIC-IV M1–M4 OR 森林图（cluster-robust SE）
Figure 3: (A) 冻结系数外验 AUC 森林（含 95% CI）; (B–D) 三库校准图（十分位）
输出: figures/revision/fig2_distribution_or.{png,pdf}
      figures/revision/fig3_validation_calibration.{png,pdf}
      analysis/db_audit/fig23_numbers.json
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.unicode_minus": False,
})
ROOT = ROOT
ANA = f"{ROOT}/analysis"
FIG = f"{ROOT}/figures/revision"
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20261025)
COL = {"MIMIC-IV": "#4C72B0", "NWICU": "#DD8452", "eICU": "#55A868"}

def load(db):
    d = pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
    if db == "nwicu":
        # 第七轮：NWICU 使用最终（清洁）院内死亡标签，未知结局者剔除
        lab = pd.read_csv(f"{ANA}/db_audit/nwicu_final_labels.csv")
        lab["stay_id"] = lab["stay_id"].astype(str)
        d["stay_id"] = d["stay_id"].astype(str)
        d = d.merge(lab[["stay_id", "died_clean"]], on="stay_id", how="left")
        d["died"] = d["died_clean"]
        d = d.dropna(subset=["died"]).copy()
    d["fi10"] = d["fi_lab"] * 10
    return d

def boot_idx(g):
    g = np.asarray(g); ug = np.unique(g)
    return ug, {u: np.where(g == u)[0] for u in ug}

def logitp(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def calib_deciles(y, p, nb=10):
    d = pd.DataFrame(dict(y=y, p=p))
    d["bin"] = pd.qcut(d["p"], nb, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(x=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    lp = logitp(p)
    cm = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    gl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return g["x"].values, g["y"].values, float(cm.params[1]), float(gl.params[0])

def full_metrics(y, p, g, B=1000):
    y = np.asarray(y, float); p = np.asarray(p, float)
    lp = logitp(p)
    cm = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    gl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    obs, exp = y.sum(), p.sum(); oe = obs / exp
    ug, idx = boot_idx(g)
    bA, bS = [], []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        if y[ii].min() == y[ii].max():
            continue
        bA.append(roc_auc_score(y[ii], p[ii]))
        l2 = logitp(p[ii])
        X = np.column_stack([np.ones_like(l2), l2]); b = np.array([0.0, 1.0])
        for _ in range(40):
            eta = X @ b; pr = 1/(1+np.exp(-np.clip(eta, -30, 30)))
            w = np.clip(pr*(1-pr), 1e-12, None)
            try: step = np.linalg.solve(X.T@(X*w[:, None]) + 1e-8*np.eye(2), X.T@(y[ii]-pr))
            except np.linalg.LinAlgError: break
            b = b + step
            if np.abs(step).max() < 1e-9: break
        bS.append(b[1])
    q = lambda a, t: float(np.nanpercentile(a, t))
    return dict(n=int(len(y)), deaths=int(obs),
                auc=float(roc_auc_score(y, p)), auc_lo=q(bA, 2.5), auc_hi=q(bA, 97.5),
                brier=float(brier_score_loss(y, p)),
                oe=float(oe), oe_lo=float(oe*np.exp(-1.96/np.sqrt(obs))), oe_hi=float(oe*np.exp(1.96/np.sqrt(obs))),
                slope=float(cm.params[1]), slope_lo=q(bS, 2.5), slope_hi=q(bS, 97.5),
                citl=float(gl.params[0]), citl_lo=float(gl.params[0]-1.96*gl.bse[0]),
                citl_hi=float(gl.params[0]+1.96*gl.bse[0]))

def savefig(fig, base):
    fig.savefig(f"{FIG}/{base}.png", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{FIG}/{base}.pdf", bbox_inches="tight", pad_inches=0.15)
    print("saved", base)

def letter(ax, s, dx=-0.10, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

def main():
    res = {}
    mim, nwi, eic = load("mimiciv"), load("nwicu"), load("eicu")

    # ---------- M1–M4（cluster-robust SE）----------
    rev = pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv"); rev["stay_id"] = rev["stay_id"].astype(str)
    mim["stay_id"] = mim["stay_id"].astype(str)
    mm = mim.merge(rev[["stay_id", "sofa", "oasis", "charlson", "vent", "vaso", "rrt"]],
                   on="stay_id", how="left", suffixes=("", "_rev"))
    mm = mm.dropna(subset=["died", "fi10", "age", "male", "charlson", "sofa", "oasis", "vent", "vaso", "rrt"])
    specs = [("M1", "Age + Sex", "died ~ fi10 + age + male"),
             ("M2", "+Charlson", "died ~ fi10 + age + male + charlson"),
             ("M3", "+SOFA+OASIS", "died ~ fi10 + age + male + charlson + sofa + oasis"),
             ("M4", "+Vent/Vaso/RRT", "died ~ fi10 + age + male + charlson + sofa + oasis + vent + vaso + rrt")]
    rows = []
    for lab, desc, f in specs:
        g = smf.glm(f, data=mm, family=sm.families.Binomial())
        fit = g.fit(cov_type="cluster", cov_kwds={"groups": mm["subject_id"].values})
        b, se = fit.params["fi10"], fit.bse["fi10"]
        rows.append(dict(model=lab, desc=desc, OR=float(np.exp(b)),
                         lo=float(np.exp(b-1.96*se)), hi=float(np.exp(b+1.96*se)),
                         p=float(fit.pvalues["fi10"])))
        print(f"[{lab}] OR={rows[-1]['OR']:.3f} ({rows[-1]['lo']:.3f}-{rows[-1]['hi']:.3f})")
    res["ladder"] = rows

    # ---------- 冻结 parsimonious 模型 ----------
    mod = smf.logit("died ~ fi10 + age + male", data=mim).fit(disp=0)
    coef = {k: float(v) for k, v in mod.params.items()}
    res["coef"] = coef
    print("coef:", coef)

    ext = {}
    p_all = {}
    for name, d in [("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]:
        lp = coef["Intercept"] + coef["fi10"]*d["fi10"] + coef["age"]*d["age"] + coef["male"]*d["male"]
        p = 1/(1+np.exp(-lp))
        p_all[name] = (d, p.values)
        g = d["subject_id"].values if "subject_id" in d.columns and d["subject_id"].notna().any() else np.arange(len(d))
        ext[name] = full_metrics(d["died"].values, p.values, g)
        xs, ys, sl, it = calib_deciles(d["died"].values, p.values)
        ext[name].update(cal_x=[float(v) for v in xs], cal_y=[float(v) for v in ys],
                         cal_slope=sl, cal_citl=it)
        print(f"[{name}] AUC={ext[name]['auc']:.4f} ({ext[name]['auc_lo']:.4f}-{ext[name]['auc_hi']:.4f}) "
              f"O:E={ext[name]['oe']:.3f} slope={ext[name]['slope']:.3f} CITL={ext[name]['citl']:.3f}")
    res["ext"] = ext
    json.dump(res, open(f"{ANA}/db_audit/fig23_numbers.json", "w"), indent=2)

    # ================= Figure 2 =================
    fig = plt.figure(figsize=(12, 4.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.26)
    gsA = gs[0].subgridspec(1, 3, wspace=0.30)
    for j, (name, d) in enumerate([("MIMIC-IV", mim), ("NWICU", nwi), ("eICU", eic)]):
        ax = fig.add_subplot(gsA[0, j])
        ax.hist(d["fi_lab"].dropna(), bins=30, color=COL[name], edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name}\n(n = {len(d):,})", fontsize=10.5)
        ax.set_xlabel("FI-Lab")
        if j == 0:
            ax.set_ylabel("Number of hospitalizations")
        med = d["fi_lab"].median()
        ax.axvline(med, color="#C44E52", ls="--", lw=1.2)
        ax.text(0.96, 0.04, f"Median {med:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#C44E52",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
    # 面板字母 A 放在第一个子图左上
    fig.text(0.075, 0.94, "A", fontsize=14, fontweight="bold")

    axB = fig.add_subplot(gs[1])
    yy = np.arange(len(rows))
    ors = np.array([r["OR"] for r in rows]); los = np.array([r["lo"] for r in rows]); his = np.array([r["hi"] for r in rows])
    axB.errorbar(ors, yy, xerr=[ors-los, his-ors], fmt="o", color="#4C72B0",
                 capsize=3.5, ms=5.5, lw=1.2)
    for i, r in enumerate(rows):
        axB.annotate(f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", xy=(r["hi"], i),
                     xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.4)
    axB.set_yticks(yy)
    axB.set_yticklabels([f"{r['model']}\n{r['desc']}" for r in rows], fontsize=8.8)
    axB.invert_yaxis(); axB.margins(y=0.16)
    axB.set_xscale("log")
    axB.axvline(1.0, color="grey", ls=":", lw=0.9)
    axB.set_xlim(1.20, 1.72)
    axB.set_xticks([1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    axB.set_xticklabels(["1.2", "1.3", "1.4", "1.5", "1.6", "1.7"])
    axB.set_xlabel("FI-Lab per 0.1 increase: OR (95% CI, log scale)")
    axB.set_title("Attenuation after severity adjustment (MIMIC-IV)", fontsize=10.5, pad=8)
    letter(axB, "B")
    savefig(fig, "fig2_distribution_or")
    plt.close(fig)

    # ================= Figure 3 =================
    names = ["MIMIC-IV", "NWICU", "eICU"]
    labels = ["MIMIC-IV\n(apparent)", "NWICU\n(frozen external)", "eICU\n(frozen external)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    fig.subplots_adjust(hspace=0.46, wspace=0.28)
    axA = axes[0, 0]
    aucs = np.array([ext[n]["auc"] for n in names])
    los_ = np.array([ext[n]["auc_lo"] for n in names]); his_ = np.array([ext[n]["auc_hi"] for n in names])
    axA.errorbar(aucs, np.arange(3), xerr=[aucs-los_, his_-aucs], fmt="o", ms=6,
                 color="#55A868", capsize=3.5, lw=1.2)
    for i, n in enumerate(names):
        e = ext[n]
        axA.text(e["auc_hi"]+0.006, i, f"{e['auc']:.3f} ({e['auc_lo']:.3f}–{e['auc_hi']:.3f})",
                 va="center", fontsize=8.5)
    axA.set_yticks(np.arange(3)); axA.set_yticklabels(labels, fontsize=9)
    axA.invert_yaxis(); axA.margins(y=0.20)
    axA.axvline(0.5, color="grey", ls=":", lw=0.9)
    axA.set_xlim(0.665, 0.775)
    axA.set_xlabel("In-hospital mortality AUC (95% CI)")
    axA.set_title("Cross-database discrimination", fontsize=10.5)
    letter(axA, "A")

    for (r, c), tag, name in zip([(0, 1), (1, 0), (1, 1)], ["B", "C", "D"], names):
        ax = axes[r, c]; e = ext[name]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal calibration")
        ax.plot(e["cal_x"], e["cal_y"], "o-", color=COL[name], ms=5, lw=1.4,
                label=f"{name}\nslope {e['slope']:.2f} ({e['slope_lo']:.2f}–{e['slope_hi']:.2f})\n"
                      f"CITL {e['citl']:.2f}; O:E {e['oe']:.2f}")
        ax.set_xlabel("Predicted probability of in-hospital death")
        ax.set_ylabel("Observed proportion")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7.6, loc="upper left", frameon=False)
        ax.text(0.97, 0.05, f"n = {e['n']:,}; deaths = {e['deaths']:,}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        ax.set_title(f"Calibration — {name}", fontsize=10.5)
        letter(ax, tag, dx=-0.16)
    savefig(fig, "fig3_validation_calibration")
    plt.close(fig)
    print("JSON →", f"{ANA}/db_audit/fig23_numbers.json")

if __name__ == "__main__":
    main()
