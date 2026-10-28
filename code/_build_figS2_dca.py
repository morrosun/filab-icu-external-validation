# -*- coding: utf-8 -*-
"""Supplementary Figure S2: decision curve analysis (MIMIC-IV, apparent/in-sample)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

FIG = "figures/revision"
ANA = "analysis"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.8})

a = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
r = pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv")
a["stay_id"] = a["stay_id"].astype(str); r["stay_id"] = r["stay_id"].astype(str)
d = a.merge(r[["stay_id", "sofa", "oasis", "charlson", "vent", "vaso", "rrt"]],
            on="stay_id", how="left", suffixes=("", "_rev"))
d["fi10"] = d["fi_lab"] * 10
cols = ["died", "fi10", "age", "male", "charlson", "sofa", "oasis", "vent", "vaso", "rrt"]
d = d.dropna(subset=cols).copy()
print("DCA cohort n=%d deaths=%d" % (len(d), int(d.died.sum())))

base = "age + male + charlson + sofa + oasis + vent + vaso + rrt"
p_base = smf.logit(f"died ~ {base}", data=d).fit(disp=0).predict(d)
p_full = smf.logit(f"died ~ {base} + fi10", data=d).fit(disp=0).predict(d)

def nb(y, p, t):
    y = np.asarray(y, float); p = np.asarray(p, float)
    tp = ((p >= t) & (y == 1)).sum() / len(y)
    fp = ((p >= t) & (y == 0)).sum() / len(y)
    return tp - fp * t / (1 - t)

thr = np.arange(0.02, 0.751, 0.005)
y = d["died"].values
nb_b = np.array([nb(y, p_base.values, t) for t in thr])
nb_f = np.array([nb(y, p_full.values, t) for t in thr])
prev = y.mean()
nb_all = prev - (1 - prev) * thr / (1 - thr)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(thr * 100, nb_all, color="#9E9E9E", lw=1.2, ls=":", label="Treat all")
ax.axhline(0, color="black", lw=1.0, ls="--", label="Treat none")
ax.plot(thr * 100, nb_b, color="#4C72B0", lw=1.8, label="Severity baseline")
ax.plot(thr * 100, np.maximum(nb_f, -0.02), color="#C44E52", lw=1.8, label="Baseline + FI-Lab")
ax.set_xlim(0, 60); ax.set_ylim(-0.005, max(0.06, float(nb_all[0]) * 1.05))
ax.set_xlabel("Threshold probability (%)")
ax.set_ylabel("Net benefit")
ax.legend(frameon=False, fontsize=8.2, loc="upper right")
ax.grid(alpha=0.25, lw=0.6)
ax.set_title("Decision curve analysis — in-hospital mortality, MIMIC-IV", fontsize=9.5)
ax.text(0.985, 0.055,
        "Apparent, in-sample estimates\n(no optimism correction)\nn = {:,}; deaths = {:,}".format(len(d), int(y.sum())),
        transform=ax.transAxes, fontsize=7.6, va="bottom", ha="right", color="#333333")

fig.tight_layout()
fig.savefig(f"{FIG}/figS2_dca.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
fig.savefig(f"{FIG}/figS2_dca.pdf", bbox_inches="tight", pad_inches=0.12)
plt.close(fig)

# 存档数值（5%–10% 窗口的净获益差）
sel = (thr >= 0.05) & (thr <= 0.10)
out = pd.DataFrame(dict(threshold=thr, nb_base=nb_b, nb_full=nb_f, nb_treat_all=nb_all))
out.to_csv(f"{ANA}/db_audit/dca_mimic.csv", index=False)
print("net benefit diff at 5%%: %.5f ; at 7.5%%: %.5f ; at 10%%: %.5f ; max in window: %.5f" % (
    nb_f[np.argmin(abs(thr - 0.05))] - nb_b[np.argmin(abs(thr - 0.05))],
    nb_f[np.argmin(abs(thr - 0.075))] - nb_b[np.argmin(abs(thr - 0.075))],
    nb_f[np.argmin(abs(thr - 0.10))] - nb_b[np.argmin(abs(thr - 0.10))],
    float((nb_f - nb_b)[sel].max())))
print("saved figS2_dca.png/.pdf and dca_mimic.csv")
