# -*- coding: utf-8 -*-
"""SF1 缺失率热图 —— 与补充表 S1 同一数据来源（per_lab_missingness_final.csv）"""
import numpy as np, pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5, "axes.linewidth": 1.0,
    "savefig.dpi": 300, "axes.unicode_minus": False,
})
SRC = os.path.join(ROOT, 'analysis', 'db_audit', 'per_lab_missingness_final.csv')
OUT = os.path.join(ROOT, 'figures', 'revision', 'figS1_missingness')

d = pd.read_csv(SRC).set_index("item")
cols = ["MIMIC-IV", "NWICU", "eICU"]
M = d[cols].values  # % missing

fig, ax = plt.subplots(figsize=(4.6, 8.2))
im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9.5)
ax.set_yticks(range(len(d.index))); ax.set_yticklabels(d.index, fontsize=8.2)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                color="white" if v > 55 else "black")
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(d.index), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Missing (%)", fontsize=9)
ax.set_title("Per-laboratory missingness\nin the FI-Lab–assessable cohort", fontsize=10)
fig.savefig(OUT + ".png", bbox_inches="tight", pad_inches=0.12)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.12)
print("saved", OUT, "| items:", len(d))
print(d.loc[["rdw", "mcv", "inr", "ld_ldh"]][cols].to_string())
