# -*- coding: utf-8 -*-
"""Figure 1 (STROBE flow) — 重绘版 v2（第八轮）
三层：老年首次 ICU 住院 -> FI-Lab 可评估/不可评估 -> 院内死亡分析（NWICU 另列 3 例结局不一致）
计数来自 analysis/analysis_{db}.csv。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

FIG = "figures/revision"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.linewidth": 0.8})

DB = ["mimiciv", "nwicu", "eicu"]
LAB = {"mimiciv": "MIMIC-IV", "nwicu": "NWICU", "eicu": "eICU"}
NW_UNCERTAIN = 3

rows = {}
for db in DB:
    d = pd.read_csv(f"analysis/analysis_{db}.csv")
    tot = len(d); ev = int(d["fi_lab"].notna().sum())
    rows[db] = dict(total=tot, ev=ev, ex=tot - ev, analysed=ev)
rows["nwicu"]["analysed"] = rows["nwicu"]["ev"] - NW_UNCERTAIN

fig, ax = plt.subplots(figsize=(10.6, 7.6))
ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(1.5, 8.62)

cols = [dict(xL=0.35, xR=3.35, cx=1.85),
        dict(xL=3.90, xR=6.90, cx=5.40),
        dict(xL=7.45, xR=10.45, cx=8.95)]

def box(x, y, w, h, text, fc, fs=8.6, dashed=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                           linewidth=1.0, linestyle="--" if dashed else "-"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.32)

def arrow(x0, y0, x1, y1, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0,
                                linestyle="--" if dashed else "-"))

Y1, H1 = 7.42, 1.02
Y2, H2 = 5.18, 1.16
Y3, H3 = 3.05, 1.02
YF     = 6.86     # 分叉点（位于层1 与层2 之间）

for db, c in zip(DB, cols):
    r = rows[db]
    xL, xR, cx, W = c["xL"], c["xR"], c["cx"], c["xR"] - c["xL"]
    bw = W/2 - 0.12
    ev_p = r["ev"]/r["total"]*100
    ex_p = r["ex"]/r["total"]*100

    box(xL, Y1, W, H1,
        f"{LAB[db]}\nolder adults (≥65): first ICU stay\nof each hospitalization\nn = {r['total']:,}",
        "#DCE6F1", fs=8.8)

    box(xL, Y2, bw, H2, f"FI-Lab\nevaluable\n(≥15/30 items)\nn = {r['ev']:,}\n({ev_p:.1f}%)", "#E2EFDA", fs=8.0)
    box(xR - bw, Y2, bw, H2, f"FI-Lab not\nevaluable\n(<15/30 items)\nn = {r['ex']:,}\n({ex_p:.1f}%)", "#F2DCDB", fs=8.0)

    arrow(cx, Y1, cx, YF)
    arrow(cx, YF, xL + bw/2, Y2 + H2)
    arrow(cx, YF, xR - bw/2, Y2 + H2)

    if db == "nwicu":
        box(xL, Y3, bw, H3,
            f"In-hospital\nmortality analysed\nn = {r['analysed']:,}\n({r['analysed']/r['total']*100:.1f}%)",
            "#DEEAF6", fs=7.9)
        box(xR - bw, Y3, bw, H3,
            "Excluded from\nall outcome\nanalyses\nn = 3\n(death date\nbefore admission)",
            "white", fs=7.2, dashed=True)
        arrow(xL + bw/2, Y2, xL + bw/2, Y3 + H3)
        arrow(xL + bw, Y3 + H3/2, xR - bw, Y3 + H3/2, dashed=True)
    else:
        box(xL, Y3, W, H3,
            f"In-hospital mortality analysed\nn = {r['analysed']:,} ({r['analysed']/r['total']*100:.1f}%)",
            "#DEEAF6", fs=8.6)
        arrow(cx, Y2, cx, Y3 + H3)

ax.text(0.35, 2.34,
        "FI-Lab was computed over the first ICU day (MIMIC-IV: up to 6 h before to 24 h after ICU admission; "
        "NWICU and eICU: 0–24 h). Patients not evaluable for FI-Lab\n"
        "(<15 of 30 items assessed) were excluded from all analyses, so the FI-Lab–evaluable cohort is the analysis cohort of every database.\n"
        "In NWICU, three hospitalizations had a recorded death date preceding the admission date; the in-hospital outcome was treated as\n"
        "unknown and these records were excluded from all NWICU outcome analyses (Methods 2.2). No records were excluded for this reason\n"
        "in MIMIC-IV or eICU.",
        ha="left", va="top", fontsize=7.9, color="#1a1a1a", linespacing=1.55)

fig.savefig(f"{FIG}/fig1_flow.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
fig.savefig(f"{FIG}/fig1_flow.pdf", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("saved fig1 (v2)")
