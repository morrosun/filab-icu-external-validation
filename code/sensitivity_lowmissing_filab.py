# -*- coding: utf-8 -*-
"""
敏感性分析（回应审稿 Issue 3）：排除高缺失变量（>70% 缺失）后的 FI-Lab 重新计算
  - 复用 build_analysis.py 的队列与化验提取逻辑（与正文全部结果同源），保证重算样本一致
  - 以 MIMIC-IV（推导队列）缺失率 >70% 为排除标准确定剔除项
  - 三库使用同一套剔除规则，重算 FI-Lab（剩余 26 项），保持跨库可比
  - 输出 filab/filab_reduced_{db}.csv（stay_id, fi_lab_reduced, n_assessed_reduced）
"""
import os, sys
import numpy as np, pandas as pd

ROOT = ROOT
sys.path.insert(0, ROOT)
import build_analysis as ba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



MISS = os.path.join(ROOT, "analysis", "revision", "per_lab_missingness.csv")
miss = pd.read_csv(MISS)
# 排除标准：MIMIC-IV 推导队列缺失率 >70%（与正文 S2 口径一致）
drop = miss.loc[miss["MIMIC-IV_miss%"] > 70, "item"].tolist()
RETAINED = [a for a in ba.ANALYTES if a not in drop]
print("Excluded (>70%% missing in MIMIC-IV derivation):", drop)
print(f"Retained {len(RETAINED)}/{len(ba.ANALYTES)} analytes across all 3 databases")

def compute_fi_subset(labs, subset):
    """与 build_analysis.compute_fi 同逻辑，仅在 subset 内累加缺陷/已评估。"""
    defs = []
    for a in subset:
        lo, hi = ba.D[a][3], ba.D[a][4]
        mn = labs.get(f"{a}_min"); mx = labs.get(f"{a}_max")
        assessed = mn.notna() | mx.notna()
        deficit = ((mn < lo) | (mx > hi)).fillna(False)
        defs.append(deficit.where(assessed))
    Dmat = pd.concat(defs, axis=1)
    an = Dmat.notna().sum(axis=1)
    dn = Dmat.sum(axis=1, min_count=1).fillna(0)
    fi = np.full(len(labs), np.nan)
    mask = an >= ba.MIN_THRESH
    fi[mask.values] = (dn[mask] / an[mask]).values
    return pd.Series(fi, index=labs.index), an

for db in ["mimiciv", "nwicu", "eicu"]:
    print(f"\n--- building cohort + labs for {db} ---")
    if db == "mimiciv":
        coh = ba.cohort_mimic_like("mimiciv", "mimiciv_hosp", "mimiciv_icu")
        ids = tuple(int(x) for x in coh["stay_id"])
        L = ba.fi_mimic(ids)
    elif db == "nwicu":
        coh = ba.cohort_mimic_like("nwicu", "hosp", "icu")
        ids = tuple(int(x) for x in coh["stay_id"])
        L = ba.fi_nwicu(ids)
    else:
        coh = ba.cohort_eicu()
        ids = tuple(int(x) for x in coh["stay_id"])
        L = ba.fi_eicu(ids)
    fi, nass = compute_fi_subset(L, RETAINED)
    out = pd.DataFrame({
        "stay_id": L["stay_id"].astype(int),
        "fi_lab_reduced": fi.values,
        "n_assessed_reduced": nass.values,
    })
    out.to_csv(os.path.join(ROOT, "filab", f"filab_reduced_{db}.csv"), index=False)
    d = out.dropna(subset=["fi_lab_reduced"]).copy()
    qs = d["fi_lab_reduced"].quantile([0.25, 0.5, 0.75]).round(3).tolist()
    print(f"{db}: n_fi={len(d):,}  FI median={qs[1]}  n_assessed median={int(d.n_assessed_reduced.median())}  "
          f"Q1/Q4={qs[0]}/{qs[2]}")
print("\nReduced FI-Lab CSVs written to filab/")
