# -*- coding: utf-8 -*-
"""FI-Lab 外部验证计算脚本（Rockwood 缺陷累积法 + 冻结 M2 模型，与三库同款）。
用法：将本脚本与 FI_Lab_Validation_Template.csv 放在同一目录，运行：
      python compute_FI_from_template.py
输出：
  - FI_Lab_Computed.csv : 每位患者的 FI-Lab、缺陷数/评估数、M2 线性预测值 lp 与预测概率 p
  - calibration_M2.csv  : 10 分位校准表（预测概率 vs 观察死亡率），用于绘制校准曲线
屏幕打印：样本量、死亡事件数、AUC(Harrell's C)、校准斜率/截距、FI-Lab 四分位死亡率梯度。
依赖：仅 pandas + numpy（AUC/校准为无第三方库实现；若装了 scipy 会更稳）。
"""
import pandas as pd, numpy as np, os
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "FI_Lab_Validation_Template.csv")

# 30 项 FI-Lab 缺陷定义：code -> (正常下限, 正常上限)
ITEMS = {
    "albumin":(3.5,5.0),"bilirubin_total":(0.1,1.2),"bilirubin_direct":(0.0,0.3),
    "bun":(7.0,20.0),"calcium":(8.5,10.5),"chloride":(98.0,110.0),"creatinine":(0.5,1.3),
    "glucose":(70.0,140.0),"hematocrit":(36.0,50.0),"hemoglobin":(10.0,17.0),"lactate":(0.5,2.0),
    "magnesium":(1.5,2.5),"phosphate":(2.5,4.5),"platelets":(150.0,450.0),"potassium":(3.5,5.5),
    "sodium":(135.0,145.0),"bicarbonate":(22.0,29.0),"aniongap":(3.0,16.0),"alt":(7.0,40.0),
    "ast":(10.0,40.0),"alp":(44.0,130.0),"total_protein":(6.0,8.5),"pt":(9.0,14.0),"ptt":(25.0,40.0),
    "inr":(0.8,1.3),"fibrinogen":(150.0,450.0),"ldh":(100.0,250.0),"ck":(30.0,300.0),
    "rdw":(11.0,15.0),"mcv":(80.0,100.0),
}
MIN_THRESH = 15  # 至少评估 15/30 项方可计算 FI

# 冻结 M2 模型 (MIMIC-IV 表观队列拟合, 外部验证不重估任何参数)
# died ~ fi10 + age + male + ard ;  fi10 = FI-Lab x 10
B0, B_FI, B_AGE, B_MALE, B_ARD = -6.7545, 0.4955, 0.0327, -0.0361, -0.0067

def male_ind(s):
    if pd.isna(s): return np.nan
    return 1.0 if str(s).strip().lower() in ("男","m","male","1","true") else 0.0

df = pd.read_csv(CSV)
deficits = []
for code,(lo,hi) in ITEMS.items():
    v = pd.to_numeric(df[code], errors="coerce")
    assessed = v.notna()
    d = ((v < lo) | (v > hi)).fillna(False)
    deficits.append(d.where(assessed))
Dmat = pd.concat(deficits, axis=1)
n_assessed = Dmat.notna().sum(axis=1)
n_deficit = Dmat.sum(axis=1, min_count=1).fillna(0)
fi = np.where(n_assessed >= MIN_THRESH, n_deficit / n_assessed, np.nan)
df["n_assessed"] = n_assessed.astype(int)
df["n_deficits"] = n_deficit.astype(int)
df["FI_Lab"] = np.round(fi, 4)
df["fi10"] = df["FI_Lab"] * 10.0
df["male_ind"] = df["sex"].map(male_ind)

valid = df["FI_Lab"].notna() & df["age"].notna() & df["male_ind"].notna() & df["ard"].notna()
lp = np.full(len(df), np.nan); pp = np.full(len(df), np.nan)
fi10 = df["fi10"].values; age = df["age"].values
male = df["male_ind"].values; ard = df["ard"].values
lp_v = B0 + B_FI*fi10 + B_AGE*age + B_MALE*male + B_ARD*ard
pp_v = 1.0 / (1.0 + np.exp(-lp_v))
lp[valid.values] = lp_v[valid.values]; pp[valid.values] = pp_v[valid.values]
df["lp_M2"] = np.round(lp, 4); df["p_M2"] = np.round(pp, 4)
df.to_csv(os.path.join(HERE, "FI_Lab_Computed.csv"), index=False)

ok = df.dropna(subset=["FI_Lab"]).copy()
print(f"共 {len(df)} 行；可计算 FI-Lab {len(ok)} 行 (评估>={MIN_THRESH}项)。")
print(f"FI-Lab 中位数={ok['FI_Lab'].median():.3f}，均值={ok['FI_Lab'].mean():.3f}")

val = ok.dropna(subset=["lp_M2","inhosp_death"]).copy()
if len(val) and val["inhosp_death"].notna().any():
    y = val["inhosp_death"].astype(float).values
    p = val["p_M2"].values; lpv = val["lp_M2"].values
    # Harrell's C (AUC) —— 无第三方库实现（带结的平均秩次法）
    npos = int(y.sum()); nneg = len(y) - npos
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float); ranks[order] = np.arange(1, len(p)+1)
    _, inv = np.unique(p, return_inverse=True)
    counts = np.bincount(inv)
    cum = np.cumsum(counts); avg = cum - (counts - 1) / 2.0
    ranks = avg[inv]
    auc = float((ranks[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg))
    print(f"死亡事件={npos}；M2 模型 AUC(Harrell's C)={auc:.3f}")
    # 校准：died ~ lp 逻辑回归（numpy Newton-Raphson，自由截距+斜率；稳健化）
    if npos >= 2 and nneg >= 2:
        X = np.column_stack([np.ones_like(lpv), lpv]); b = np.zeros(2)
        try:
            for _ in range(100):
                eta = np.clip(X @ b, -30, 30); pr = 1.0 / (1.0 + np.exp(-eta))
                g = X.T @ (y - pr)
                H = X.T @ (X * (pr*(1-pr))[:, None]) + 1e-8 * np.eye(2)
                step = np.linalg.solve(H, g); b -= step
                if np.max(np.abs(step)) < 1e-6:
                    break
            cal_int, cal_slope = b
            print(f"校准：截距={cal_int:.3f} (理想 0)，斜率={cal_slope:.3f} (理想 1)")
        except Exception as e:
            print(f"校准斜率无法自动估计（事件过少或完全分离），请用统计软件(R/SPSS)正式评估：{e}")
    else:
        print("校准：事件数不足（需正/负类均≥2），跳过；请用统计软件正式评估。")
    # 10 分位校准表
    dec = pd.qcut(p, 10, duplicates="drop")
    cal = (val.assign(dec=dec).groupby("dec", observed=True)
           .agg(n=("inhosp_death","size"), pred=("p_M2","mean"), obs=("inhosp_death","mean")).reset_index())
    cal["pred"] = cal["pred"].round(4); cal["obs"] = cal["obs"].round(4)
    cal.to_csv(os.path.join(HERE, "calibration_M2.csv"), index=False)
    print("校准分位表 -> calibration_M2.csv")
    # FI-Lab 四分位院内死亡梯度
    q = pd.qcut(ok["FI_Lab"], 4, labels=["Q1","Q2","Q3","Q4"])
    g = ok.assign(q=q).groupby("q", observed=True)["inhosp_death"].agg(["size","mean"])
    print("FI-Lab 四分位院内死亡：")
    for idx, row in g.iterrows():
        print(f"  {idx}: n={int(row['size'])} 死亡率={row['mean']*100:.1f}%")
else:
    print("提示：填写 inhosp_death(0/1) 后重跑，可得到 AUC 与校准。")
