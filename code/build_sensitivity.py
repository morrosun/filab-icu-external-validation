# -*- coding: utf-8 -*-
"""
方向 B 计划 3：敏感性 / 分层分析
  (1) 按年代分层的 FI-Lab 效应（跨年代稳健性）
  (2) 剔除 COVID 后重估 FI-Lab 效应（MIMIC/NWICU；eICU 2014-15 天然无 COVID）
  (3) 按 ARD 病种亚组(SLE / RA / 血管炎)分层的 FI-死亡关联
数据源：analysis/analysis_{mimiciv,nwicu,eicu}.csv（含 fi_lab/died/ard/age/male/hadm_id/stay_id/subject_id）
标签补充：从 PostgreSQL 查 COVID(U07.1)、病种亚组(ICD)、年代(anchor_year_group / hospitaldischargeyear)
产出：analysis/results_sensitivity_{era,covid,subtype}.csv + figures/fig9/fig10/fig11
"""
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

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



OUT = os.path.join(ROOT, 'figures')
ANA = os.path.join(ROOT, 'analysis')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.labelsize": 12,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": 300,
})
C1 = "#C0392B"; C2 = "#2E86C1"; C3 = "#27AE60"

def conn(db):
    return psycopg2.connect(**_DB_AUTH, dbname=db)

# ---------- 载入既有队列 ----------
ana = {}
for db in ["mimiciv", "nwicu", "eicu"]:
    d = pd.read_csv(f"{ANA}/analysis_{db}.csv")
    d = d.dropna(subset=["fi_lab"]).copy()
    d["male"] = d["male"].astype(float)
    ana[db] = d

# ---------- 病种 ICD 正则(ICD-9 前缀 / ICD-10 前缀) ----------
SUBTYPES = {
    "SLE":        (r"^7100",             r"^M32"),
    "RA":         (r"^(7140|7141|7142)", r"^(M05|M06)"),
    "Vasculitis": (r"^(446|4471|4476)",  r"^(M30|M31)"),
}

def label_mimic_like(db, hosp_schema):
    d = ana[db]
    all_ids = tuple(int(x) for x in d["hadm_id"].unique())
    ard_ids = tuple(int(x) for x in d.loc[d.ard == 1, "hadm_id"].unique())
    c = conn(db)
    cov = pd.read_sql_query(
        f"SELECT DISTINCT hadm_id FROM {hosp_schema}.diagnoses_icd "
        f"WHERE icd_version=10 AND icd_code LIKE 'U071%' AND hadm_id IN {all_ids}", c)
    d["covid"] = d["hadm_id"].isin(cov["hadm_id"]).astype(int)
    dx = pd.read_sql_query(
        f"SELECT hadm_id, icd_version, icd_code FROM {hosp_schema}.diagnoses_icd "
        f"WHERE hadm_id IN {ard_ids}", c)
    yg = pd.read_sql_query(
        f"SELECT subject_id, anchor_year_group FROM {hosp_schema}.patients "
        f"WHERE subject_id IN {tuple(int(x) for x in d['subject_id'].unique())}", c)
    c.close()
    for name, (r9, r10) in SUBTYPES.items():
        hit = dx[((dx.icd_version == 9) & dx.icd_code.str.match(r9)) |
                 ((dx.icd_version == 10) & dx.icd_code.str.match(r10))]["hadm_id"].unique()
        d["sub_" + name] = ((d.ard == 1) & (d.hadm_id.isin(hit))).astype(int)
    d = d.merge(yg, on="subject_id", how="left")
    ana[db] = d

def label_eicu():
    d = ana["eicu"]
    ard_ids = tuple(int(x) for x in d.loc[d.ard == 1, "stay_id"].unique())
    c = conn("eicu")
    dx = pd.read_sql_query(
        f"SELECT patientunitstayid AS stay_id, icd9code FROM eicu_crd.diagnosis "
        f"WHERE patientunitstayid IN {ard_ids} AND icd9code IS NOT NULL", c)
    yr = pd.read_sql_query(
        f"SELECT patientunitstayid AS stay_id, hospitaldischargeyear FROM eicu_crd.patient "
        f"WHERE patientunitstayid IN {tuple(int(x) for x in d['stay_id'].unique())}", c)
    c.close()
    d["covid"] = 0  # eICU 2014-15 无 COVID
    dxmap = {}
    for sid, g in dx.groupby("stay_id"):
        dxmap[sid] = ",".join(g["icd9code"].astype(str))
    codeser = d["stay_id"].map(dxmap).fillna("")
    d["sub_SLE"]        = ((d.ard == 1) & codeser.str.contains(r"710\.0|M32")).astype(int)
    d["sub_RA"]         = ((d.ard == 1) & codeser.str.contains(r"714\.[012]|M0[56]")).astype(int)
    d["sub_Vasculitis"] = ((d.ard == 1) & codeser.str.contains(r"446|447\.6|M3[01]")).astype(int)
    d = d.merge(yr, on="stay_id", how="left")
    d["anchor_year_group"] = d["hospitaldischargeyear"].astype("Int64").astype(str)
    ana["eicu"] = d

label_mimic_like("mimiciv", "mimiciv_hosp")
label_mimic_like("nwicu", "hosp")
label_eicu()

DBLAB = {"mimiciv": "MIMIC-IV", "nwicu": "NWICU", "eicu": "eICU"}
print("=== 标签补充完成 ===")
for db in ["mimiciv", "nwicu", "eicu"]:
    d = ana[db]
    print(f"{db}: n={len(d):,} COVID={int(d.covid.sum())} "
          f"SLE={int(d.sub_SLE.sum())} RA={int(d.sub_RA.sum())} Vasc={int(d.sub_Vasculitis.sum())}")

# ---------- 拟合工具：每 0.1 FI 的 OR ----------
def fit_fi(d, formula):
    try:
        m = smf.logit(formula, d).fit(disp=0)
        if "fi_lab" not in m.params:
            return dict(OR=np.nan, lo=np.nan, hi=np.nan, p="no_fi")
        b = m.params["fi_lab"]; se = m.bse["fi_lab"]
        return dict(OR=round(np.exp(0.1 * b), 3),
                    lo=round(np.exp(0.1 * (b - 1.96 * se)), 3),
                    hi=round(np.exp(0.1 * (b + 1.96 * se)), 3),
                    p=f"{m.pvalues['fi_lab']:.1e}")
    except Exception as e:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=f"fail:{type(e).__name__}")

# ============================================================
# (1) 按年代分层（全队列，院内死亡；died ~ fi_lab + age + male）
# ============================================================
era_rows = []
era_order = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022", "2014", "2015"]
for db in ["mimiciv", "nwicu", "eicu"]:
    d = ana[db]
    for g in [x for x in era_order if x in set(d["anchor_year_group"].dropna())]:
        sub = d[d["anchor_year_group"] == g]
        if len(sub) < 50 or sub["died"].sum() < 5:
            era_rows.append(dict(database=DBLAB[db], era=g, n=len(sub), deaths=int(sub.died.sum()),
                                 OR=np.nan, lo=np.nan, hi=np.nan, p="n<50 or events<5"))
            continue
        r = fit_fi(sub, "died ~ fi_lab + age + male")
        era_rows.append(dict(database=DBLAB[db], era=g, n=len(sub), deaths=int(sub.died.sum()), **r))
era_tbl = pd.DataFrame(era_rows)
era_tbl.to_csv(f"{ANA}/results_sensitivity_era.csv", index=False)
print("\n=== (1) 年代分层 (每0.1 FI 院内死亡 OR) ===")
print(era_tbl.to_string(index=False))

# ============================================================
# (2) 剔除 COVID 敏感性（MIMIC/NWICU；died ~ ard*fi_lab + age + male）
# ============================================================
cov_rows = []
for db in ["mimiciv", "nwicu"]:
    d = ana[db]
    for tag, sub in [("All", d), ("Excl. COVID", d[d.covid == 0])]:
        r = fit_fi(sub, "died ~ ard*fi_lab + age + male")
        cov_rows.append(dict(database=DBLAB[db], cohort=tag, n=len(sub),
                             covid_n=int(d.covid.sum()) if tag == "All" else 0,
                             deaths=int(sub.died.sum()), **r))
r = fit_fi(ana["eicu"], "died ~ ard*fi_lab + age + male")
cov_rows.append(dict(database="eICU (no COVID era)", cohort="All", n=len(ana["eicu"]),
                     covid_n=0, deaths=int(ana["eicu"].died.sum()), **r))
cov_tbl = pd.DataFrame(cov_rows)
cov_tbl.to_csv(f"{ANA}/results_sensitivity_covid.csv", index=False)
print("\n=== (2) 剔除 COVID 敏感性 (每0.1 FI 死亡 OR) ===")
print(cov_tbl.to_string(index=False))

# ============================================================
# (3) ARD 病种亚组分层（ARD 内，院内死亡；died ~ fi_lab + age + male）
# ============================================================
sub_rows = []
for db in ["mimiciv", "nwicu", "eicu"]:
    d = ana[db]
    ard = d[d.ard == 1]
    groups = [("All ARD", ard)]
    for name in ["SLE", "RA", "Vasculitis"]:
        groups.append((name, d[d["sub_" + name] == 1]))
    for name, sub in groups:
        if len(sub) < 30 or sub["died"].sum() < 5:
            sub_rows.append(dict(database=DBLAB[db], subtype=name, n=len(sub), deaths=int(sub.died.sum()),
                                 OR=np.nan, lo=np.nan, hi=np.nan, p="n<30 or events<5"))
            continue
        r = fit_fi(sub, "died ~ fi_lab + age + male")
        sub_rows.append(dict(database=DBLAB[db], subtype=name, n=len(sub), deaths=int(sub.died.sum()), **r))
sub_tbl = pd.DataFrame(sub_rows)
sub_tbl.to_csv(f"{ANA}/results_sensitivity_subtype.csv", index=False)
print("\n=== (3) ARD 病种亚组 (每0.1 FI 院内死亡 OR) ===")
print(sub_tbl.to_string(index=False))

# ============================================================
# 通用森林图
# ============================================================
def forest(df, ylabels, title, fname, xlim=(0.8, 2.4), color=C1):
    df = df.reset_index(drop=True)
    yp = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 2))
    for i, (_, r) in enumerate(df.iterrows()):
        if pd.notna(r.OR):
            ax.plot([r.lo, r.hi], [yp[i], yp[i]], color="#34495E", lw=1.5)
            ax.plot([r.OR], [yp[i]], "o", color=color, ms=7)
            ax.annotate(f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]", xy=(r.hi, yp[i]),
                        xytext=(min(r.hi + 0.03, xlim[1] - 0.02), yp[i]), va="center", fontsize=9)
        else:
            ax.annotate("(insufficient sample)", xy=(xlim[0] + 0.05, yp[i]),
                        va="center", fontsize=8, color="gray")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(yp); ax.set_yticklabels(ylabels)
    ax.set_xlabel("Odds ratio for in-hospital death per 0.1 increase in FI-Lab (95% CI)")
    ax.set_title(title); ax.set_xlim(*xlim); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)

# 图9：年代分层森林图
e = era_tbl.copy()
forest(e, [f"{r.database} | {r.era}\n(n={r.n:,}, d={r.deaths})" for _, r in e.iterrows()],
       "Figure 9. FI-Lab and in-hospital mortality across admission eras (three databases)",
       f"{OUT}/fig9_era_forest.png", xlim=(0.8, 2.4), color=C2)

# 图10：COVID 敏感性对比
fig, ax = plt.subplots(figsize=(8, 4.2))
cc = cov_tbl[cov_tbl.database.isin(["MIMIC-IV", "NWICU"])].copy()
labels = []; k = 0
for db in ["MIMIC-IV", "NWICU"]:
    for tag, off, col in [("All", 0.16, C1), ("Excl. COVID", -0.16, C2)]:
        row = cc[(cc.database == db) & (cc.cohort == tag)].iloc[0]
        y = k + off
        ax.plot([row.lo, row.hi], [y, y], color="#34495E", lw=1.5)
        ax.plot([row.OR], [y], "o", color=col, ms=8, label=tag if k == 0 else "")
        ax.annotate(f"{row.OR:.2f}", xy=(row.hi, y), xytext=(row.hi + 0.02, y), va="center", fontsize=9)
    labels.append(db); k += 1
ax.axvline(1.0, color="gray", ls="--", lw=1)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("OR for death per 0.1 increase in FI-Lab (95% CI)")
ax.set_title("Figure 10. FI-Lab effect is robust to exclusion of COVID-19 admissions")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_covid_sensitivity.png"); plt.close(fig)

# 图11：ARD 病种亚组森林图
s = sub_tbl.copy()
forest(s, [f"{r.database} | {r.subtype}\n(n={r.n:,}, d={r.deaths})" for _, r in s.iterrows()],
       "Figure 11. FI-Lab and in-hospital mortality within ARD subtypes",
       f"{OUT}/fig11_subtype_forest.png", xlim=(0.6, 3.2), color=C1)

print("\nAll sensitivity figures saved to", OUT)
