# -*- coding: utf-8 -*-
"""
方向 B 选项 2：扩展结局 —— 30 天死亡 + 30 天再入院
数据可得性：
  MIMIC-IV / NWICU : patients.dod 可用(区间保留) -> 30天死亡可算；admissions 多行 -> 30天再入院可算
  eICU             : 无 dod、无绝对日期 -> 仅院内死亡(沿用既有结果)
产出：analysis/outcomes_{mimiciv,nwicu}.csv, analysis/results_outcomes.csv, figures/fig6_*.png, fig7/fig8
"""
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.labelsize": 12,
    "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": 300,
})
C_ARD="#C0392B"; C_NON="#2E86C1"
DBS=["mimiciv","nwicu"]; DBLAB={"mimiciv":"MIMIC-IV","nwicu":"NWICU"}

def conn(db):
    return psycopg2.connect(**_DB_AUTH,dbname=db)

def get_outcomes(db):
    c=conn(db)
    if db=="mimiciv":
        A,P="mimiciv_hosp","mimiciv_hosp"
    else:
        A,P="hosp","hosp"
    sql=f"""
    WITH base AS (
      SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime,
             p.dod
      FROM {A}.admissions a
      JOIN {P}.patients p ON a.subject_id = p.subject_id
    ),
    nxt AS (
      SELECT hadm_id, dod, admittime, dischtime,
        LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS next_adm
      FROM base
    )
    SELECT hadm_id,
      CASE WHEN dod IS NOT NULL AND dod >= admittime
                AND (dod - admittime) <= interval '30 day' THEN 1 ELSE 0 END AS mort30,
      CASE WHEN next_adm IS NOT NULL
                AND next_adm > dischtime
                AND next_adm <= dischtime + interval '30 day' THEN 1 ELSE 0 END AS readmit30
    FROM nxt
    """
    o=pd.read_sql_query(sql,c); c.close()
    return o

# ---------- 载入既有合并队列(含 fi_lab/age/male/ard) ----------
ana={}
for db in ["mimiciv","nwicu","eicu"]:
    d=pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{db}.csv'))
    d=d.dropna(subset=["fi_lab"]).copy(); d["male"]=d["male"].astype(float)
    ana[db]=d

# ---------- 附加 30 天结局到 MIMIC / NWICU ----------
out_rows=[]
for db in DBS:
    o=get_outcomes(db)
    d=ana[db].merge(o,on="hadm_id",how="left")
    d["mort30"]=d["mort30"].fillna(0).astype(int)
    d["readmit30"]=d["readmit30"].fillna(0).astype(int)
    d[["hadm_id","subject_id","stay_id","age","male","ard","fi_lab","died","mort30","readmit30"]]\
      .to_csv(os.path.join(ROOT, 'analysis', f'outcomes_{db}.csv'),index=False)
    print(f"{db}: n={len(d):,} inhosp={d.died.mean()*100:.1f}% 30d_mort={d.mort30.mean()*100:.1f}% 30d_readmit={d.readmit30.mean()*100:.1f}%")
    ana[db]=d

# ---------- 拟合：每 0.1 FI 的 OR（三结局） ----------
def fit(d, y):
    m=smf.logit(f"{y} ~ ard*fi_lab + age + male", d).fit(disp=0)
    b=m.params["fi_lab"]; se=m.bse["fi_lab"]
    return dict(OR=round(np.exp(0.1*b),3), lo=round(np.exp(0.1*(b-1.96*se)),3),
                hi=round(np.exp(0.1*(b+1.96*se)),3), p=f"{m.pvalues['fi_lab']:.1e}")

# eICU 仅有院内死亡
res=[]
for db in DBS:
    for y,lab in [("died","In-hospital mortality"),("mort30","30-day mortality"),("readmit30","30-day readmission")]:
        r=fit(ana[db],y); res.append(dict(database=DBLAB[db],outcome=lab,**r))
# MIMIC + Charlson 敏感性(院内 & 30d mort)
dm=ana["mimiciv"].dropna(subset=["charlson"]).copy(); dm["charlson"]=dm["charlson"].astype(float)
for y,lab in [("died","In-hospital mortality"),("mort30","30-day mortality")]:
    m=smf.logit(f"{y} ~ ard*fi_lab + age + male + charlson", dm).fit(disp=0)
    b=m.params["fi_lab"]; se=m.bse["fi_lab"]
    res.append(dict(database="MIMIC-IV (+Charlson)",outcome=lab,
                     OR=round(np.exp(0.1*b),3),lo=round(np.exp(0.1*(b-1.96*se)),3),
                     hi=round(np.exp(0.1*(b+1.96*se)),3),p=f"{m.pvalues['fi_lab']:.1e}"))
# eICU 院内死亡(既有)
r=fit(ana["eicu"],"died"); res.append(dict(database="eICU",outcome="In-hospital mortality",**r))
out_tbl=pd.DataFrame(res)
out_tbl.to_csv(os.path.join(ROOT, 'analysis', 'results_outcomes.csv'),index=False)
print("\n=== results_outcomes.csv ===")
print(out_tbl.to_string(index=False))

# ============================================================
# 图6：森林图 — 每 0.1 FI 的 OR（三结局 × 各库）
# ============================================================
plot_df=out_tbl.copy()
order=["In-hospital mortality","30-day mortality","30-day readmission"]
plot_df["outcome"]=pd.Categorical(plot_df["outcome"],order,ordered=True)
plot_df=plot_df.sort_values(["outcome","database"])
yp=np.arange(len(plot_df))[::-1]
fig,ax=plt.subplots(figsize=(9,5.2))
for i,(_,r) in enumerate(plot_df.iterrows()):
    ax.plot([r.lo,r.hi],[yp[i],yp[i]],color="#34495E",lw=1.5)
    ax.plot([r.OR],[yp[i]],"o",color="#C0392B",ms=7)
    ax.annotate(f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}]",xy=(r.hi,yp[i]),
                xytext=(r.hi+0.03,yp[i]),va="center",fontsize=9)
ax.axvline(1.0,color="gray",ls="--",lw=1)
ax.set_yticks(yp); ax.set_yticklabels([f"{r.database}\n{r.outcome}" for _,r in plot_df.iterrows()])
ax.set_xlabel("Odds ratio for adverse outcome per 0.1 increase in FI-Lab (95% CI)")
ax.set_title("Figure 6. FI-Lab association with in-hospital, 30-day mortality and 30-day readmission")
ax.set_xlim(1.0,2.1)
ax.grid(axis="x",alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_forest_outcomes.png"); plt.close(fig)

# ============================================================
# 图7 / 图8：30天死亡 / 30天再入院 按 FI 四分位（ARD 分层，MIMIC+NWICU）
# ============================================================
def quartile_panel(db, y, title, fname):
    d=ana[db]
    q=pd.qcut(d["fi_lab"],4,labels=["Q1","Q2","Q3","Q4"]); d=d.assign(q=q)
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    rates=[]
    for lv in ["Q1","Q2","Q3","Q4"]:
        sub=d[d.q==lv]
        rates.append((sub.loc[sub.ard==1,y].mean()*100, sub.loc[sub.ard==0,y].mean()*100))
    rates=np.array(rates); x=np.arange(4); w=0.38
    ax.bar(x-w/2,rates[:,0],w,color=C_ARD,alpha=0.85,label="ARD")
    ax.bar(x+w/2,rates[:,1],w,color=C_NON,alpha=0.85,label="Non-ARD")
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n(least)","Q2","Q3","Q4\n(most)"])
    ax.set_xlabel("FI-Lab quartile")
    ax.set_ylabel(f"{title} (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title(f"{DBLAB[db]}")
    ax.legend(frameon=False,loc="upper left")
    ax.grid(axis="y",alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

for db in DBS:
    quartile_panel(db,"mort30","30-day mortality","fig7_30dmort_quartile.png" if db=="mimiciv" else "fig7b_30dmort_quartile.png")
    quartile_panel(db,"readmit30","30-day readmission","fig8_30dreadmit_quartile.png" if db=="mimiciv" else "fig8b_30dreadmit_quartile.png")
print("\nAll outcome figures saved to",OUT)
