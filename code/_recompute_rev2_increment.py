# -*- coding: utf-8 -*-
"""增量价值 ΔAUC / Brier 差的 bootstrap 95%CI (P: 结果表应补充95%CI)"""
import os, warnings, numpy as np, pandas as pd
import psycopg2, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

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


warnings.filterwarnings("ignore"); np.random.seed(42)
ANA = os.path.join(ROOT, 'analysis'); OUT = os.path.join(ROOT, 'analysis', 'revision')
CONN = dict(**_DB_AUTH)
def q(db, sql):
    c = psycopg2.connect(dbname=db, **CONN)
    try: return pd.read_sql(sql, c)
    finally: c.close()

mim = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
sids = tuple(int(x) for x in mim["stay_id"].unique())
sofa = q("mimiciv", f"SELECT stay_id, sofa FROM mimiciv_derived.first_day_sofa WHERE stay_id IN {sids}")
oasis = q("mimiciv", f"SELECT stay_id, oasis FROM mimiciv_derived.oasis WHERE stay_id IN {sids}")
hids = tuple(int(x) for x in mim["hadm_id"].unique())
chl = q("mimiciv", f"SELECT hadm_id, charlson_comorbidity_index AS cci, rheumatic_disease FROM mimiciv_derived.charlson WHERE hadm_id IN {hids}")
chl["cci_excl_rheum"] = chl["cci"] - chl["rheumatic_disease"].fillna(0)
vent = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS vent FROM mimiciv_derived.ventilation WHERE stay_id IN {sids} AND ventilation_status IN ('InvasiveVent','Tracheostomy')")
vaso = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS vaso FROM mimiciv_derived.vasoactive_agent WHERE stay_id IN {sids}")
rrt = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS rrt FROM mimiciv_derived.rrt WHERE stay_id IN {sids}")
for tbl, col in [(sofa,"sofa"),(oasis,"oasis")]:
    mim[col] = mim["stay_id"].map(tbl.groupby("stay_id")[col].max())
mim["cci_excl_rheum"] = mim["hadm_id"].map(chl.set_index("hadm_id")["cci_excl_rheum"])
for tbl, col in [(vent,"vent"),(vaso,"vaso"),(rrt,"rrt")]:
    mim[col] = mim["stay_id"].map(tbl.set_index("stay_id")[col]).fillna(0).astype(int)
mim["fi10"] = mim["fi_lab"]*10
mc = mim.dropna(subset=["sofa","oasis","cci_excl_rheum"]).copy()
print("MIMIC complete-case w/ severity:", len(mc))

base_f = "died ~ age + male + ard + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
full_f = base_f + " + fi10"
m_base = smf.logit(base_f, data=mc).fit(disp=0)
m_full = smf.logit(full_f, data=mc).fit(disp=0)
auc_b = roc_auc_score(mc["died"], m_base.predict(mc))
auc_f = roc_auc_score(mc["died"], m_full.predict(mc))
brier_b = brier_score_loss(mc["died"], m_base.predict(mc))
brier_f = brier_score_loss(mc["died"], m_full.predict(mc))
print(f"point: AUC {auc_b:.4f}->{auc_f:.4f} Δ{auc_f-auc_b:+.4f}; Brier {brier_b:.4f}->{brier_f:.4f}")

y = mc["died"].values.astype(int); pb = m_base.predict(mc).values; pf = m_full.predict(mc).values
def boot():
    idx = np.random.randint(0, len(y), len(y))
    try:
        a1 = roc_auc_score(y[idx], pb[idx]); a2 = roc_auc_score(y[idx], pf[idx])
        b1 = brier_score_loss(y[idx], pb[idx]); b2 = brier_score_loss(y[idx], pf[idx])
        return (a1, a2, b1, b2)
    except Exception: return None
B = 1000; res = [r for r in (boot() for _ in range(B)) if r]
ab = np.array([r[0] for r in res]); af = np.array([r[1] for r in res])
bb = np.array([r[2] for r in res]); bf = np.array([r[3] for r in res])
d_auc = af - ab; d_br = bf - bb
out = dict(delta_auc=round(auc_f-auc_b,4), delta_auc_lo=round(np.percentile(d_auc,2.5),4),
           delta_auc_hi=round(np.percentile(d_auc,97.5),4),
           brier_diff=round(brier_f-brier_b,4), brier_lo=round(np.percentile(d_br,2.5),4),
           brier_hi=round(np.percentile(d_br,97.5),4))
print("bootstrap 95%CI:", out)
pd.DataFrame([out]).to_csv(f"{OUT}/rev2_increment_ci.csv", index=False)
print("DONE -> rev2_increment_ci.csv")

# 表4 绝对值 95%CI（基准模型 vs 基准+FI）
def ci(v): return (round(np.percentile(v,2.5),4), round(np.percentile(v,97.5),4))
t4 = []
alo, ahi = ci(ab); t4.append(dict(model="严重度基准模型", metric="AUC", value=round(auc_b,3), lo=alo, hi=ahi))
alo, ahi = ci(af); t4.append(dict(model="基准+FI-Lab", metric="AUC", value=round(auc_f,3), lo=alo, hi=ahi))
blo, bhi = ci(bb); t4.append(dict(model="严重度基准模型", metric="Brier", value=round(brier_b,4), lo=blo, hi=bhi))
blo, bhi = ci(bf); t4.append(dict(model="基准+FI-Lab", metric="Brier", value=round(brier_f,4), lo=blo, hi=bhi))
t4df = pd.DataFrame(t4)
t4df.to_csv(f"{OUT}/rev2_table4_ci.csv", index=False)
print("\n=== 表4 绝对值 95%CI ===")
print(t4df.to_string(index=False))
print("DONE -> rev2_table4_ci.csv")
