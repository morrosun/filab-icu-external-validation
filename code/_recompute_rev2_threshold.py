# -*- coding: utf-8 -*-
"""
P6 阈值敏感性：从老年首次ICU原始母队列重新定义 FI-Lab 可评估标准，
分别构建 已评估项>=10 / >=15 / >=20 三个队列，比较 FI-Lab 每0.1 的 OR。
（原表7的 >=10 行与 >=15 完全相同，是因主队列已要求>=15，>=10 不可能再增加人。）
重写：用 first_day_lab + labevents pivot 抽取每项的 min/max，按阈值重算 FI。
"""
import os, warnings, numpy as np, pandas as pd
import psycopg2, statsmodels.api as sm, statsmodels.formula.api as smf

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


warnings.filterwarnings("ignore")
ANA = os.path.join(ROOT, 'analysis'); OUT = os.path.join(ROOT, 'analysis', 'revision')
CONN = dict(**_DB_AUTH)
def conn(db): return psycopg2.connect(dbname=db, **CONN)

D = {
 "albumin":("albumin",3.5,5.0),"bilirubin_tot":("bilirubin_total",0.1,1.2),
 "bilirubin_dir":("bilirubin_direct",0.0,0.3),"bun":("bun",7.0,20.0),
 "calcium":("calcium",8.5,10.5),"chloride":("chloride",98.0,110.0),
 "creatinine":("creatinine",0.5,1.3),"glucose":("glucose",70.0,140.0),
 "hematocrit":("hematocrit",36.0,50.0),"hemoglobin":("hemoglobin",10.0,17.0),
 "lactate":(None,0.5,2.0),"magnesium":(None,1.5,2.5),"phosphate":(None,2.5,4.5),
 "platelet":("platelets",150.0,450.0),"potassium":("potassium",3.5,5.5),
 "sodium":("sodium",135.0,145.0),"bicarbonate":("bicarbonate",22.0,29.0),
 "aniongap":("aniongap",3.0,16.0),"alt":("alt",7.0,40.0),"ast":("ast",10.0,40.0),
 "alp":("alp",44.0,130.0),"total_protein":("total_protein",6.0,8.5),
 "pt":("pt",9.0,14.0),"ptt":("ptt",25.0,40.0),"inr":("inr",0.8,1.3),
 "fibrinogen":("fibrinogen",150.0,450.0),"ld_ldh":("ld_ldh",100.0,250.0),
 "ck_cpk":("ck_cpk",30.0,300.0),"rdw":(None,11.0,15.0),"mcv":(None,80.0,100.0),
}
ANALYTES = list(D.keys())

# 母队列（所有老年首次ICU，含 FI 不可评估者）
parent = pd.read_csv(f"{ANA}/analysis_mimiciv.csv").copy()
parent["stay_id"] = parent["stay_id"].astype(str)
ids = tuple(parent["stay_id"].unique())
print("母队列 n =", len(parent))

with conn("mimiciv") as c:
    cols = ", ".join([f"{D[a][0]}_min AS {a}_min, {D[a][0]}_max AS {a}_max"
                      for a in ANALYTES if D[a][0] is not None])
    fdl = pd.read_sql_query(f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {ids}", c)
    fdl["stay_id"] = fdl["stay_id"].astype(str)
    extra = pd.read_sql_query(f"""
        SELECT i.stay_id,
            MAX(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) AS rdw_max,
            MIN(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) AS rdw_min,
            MAX(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) AS mcv_max,
            MIN(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) AS mcv_min,
            MAX(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) AS lactate_max,
            MIN(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) AS lactate_min,
            MAX(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) AS magnesium_max,
            MIN(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) AS magnesium_min,
            MAX(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) AS phosphate_max,
            MIN(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) AS phosphate_min
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.labevents le ON le.subject_id=i.subject_id
            AND le.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
            AND le.valuenum IS NOT NULL
            AND le.itemid IN (51277,53165,51250,51691,50813,52442,53154,50960,50970)
        WHERE i.stay_id IN {ids} GROUP BY i.stay_id""", c)
    extra["stay_id"] = extra["stay_id"].astype(str)
L = fdl.merge(extra, on="stay_id", how="left")

# 缺陷矩阵
deficits = []
for a in ANALYTES:
    lo, hi = D[a][1], D[a][2]
    mn = L.get(f"{a}_min"); mx = L.get(f"{a}_max")
    assessed = mn.notna() | mx.notna()
    deficit = ((mn < lo) | (mx > hi)).fillna(False)
    deficits.append(deficit.where(assessed))
Dmat = pd.concat(deficits, axis=1)
assessed_n = Dmat.notna().sum(axis=1)
deficit_n = Dmat.sum(axis=1, min_count=1).fillna(0)
L = L.copy()
L["assessed_n"] = assessed_n.values
L["deficit_n"] = deficit_n.values

df = parent.merge(L[["stay_id","assessed_n","deficit_n"]], on="stay_id", how="left")
# 仅对 已评估项>0 计算 FI；避免 np.where 同时计算 0 分母
fi_raw = df["deficit_n"] / df["assessed_n"].replace(0, np.nan)
df["fi_lab"] = fi_raw.where(df["assessed_n"] > 0)
df["fi10"] = df["fi_lab"]*10

def fit_or(d, label):
    """用显式设计矩阵拟合 died~fi10+age+male+ard，返回 (n, OR, lo, hi, p_str)。"""
    d = d.dropna(subset=["fi10","died","age","male","ard"]).copy()
    y = d["died"].astype(float).values
    X = np.column_stack([np.ones(len(d)), d["fi10"].astype(float).values,
                         d["age"].astype(float).values, d["male"].astype(float).values,
                         d["ard"].astype(float).values]).astype(float)
    if len(np.unique(y)) < 2 or np.linalg.matrix_rank(X) < X.shape[1]:
        return (len(d), np.nan, np.nan, np.nan, "rank/sep")
    try:
        m = sm.Logit(y, X).fit(disp=0)
        b = m.params[1]; se = m.bse[1]; p = m.pvalues[1]
        return (len(d), round(np.exp(b),3), round(np.exp(b-1.96*se),3),
                round(np.exp(b+1.96*se),3), ("<0.001" if p<0.001 else round(p,3)))
    except Exception as e:
        return (len(d), np.nan, np.nan, np.nan, f"fail:{e}")

rows = []
for T in [10, 15, 20]:
    d = df[df["assessed_n"] >= T].copy()
    n, OR, lo, hi, p = fit_or(d, f"已评估项>={T}")
    print(f"[diag] T>={T}: n={n}, OR={OR}, p={p}")
    rows.append(dict(threshold=f"已评估项>={T}", n=n, OR=OR, lo=lo, hi=hi, p=p))

# 早期死亡泄漏：用 filab 的 icu_los（stay_id 统一为字符串，避免 map 全 NaN）
_fl = pd.read_csv(os.path.join(ROOT, 'filab', 'filab_mimiciv.csv'))
_fl["stay_id"] = _fl["stay_id"].astype(str)
fl = _fl.set_index("stay_id")["icu_los_days"]
df["icu_los_days"] = df["stay_id"].map(fl)
d0 = df[~((df["died"]==1) & (df["icu_los_days"]<1))].copy()
n, OR, lo, hi, p = fit_or(d0[d0["assessed_n"]>=15], "排除ICU死亡<24h(>=15)")
print(f"[diag] early-death-excl: n={n}, OR={OR}, p={p}")
rows.append(dict(threshold="排除ICU死亡<24h(>=15)", n=n, OR=OR, lo=lo, hi=hi, p=p))

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/rev2_threshold_sensitivity.csv", index=False)
print("\n=== 阈值敏感性（重建, MIMIC）===")
print(out.to_string(index=False))
print("DONE -> rev2_threshold_sensitivity.csv")
