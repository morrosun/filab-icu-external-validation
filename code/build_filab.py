"""
方向 B — FI-Lab 衰弱指数 (30 项统一缺陷, 三库可比)
- MIMIC-IV : mimiciv_derived.first_day_lab (28项) + labevents pivot (rdw,mcv)
- NWICU     : hosp.labevents pivot (30项, 100xxx itemid)
- eICU      : eicu_crd.lab pivot (30项, labname)
缺陷定义: 每项化验在入院首日(ICU 24h)取 min/max, 超出参考范围即计 1 个 deficit.
FI = 缺陷数 / 已评估项数 (要求已评估 >= 20/30 才计算).
"""
import psycopg2, pandas as pd, numpy as np, os
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


CONN = dict(**_DB_AUTH)
def conn(db):
    c = CONN.copy(); c["dbname"] = db
    return psycopg2.connect(**c)

# 30 项缺陷: analyte -> (MIMIC first_day 列, NWICU itemid, eICU labname, low, high)
D = {
 "albumin":        ("albumin",        100021, "albumin",            3.5, 5.0),
 "bilirubin_tot":  ("bilirubin_total",100020, "total bilirubin",    0.1, 1.2),
 "bilirubin_dir":  ("bilirubin_direct",100049,"direct bilirubin",   0.0, 0.3),
 "bun":            ("bun",            100004, "BUN",                7.0, 20.0),
 "calcium":        ("calcium",        100003, "calcium",            8.5, 10.5),
 "chloride":       ("chloride",       100012, "chloride",           98.0, 110.0),
 "creatinine":     ("creatinine",     100002, "creatinine",         0.5, 1.3),
 "glucose":        ("glucose",        100001, "glucose",            70.0, 140.0),
 "hematocrit":     ("hematocrit",     100006, "Hct",                36.0, 50.0),
 "hemoglobin":     ("hemoglobin",     100007, "Hgb",                10.0, 17.0),
 "lactate":        (None,             100031, "lactate",            0.5, 2.0),
 "magnesium":      (None,             100009, "magnesium",          1.5, 2.5),
 "phosphate":      (None,             100015, "phosphate",          2.5, 4.5),
 "platelet":       ("platelets",      100014, "platelets x 1000",  150.0, 450.0),
 "potassium":      ("potassium",      100011, "potassium",          3.5, 5.5),
 "sodium":         ("sodium",         100010, "sodium",             135.0, 145.0),
 "bicarbonate":    ("bicarbonate",    100013, "bicarbonate",        22.0, 29.0),
 "aniongap":       ("aniongap",       100040, "anion gap",          3.0, 16.0),
 "alt":            ("alt",            100042, "ALT (SGPT)",         7.0, 40.0),
 "ast":            ("ast",            100032, "AST (SGOT)",        10.0, 40.0),
 "alp":            ("alp",            100033, "alkaline phos.",     44.0, 130.0),
 "total_protein":  ("total_protein",  100027, "total protein",      6.0, 8.5),
 "pt":             ("pt",             100034, "PT",                 9.0, 14.0),
 "ptt":            ("ptt",            100046, "PTT",                25.0, 40.0),
 "inr":            ("inr",            100030, "PT - INR",           0.8, 1.3),
 "fibrinogen":     ("fibrinogen",     100048, "fibrinogen",         150.0, 450.0),
 "ld_ldh":         ("ld_ldh",         100055, "LDH",                100.0, 250.0),
 "ck_cpk":         ("ck_cpk",         100085, "CPK",                30.0, 300.0),
 "rdw":            (None,             100008, "RDW",                11.0, 15.0),
 "mcv":            (None,             100017, "MCV",                80.0, 100.0),
}
ANALYTES = list(D.keys())
MIN_THRESH = 15  # 至少评估 15/30 项才计算 FI (Rockwood: 用可用项)

COHORT_DIR = os.path.join(ROOT, 'cohort')

def load_cohort(db):
    df = pd.read_csv(f"{COHORT_DIR}/cohort_{db}.csv")
    return df[["stay_id"]].astype(str)

# ---------- MIMIC-IV ----------
def extract_mimic():
    ids = tuple(load_cohort("mimiciv")["stay_id"])
    with conn("mimiciv") as c:
        # 28 项来自 first_day_lab
        cols = ", ".join([f"{D[a][0]}_min AS {a}_min, {D[a][0]}_max AS {a}_max"
                          for a in ANALYTES if D[a][0] is not None])
        fdl = pd.read_sql_query(
            f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {ids}", c)
        fdl["stay_id"] = fdl["stay_id"].astype(str)
        # rdw / mcv / lactate 来自 labevents pivot (first_day_lab 不含这三项)
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
            JOIN mimiciv_hosp.labevents le
              ON le.subject_id = i.subject_id
             AND le.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
             AND le.valuenum IS NOT NULL
             AND le.itemid IN (51277,53165,51250,51691,50813,52442,53154,50960,50970)
            WHERE i.stay_id IN {ids}
            GROUP BY i.stay_id""", c)
        extra["stay_id"] = extra["stay_id"].astype(str)
    return fdl.merge(extra, on="stay_id", how="left")

# ---------- NWICU ----------
def extract_nwicu():
    ids = tuple(load_cohort("nwicu")["stay_id"])
    itemids = [D[a][1] for a in ANALYTES]
    case_sql = ", ".join([
        f"MAX(CASE WHEN le.itemid={D[a][1]} THEN le.valuenum END) AS {a}_max, "
        f"MIN(CASE WHEN le.itemid={D[a][1]} THEN le.valuenum END) AS {a}_min"
        for a in ANALYTES])
    with conn("nwicu") as c:
        df = pd.read_sql_query(f"""
            SELECT i.stay_id,
                {case_sql}
            FROM icu.icustays i
            JOIN hosp.labevents le
              ON le.subject_id = i.subject_id
             AND le.hadm_id = i.hadm_id
             AND le.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
             AND le.valuenum IS NOT NULL
             AND le.itemid IN {tuple(itemids)}
            WHERE i.stay_id IN {ids}
            GROUP BY i.stay_id""", c)
        df["stay_id"] = df["stay_id"].astype(str)
    return df

# ---------- eICU ----------
def extract_eicu():
    ids = tuple(load_cohort("eicu")["stay_id"])
    names = [D[a][2] for a in ANALYTES]
    case_sql = ", ".join([
        f"MAX(CASE WHEN labname='{D[a][2]}' THEN labresult END) AS {a}_max, "
        f"MIN(CASE WHEN labname='{D[a][2]}' THEN labresult END) AS {a}_min"
        for a in ANALYTES])
    with conn("eicu") as c:
        df = pd.read_sql_query(f"""
            SELECT patientunitstayid AS stay_id,
                {case_sql}
            FROM eicu_crd.lab
            WHERE patientunitstayid IN {ids}
              AND labresultoffset BETWEEN 0 AND 1440
              AND labresult IS NOT NULL
              AND labname IN {tuple(names)}
            GROUP BY patientunitstayid""", c)
        df["stay_id"] = df["stay_id"].astype(str)
    return df

def compute_fi(labs):
    defs = []
    for a in ANALYTES:
        lo, hi = D[a][3], D[a][4]
        mn = labs.get(f"{a}_min")
        mx = labs.get(f"{a}_max")
        assessed = mn.notna() | mx.notna()
        deficit = ((mn < lo) | (mx > hi)).fillna(False)
        defs.append(deficit.where(assessed))
    Dmat = pd.concat(defs, axis=1)
    assessed_n = Dmat.notna().sum(axis=1)
    deficit_n = Dmat.sum(axis=1, min_count=1).fillna(0)
    fi = np.full(len(labs), np.nan)
    mask = assessed_n >= MIN_THRESH
    fi[mask.values] = (deficit_n[mask] / assessed_n[mask]).values
    return pd.Series(fi, index=labs.index), assessed_n

def summarize(df, name):
    d = df.dropna(subset=["fi_lab"])
    qs = d["fi_lab"].quantile([0.25, 0.5, 0.75]).round(3).tolist()
    # 四分位死亡梯度
    d = d.copy()
    d["q"] = pd.qcut(d["fi_lab"], 4, labels=["Q1","Q2","Q3","Q4"])
    grad = d.groupby("q", observed=True).agg(n=("died","size"), deaths=("died","sum"))
    grad["mort"] = (grad["deaths"]/grad["n"]*100).round(1)
    r = d["fi_lab"].corr(d["died"])
    return dict(db=name, n_fi=len(d), fi_med=qs[1], fi_q1=qs[0], fi_q3=qs[2],
                n_assessed_med=int(d["n_assessed"].median()),
                q1_mort=grad.loc["Q1","mort"], q4_mort=grad.loc["Q4","mort"],
                corr=r)

def main():
    os.makedirs(os.path.join(ROOT, 'filab'), exist_ok=True)
    cohorts = {db: pd.read_csv(f"{COHORT_DIR}/cohort_{db}.csv") for db in ["mimiciv","nwicu","eicu"]}
    labs = {"mimiciv": extract_mimic(), "nwicu": extract_nwicu(), "eicu": extract_eicu()}
    rows = []
    for db in ["mimiciv","nwicu","eicu"]:
        L = labs[db]
        fi, nass = compute_fi(L)
        L = L.copy()
        L["fi_lab"] = fi.values
        L["n_assessed"] = nass.values
        out = cohorts[db][["stay_id","database","age","gender","died","icu_los_days"]].copy()
        out["stay_id"] = out["stay_id"].astype(str)
        out = out.merge(L[["stay_id","fi_lab","n_assessed"]], on="stay_id", how="left")
        out.to_csv(os.path.join(ROOT, 'filab', f'filab_{db}.csv'), index=False)
        rows.append(summarize(out, db.upper()))
    summ = pd.DataFrame(rows)[["db","n_fi","fi_med","fi_q1","fi_q3","n_assessed_med","q1_mort","q4_mort","corr"]]
    summ.to_csv(os.path.join(ROOT, 'filab', 'filab_summary.csv'), index=False)
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()
