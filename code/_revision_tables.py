"""
Revision supplementary tables:
  1) Per-lab-item missingness across 3 DBs (30 items)  -> per_lab_missingness.csv
  2) Baseline Table 1 across 3 DBs                      -> baseline_table1.csv
  3) eICU negation false-positive audit                -> eicu_negation_audit.csv
  4) ARD ICD-9/10 code list (S1)                        -> ard_icd_codes.csv
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


PG = dict(**_DB_AUTH)
OUT = os.path.join(ROOT, 'analysis', 'revision')
COHORT_DIR = os.path.join(ROOT, 'cohort')
os.makedirs(OUT, exist_ok=True)

def conn(db):
    return psycopg2.connect(dbname=db, **PG)

# --- 30-item deficit map (analyte -> MIMIC first_day col, NWICU itemid, eICU labname) ---
D = {
 "albumin":("albumin",100021,"albumin"),"bilirubin_tot":("bilirubin_total",100020,"total bilirubin"),
 "bilirubin_dir":("bilirubin_direct",100049,"direct bilirubin"),"bun":("bun",100004,"BUN"),
 "calcium":("calcium",100003,"calcium"),"chloride":("chloride",100012,"chloride"),
 "creatinine":("creatinine",100002,"creatinine"),"glucose":("glucose",100001,"glucose"),
 "hematocrit":("hematocrit",100006,"Hct"),"hemoglobin":("hemoglobin",100007,"Hgb"),
 "lactate":(None,100031,"lactate"),"magnesium":(None,100009,"magnesium"),
 "phosphate":(None,100015,"phosphate"),"platelet":("platelets",100014,"platelets x 1000"),
 "potassium":("potassium",100011,"potassium"),"sodium":("sodium",100010,"sodium"),
 "bicarbonate":("bicarbonate",100013,"bicarbonate"),"aniongap":("aniongap",100040,"anion gap"),
 "alt":("alt",100042,"ALT (SGPT)"),"ast":("ast",100032,"AST (SGOT)"),
 "alp":("alp",100033,"alkaline phos."),"total_protein":("total_protein",100027,"total protein"),
 "pt":("pt",100034,"PT"),"ptt":("ptt",100046,"PTT"),"inr":("inr",100030,"PT - INR"),
 "fibrinogen":("fibrinogen",100048,"fibrinogen"),"ld_ldh":("ld_ldh",100055,"LDH"),
 "ck_cpk":("ck_cpk",100085,"CPK"),"rdw":(None,51277,"RDW"),"mcv":(None,51250,"MCV"),
}
ANALYTES = list(D.keys())

def cohort_ids(db):
    # Full analysis cohort (all eligible elderly ICU patients) for per-indicator missingness
    df = pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{db}.csv'))
    return tuple(df["stay_id"].astype(str))

# ---------------- 1) per-lab missingness ----------------
def mimic_assessed():
    ids = cohort_ids("mimiciv")
    with conn("mimiciv") as c:
        cols = ", ".join([f"({D[a][0]}_min IS NOT NULL OR {D[a][0]}_max IS NOT NULL)::int AS {a}"
                          for a in ANALYTES if D[a][0]])
        fdl = pd.read_sql_query(
            f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {ids}", c)
        fdl["stay_id"] = fdl["stay_id"].astype(str)
        extra = pd.read_sql_query(f"""
            SELECT i.stay_id,
              (BOOL_OR(le.itemid IN (51277,53165)))::int AS rdw,
              (BOOL_OR(le.itemid IN (51250,51691)))::int AS mcv,
              (BOOL_OR(le.itemid IN (50813,52442,53154)))::int AS lactate,
              (BOOL_OR(le.itemid=50960))::int AS magnesium,
              (BOOL_OR(le.itemid=50970))::int AS phosphate
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents le ON le.subject_id=i.subject_id
             AND le.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
             AND le.valuenum IS NOT NULL
             AND le.itemid IN (51277,53165,51250,51691,50813,52442,53154,50960,50970)
            WHERE i.stay_id IN {ids} GROUP BY i.stay_id""", c)
        extra["stay_id"] = extra["stay_id"].astype(str)
    m = fdl.merge(extra, on="stay_id", how="left")
    for a in ANALYTES:
        if a not in m: m[a] = 0
    return m

def nwicu_assessed():
    ids = cohort_ids("nwicu")
    itemids = tuple(D[a][1] for a in ANALYTES)
    case = ", ".join([f"(BOOL_OR(le.itemid={D[a][1]}))::int AS {a}" for a in ANALYTES])
    with conn("nwicu") as c:
        df = pd.read_sql_query(f"""
            SELECT i.stay_id, {case}
            FROM icu.icustays i
            JOIN hosp.labevents le ON le.subject_id=i.subject_id AND le.hadm_id=i.hadm_id
             AND le.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
             AND le.valuenum IS NOT NULL AND le.itemid IN {itemids}
            WHERE i.stay_id IN {ids} GROUP BY i.stay_id""", c)
    df["stay_id"] = df["stay_id"].astype(str)
    return df

def eicu_assessed():
    ids = cohort_ids("eicu")
    names = tuple(D[a][2] for a in ANALYTES)
    case = ", ".join([f"(BOOL_OR(labname='{D[a][2]}'))::int AS {a}" for a in ANALYTES])
    with conn("eicu") as c:
        df = pd.read_sql_query(f"""
            SELECT patientunitstayid AS stay_id, {case}
            FROM eicu_crd.lab
            WHERE patientunitstayid IN {ids} AND labresultoffset BETWEEN 0 AND 1440
              AND labresult IS NOT NULL AND labname IN {names}
            GROUP BY patientunitstayid""", c)
    df["stay_id"] = df["stay_id"].astype(str)
    return df

def missingness():
    frames = {"MIMIC-IV": mimic_assessed(), "NWICU": nwicu_assessed(), "eICU": eicu_assessed()}
    # denominators = full analysis cohort size per db (all eligible elderly ICU patients)
    ncoh = {}
    for db, key in [("MIMIC-IV","mimiciv"),("NWICU","nwicu"),("eICU","eicu")]:
        ncoh[db] = pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{key}.csv')).shape[0]
    rows = []
    for a in ANALYTES:
        row = {"item": a}
        for db in frames:
            f = frames[db]
            assessed = int(f[a].fillna(0).sum()) if a in f else 0
            n = ncoh[db]
            row[f"{db}_miss%"] = round(100*(1 - assessed/n), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/per_lab_missingness.csv", index=False)
    print("per_lab_missingness ->", out.shape)
    print(out.to_string(index=False))
    return out

# ---------------- 2) baseline table 1 ----------------
def _fmt_med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"

def baseline():
    rows = []
    for db, key in [("MIMIC-IV","mimiciv"),("NWICU","nwicu"),("eICU","eicu")]:
        a = pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{key}.csv'))
        a = a.dropna(subset=["fi_lab"]).copy()  # FI-evaluable analysis cohort
        n = len(a)
        r = {"variable":"N","MIMIC-IV" if db=="MIMIC-IV" else db: n}
        rows.append(dict(db=db, var="N (病例数)", val=str(n)))
        rows.append(dict(db=db, var="年龄, 中位数(IQR)", val=_fmt_med(a["age"])))
        male = a["male"].mean()*100 if "male" in a else np.nan
        rows.append(dict(db=db, var="男性, n(%)", val=f"{int(a['male'].sum())} ({male:.1f})" if "male" in a else "NA"))
        rows.append(dict(db=db, var="风湿免疫病(ARD), n(%)", val=f"{int(a['ard'].sum())} ({a['ard'].mean()*100:.1f})"))
        rows.append(dict(db=db, var="FI-Lab, 中位数(IQR)", val=_fmt_med(a["fi_lab"])))
        rows.append(dict(db=db, var="已评估化验项数, 中位数(IQR)", val=_fmt_med(a["n_assessed"])))
        rows.append(dict(db=db, var="院内死亡, n(%)", val=f"{int(a['died'].sum())} ({a['died'].mean()*100:.1f})"))
    out = pd.DataFrame(rows).pivot(index="var", columns="db", values="val")
    order = ["N (病例数)","年龄, 中位数(IQR)","男性, n(%)","风湿免疫病(ARD), n(%)",
             "FI-Lab, 中位数(IQR)","已评估化验项数, 中位数(IQR)","院内死亡, n(%)"]
    out = out.reindex(order)[["MIMIC-IV","NWICU","eICU"]]
    out.to_csv(f"{OUT}/baseline_table1.csv", encoding="utf-8-sig")
    print("\nbaseline_table1 ->", out.shape)
    print(out.to_string())
    return out

# ---------------- 3) eICU negation audit ----------------
NEG = ["rule out","r/o","ruled out","history of","h/o","no evidence","negative for",
       "possible","suspected","versus","cannot exclude","query "]
def eicu_negation_audit():
    ids = cohort_ids("eicu")
    with conn("eicu") as c:
        df = pd.read_sql_query(f"""
            SELECT DISTINCT patientunitstayid AS stay_id, LOWER(diagnosisstring) AS dx
            FROM eicu_crd.diagnosis
            WHERE patientunitstayid IN {ids}""", c)
    df["stay_id"] = df["stay_id"].astype(str)
    # keep only ARD-relevant dx strings
    ard_terms = ["lupus","scleroderma","systemic sclerosis","sjogren","dermatomyositis",
        "polymyositis","myositis","rheumatoid","vasculitis","wegener","granulomatosis",
        "churg","polyarteritis","takayasu","behcet","antiphospholipid","mixed connective",
        "spondyloarth","psoriatic","giant cell arteritis","polymyalgia","polychondritis",
        "ankylosing spondyl"]
    df = df[df["dx"].str.contains("|".join(ard_terms), na=False)]
    df["neg"] = df["dx"].str.contains("|".join(NEG), na=False)
    n_stay = df["stay_id"].nunique()
    neg_stays = df[df["neg"]]["stay_id"].nunique()
    out = pd.DataFrame([dict(
        eicu_ard_dx_strings=len(df),
        stays_with_ard_text=n_stay,
        stays_with_negation_phrase=neg_stays,
        pct_negation=round(100*neg_stays/max(n_stay,1),1))])
    out.to_csv(f"{OUT}/eicu_negation_audit.csv", index=False)
    print("\neicu_negation_audit:"); print(out.to_string(index=False))
    # show example negation phrases
    ex = df[df["neg"]]["dx"].drop_duplicates().head(15).tolist()
    print("examples:", ex)
    return out

# ---------------- 4) ARD ICD code list ----------------
def ard_codes():
    ARD = [
     ("系统性红斑狼疮 SLE","710.0","M32"),
     ("系统性硬化症 SSc","710.1","M34"),
     ("干燥综合征 Sjogren","710.2","M35.0"),
     ("多发性肌炎/皮肌炎 PM/DM","710.3, 710.4","M33"),
     ("类风湿关节炎 RA","714.0","M05, M06"),
     ("ANCA相关血管炎","446.4, 446.21","M31.3, M30.0, M31.81"),
     ("抗磷脂综合征 APS","289.81, 795.79","D68.61, M31.81"),
     ("混合性结缔组织病 MCTD","710.9","M35.1"),
     ("白塞病 Behcet","136.1","M35.2"),
     ("复发性多软骨炎","733.99, 446","M94.1"),
     ("巨细胞/大动脉炎 GCA/TAK","446.5","M31.5, M31.6"),
     ("风湿性多肌痛 PMR","725","M35.3"),
     ("银屑病关节炎","696.0","M07"),
     ("反应性/幼年关节炎","099.3, 714.2, 696.0","M02, M08, M09"),
     ("脊柱关节病","720, 721, 722","M45, M46, M47, M48"),
     ("其他结缔组织病 CTD","710, 711, 712, 713, 719","M35, M36"),
    ]
    out = pd.DataFrame(ARD, columns=["疾病类别","ICD-9-CM","ICD-10-CM"])
    out.to_csv(f"{OUT}/ard_icd_codes.csv", index=False, encoding="utf-8-sig")
    print("\nard_icd_codes ->", out.shape)
    return out

# ---------------- 5) ARD vs non-ARD baseline (FI-evaluable) ----------------
def ard_vs_nonard():
    rows = []
    for db, key in [("MIMIC-IV","mimiciv"),("NWICU","nwicu"),("eICU","eicu")]:
        a = pd.read_csv(os.path.join(ROOT, 'analysis', f'analysis_{key}.csv')).dropna(subset=["fi_lab"])
        for grp, sub in [("ARD", a[a["ard"]==1]), ("non-ARD", a[a["ard"]==0])]:
            rows.append(dict(db=db, group=grp,
                n=len(sub),
                age=_fmt_med(sub["age"]),
                male_pct=round(100*sub["male"].mean(),1),
                fi_med=float(sub["fi_lab"].median()),
                n_assessed_med=float(sub["n_assessed"].median()),
                mort_pct=round(100*sub["died"].mean(),1)))
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/ard_vs_nonard_baseline.csv", index=False, encoding="utf-8-sig")
    print("\nard_vs_nonard_baseline ->", out.shape)
    print(out.to_string(index=False))
    return out

if __name__ == "__main__":
    missingness()
    baseline()
    ard_vs_nonard()
    eicu_negation_audit()
    ard_codes()
    print("\nALL DONE")
