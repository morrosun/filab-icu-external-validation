"""
方向 B 分析层: 衰弱(FI-Lab) 是否修饰老年 ARD 患者 ICU 预后
- 队列: 老年(>=65) 首次 ICU 住院, 分 ARD / 非ARD 两组 (三库统一)
- FI  : 30 项统一缺陷 (见 build_filab.D)
- 主模型(三库同式): logit(death) = ARD + FI + ARD*FI + age + sex
- MIMIC 推导 -> NWICU/eICU 外部验证 (AUC, Brier, 校准)
- MIMIC 敏感性: + Charlson 合并症指数
"""
import psycopg2, pandas as pd, numpy as np, os, statsmodels.formula.api as smf
import os
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


CONN = dict(**_DB_AUTH)
def conn(db):
    c = CONN.copy(); c["dbname"] = db
    return psycopg2.connect(**c)

# ---------- ARD 定义 (ICD-9/10, 三库同义) ----------
ARD = [
 ("SLE", r"^(7100|7101)$", r"^(M32)"),
 ("SSc", r"^(7101)$", r"^(M34)"),
 ("Sjogren", r"^(7102)$", r"^(M350)"),
 ("PMDM", r"^(7103|7104)$", r"^(M33)"),
 ("RA", r"^(7140)$", r"^(M05|M06)"),
 ("ANCA", r"^(4464|44621)$", r"^(M313|M300|M3181)"),
 ("APS", r"^(28981|79579)$", r"^(D6861|M3181)"),
 ("MCTD", r"^(7109)$", r"^(M351)"),
 ("Behcet", r"^(1361)$", r"^(M352)"),
 ("GCA", r"^(4465)$", r"^(M315|M316)"),
 ("PMR", r"^(725)$", r"^(M353)"),
 ("JIA/Reactive", r"^(6960|0993|7142)$", r"^(M07|M02|M08|M09)"),
 ("SpA", r"^(720|721|722)$", r"^(M45|M46|M47|M48)"),
 ("OtherCTD", r"^(710|711|712|713|719)$", r"^(M35|M36)"),
 ("VascNOS", r"^(446)$", r"^(M31)"),
]
def ard_where(v="d.icd_version", c="d.icd_code"):
    return " OR ".join([f"({v}=9 AND {c} ~ '{r9}')" for _,r9,_ in ARD] +
                       [f"({v}=10 AND {c} ~ '{r10}')" for _,_,r10 in ARD])
EICU_TEXT = (r"sle|lupus|rheumat|scleroderma|systemic sclerosis|polymyosit|dermatomyosit|"
             r"polyarter|wegener|granulomatosis|takayasu|behcet|sj.gren|mixed connective|"
             r"antiphospholip|vasculit|giant cell arter|temporal arter|polymyalg|spondyloarthr|"
             r"psoriatic|ankylosing|reactive arthr|juvenile arthr|stills|churg|goodpasture|"
             r"microscopic polyang|eosinophilic granulomat|cryoglobulin|henoch|iga vascul")

# ---------- 30 项 FI 缺陷定义 ----------
D = {
 "albumin":("albumin",100021,"albumin",3.5,5.0), "bilirubin_tot":("bilirubin_total",100020,"total bilirubin",0.1,1.2),
 "bilirubin_dir":("bilirubin_direct",100049,"direct bilirubin",0.0,0.3), "bun":("bun",100004,"BUN",7.0,20.0),
 "calcium":("calcium",100003,"calcium",8.5,10.5), "chloride":("chloride",100012,"chloride",98.0,110.0),
 "creatinine":("creatinine",100002,"creatinine",0.5,1.3), "glucose":("glucose",100001,"glucose",70.0,140.0),
 "hematocrit":("hematocrit",100006,"Hct",36.0,50.0), "hemoglobin":("hemoglobin",100007,"Hgb",10.0,17.0),
 "lactate":(None,100031,"lactate",0.5,2.0), "magnesium":(None,100009,"magnesium",1.5,2.5),
 "phosphate":(None,100015,"phosphate",2.5,4.5), "platelet":("platelets",100014,"platelets x 1000",150.0,450.0),
 "potassium":("potassium",100011,"potassium",3.5,5.5), "sodium":("sodium",100010,"sodium",135.0,145.0),
 "bicarbonate":("bicarbonate",100013,"bicarbonate",22.0,29.0), "aniongap":("aniongap",100040,"anion gap",3.0,16.0),
 "alt":("alt",100042,"ALT (SGPT)",7.0,40.0), "ast":("ast",100032,"AST (SGOT)",10.0,40.0),
 "alp":("alp",100033,"alkaline phos.",44.0,130.0), "total_protein":("total_protein",100027,"total protein",6.0,8.5),
 "pt":("pt",100034,"PT",9.0,14.0), "ptt":("ptt",100046,"PTT",25.0,40.0),
 "inr":("inr",100030,"PT - INR",0.8,1.3), "fibrinogen":("fibrinogen",100048,"fibrinogen",150.0,450.0),
 "ld_ldh":("ld_ldh",100055,"LDH",100.0,250.0), "ck_cpk":("ck_cpk",100085,"CPK",30.0,300.0),
 "rdw":(None,100008,"RDW",11.0,15.0), "mcv":(None,100017,"MCV",80.0,100.0),
}
ANALYTES = list(D.keys()); MIN_THRESH = 15

# ---------- 合并队列 (ARD + 非ARD) ----------
def cohort_mimic_like(db, hsp, icu):
    w = ard_where()
    sql = f"""
    WITH ard_hadm AS (SELECT DISTINCT d.hadm_id FROM {hsp}.diagnoses_icd d WHERE {w}),
    base AS (
        SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime,
               a.admittime, a.dischtime, a.hospital_expire_flag,
               p.gender, (p.anchor_age + (EXTRACT(year FROM a.admittime)::int - p.anchor_year::int)) AS age
        FROM {icu}.icustays i
        JOIN {hsp}.admissions a ON i.hadm_id=a.hadm_id
        JOIN {hsp}.patients p ON i.subject_id=p.subject_id),
    ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn FROM base),
    first AS (SELECT * FROM ranked WHERE rn=1 AND age>=65 AND age<300)
    SELECT f.stay_id, f.subject_id, f.hadm_id, f.age, f.gender,
           f.hospital_expire_flag::int AS died,
           CASE WHEN f.hadm_id IN (SELECT hadm_id FROM ard_hadm) THEN 1 ELSE 0 END AS ard
    FROM first f"""
    with conn(db) as c:
        df = pd.read_sql_query(sql, c)
    if db == "nwicu":  # hospital_expire_flag 不可靠, 改用 dod<=dischtime
        with conn("nwicu") as c:
            dod = pd.read_sql_query(
                "SELECT a.hadm_id, MAX(CASE WHEN p.dod IS NOT NULL AND p.dod<=a.dischtime THEN 1 ELSE 0 END) died_dod "
                "FROM hosp.admissions a JOIN hosp.patients p ON a.subject_id=p.subject_id GROUP BY a.hadm_id", c)
        df = df.merge(dod, on="hadm_id", how="left")
        df["died"] = df["died_dod"].fillna(0).astype(int)
        df = df.drop(columns=["died_dod"])
    if db == "mimiciv":  # 加 Charlson
        with conn("mimiciv") as c:
            ch = pd.read_sql_query("SELECT hadm_id, charlson_comorbidity_index AS charlson FROM mimiciv_derived.charlson", c)
        df = df.merge(ch, on="hadm_id", how="left")
    return df

def cohort_eicu():
    w9 = " OR ".join([f"(icd9code ~ '{r9}')" for _,r9,_ in ARD])
    sql = f"""
    WITH ard_stay AS (
        SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
        WHERE ({w9}) OR (diagnosisstring ~* '{EICU_TEXT}')),
    base AS (
        SELECT p.patientunitstayid, p.patienthealthsystemstayid, p.unitadmittime24,
               CASE WHEN p.age='> 89' THEN 89 WHEN p.age ~ '^[0-9]+$' THEN p.age::int ELSE NULL END AS age,
               p.gender,
               CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END AS died
        FROM eicu_crd.patient p),
    ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY patienthealthsystemstayid ORDER BY unitadmittime24) rn FROM base),
    first AS (SELECT * FROM ranked WHERE rn=1 AND age>=65 AND age<300)
    SELECT f.patientunitstayid AS stay_id, NULL::int AS subject_id, NULL::int AS hadm_id,
           f.age, f.gender, f.died,
           CASE WHEN f.patientunitstayid IN (SELECT patientunitstayid FROM ard_stay) THEN 1 ELSE 0 END AS ard
    FROM first f"""
    with conn("eicu") as c:
        return pd.read_sql_query(sql, c)

# ---------- FI 提取 (参数化 id 列表) ----------
def fi_mimic(ids):
    with conn("mimiciv") as c:
        cols = ", ".join([f"{D[a][0]}_min AS {a}_min, {D[a][0]}_max AS {a}_max"
                          for a in ANALYTES if D[a][0] is not None])
        fdl = pd.read_sql_query(f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {ids}", c)
        fdl["stay_id"] = fdl["stay_id"].astype(str)
        extra = pd.read_sql_query(f"""
            SELECT i.stay_id,
                MAX(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) rdw_max,
                MIN(CASE WHEN le.itemid IN (51277,53165) THEN le.valuenum END) rdw_min,
                MAX(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) mcv_max,
                MIN(CASE WHEN le.itemid IN (51250,51691) THEN le.valuenum END) mcv_min,
                MAX(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) lactate_max,
                MIN(CASE WHEN le.itemid IN (50813,52442,53154) THEN le.valuenum END) lactate_min,
                MAX(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) magnesium_max,
                MIN(CASE WHEN le.itemid IN (50960) THEN le.valuenum END) magnesium_min,
                MAX(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) phosphate_max,
                MIN(CASE WHEN le.itemid IN (50970) THEN le.valuenum END) phosphate_min
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents le ON le.subject_id=i.subject_id
             AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
             AND le.valuenum IS NOT NULL AND le.itemid IN (51277,53165,51250,51691,50813,52442,53154,50960,50970)
            WHERE i.stay_id IN {ids} GROUP BY i.stay_id""", c)
        extra["stay_id"] = extra["stay_id"].astype(str)
    return fdl.merge(extra, on="stay_id", how="left")

def fi_nwicu(ids):
    itemids = [D[a][1] for a in ANALYTES]
    case = ", ".join([f"MAX(CASE WHEN le.itemid={D[a][1]} THEN le.valuenum END) {a}_max, "
                      f"MIN(CASE WHEN le.itemid={D[a][1]} THEN le.valuenum END) {a}_min" for a in ANALYTES])
    with conn("nwicu") as c:
        df = pd.read_sql_query(f"""
            SELECT i.stay_id, {case}
            FROM icu.icustays i
            JOIN hosp.labevents le ON le.subject_id=i.subject_id AND le.hadm_id=i.hadm_id
             AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
             AND le.valuenum IS NOT NULL AND le.itemid IN {tuple(itemids)}
            WHERE i.stay_id IN {ids} GROUP BY i.stay_id""", c)
        df["stay_id"] = df["stay_id"].astype(str)
    return df

def fi_eicu(ids):
    names = [D[a][2] for a in ANALYTES]
    case = ", ".join([f"MAX(CASE WHEN labname='{D[a][2]}' THEN labresult END) {a}_max, "
                      f"MIN(CASE WHEN labname='{D[a][2]}' THEN labresult END) {a}_min" for a in ANALYTES])
    with conn("eicu") as c:
        df = pd.read_sql_query(f"""
            SELECT patientunitstayid AS stay_id, {case}
            FROM eicu_crd.lab
            WHERE patientunitstayid IN {ids} AND labresultoffset BETWEEN 0 AND 1440
              AND labresult IS NOT NULL AND labname IN {tuple(names)}
            GROUP BY patientunitstayid""", c)
        df["stay_id"] = df["stay_id"].astype(str)
    return df

def compute_fi(L):
    defs=[]
    for a in ANALYTES:
        lo,hi=D[a][3],D[a][4]
        mn=L.get(f"{a}_min"); mx=L.get(f"{a}_max")
        assessed = mn.notna()|mx.notna()
        defs.append(((mn<lo)|(mx>hi)).fillna(False).where(assessed))
    M=pd.concat(defs,axis=1); an=M.notna().sum(axis=1); dn=M.sum(axis=1,min_count=1).fillna(0)
    fi=np.full(len(L),np.nan); m=an>=MIN_THRESH; fi[m.values]=(dn[m]/an[m]).values
    return pd.Series(fi,index=L.index), an

# ---------- 主流程 ----------
def build(db):
    if db=="mimiciv": coh=cohort_mimic_like("mimiciv","mimiciv_hosp","mimiciv_icu")
    elif db=="nwicu": coh=cohort_mimic_like("nwicu","hosp","icu")
    else: coh=cohort_eicu()
    coh["stay_id"]=coh["stay_id"].astype(str)
    ids=tuple(coh["stay_id"])
    if db=="mimiciv": L=fi_mimic(ids)
    elif db=="nwicu": L=fi_nwicu(ids)
    else: L=fi_eicu(ids)
    fi,na=compute_fi(L)
    L=L.copy(); L["fi_lab"]=fi.values; L["n_assessed"]=na.values
    out=coh.merge(L[["stay_id","fi_lab","n_assessed"]],on="stay_id",how="left")
    out["male"]=(out["gender"].astype(str).str.upper().str.startswith("M")).astype(int)
    return out

def fit(df, formula):
    d=df.dropna(subset=["died","fi_lab","age","gender"]).copy()
    d["died"]=d["died"].astype(int)
    m=smf.logit(formula, data=d).fit(disp=0)
    return m, d

def report(m, label):
    res=pd.DataFrame({"OR":np.exp(m.params),"lo":np.exp(m.conf_int()[0]),"hi":np.exp(m.conf_int()[1]),"p":m.pvalues})
    res.index=m.params.index
    print(f"\n=== {label} (n={int(m.nobs)}, AIC={m.aic:.1f}) ===")
    print(res.round(4).to_string())
    return res

def main():
    os.makedirs(os.path.join(ROOT, 'analysis'), exist_ok=True)
    data={db:build(db) for db in ["mimiciv","nwicu","eicu"]}
    for db,df in data.items():
        df.to_csv(os.path.join(ROOT, 'analysis', f'analysis_{db}.csv'),index=False)
        # 描述: ARD 组 vs 非ARD 组 FI / 死亡
        g=df.groupby("ard").agg(n=("stay_id","size"), fi_med=("fi_lab","median"),
                                died=("died","sum"), fi_assessed=("n_assessed","median"))
        g["mort"]=(g["died"]/g["n"]*100).round(1)
        print(f"\n##### {db.upper()} combined cohort #####")
        print(g.round(3).to_string())

    # 主模型 (三库同式)
    forms={"primary":"died ~ ard * fi_lab + age + male"}
    results={}
    for name,f in forms.items():
        print(f"\n########## MODEL: {name} ##########")
        mim,dm = fit(data["mimiciv"], f); results["mimic_primary"]=report(mim,"MIMIC-IV (derivation)")
        nw, dn = fit(data["nwicu"], f); results["nwicu_primary"]=report(nw,"NWICU (external)")
        ec, de = fit(data["eicu"], f); results["eicu_primary"]=report(ec,"eICU (external)")
        # 外部验证: 用 MIMIC 系数预测 NWICU/eICU
        params=mim.params
        def pred(d):
            x=d.copy(); x["ard:fi_lab"]=x["ard"]*x["fi_lab"]
            lin=params.get("Intercept",0.0)
            for k in ["ard","fi_lab","ard:fi_lab","age","male"]:
                if k in params: lin+=params[k]*x[k]
            return 1/(1+np.exp(-lin))
        for db,dd in [("NWICU",dn),("eICU",de)]:
            p=pred(dd); auc=roc_auc_score(dd["died"],p); br=brier_score_loss(dd["died"],p)
            print(f"\n[MIMIC-derived model -> {db}] AUC={auc:.3f}, Brier={br:.4f}")

    # MIMIC 敏感性: + Charlson
    dm2=data["mimiciv"].dropna(subset=["charlson"])
    m2,_=fit(dm2,"died ~ ard * fi_lab + age + male + charlson")
    results["mimic_sensitivity_charlson"]=report(m2,"MIMIC-IV sensitivity (+Charlson)")

    # 汇总结果存盘 (CSV)
    out=pd.concat({k:v.reset_index().rename(columns={"index":"term"})
                   for k,v in results.items()}).reset_index(level=0)
    out=out.rename(columns={"level_0":"model"})
    out.to_csv(os.path.join(ROOT, 'analysis', 'regression_results.csv'), index=False)
    print("\nDone. Outputs in analysis/")

if __name__=="__main__":
    main()
