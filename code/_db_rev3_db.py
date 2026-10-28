# -*- coding: utf-8 -*-
"""Round-3 DB tasks (authorized):
 A. MIMIC Charlson WITHOUT age_score -> cci_na_excl_rheum -> refit M2/M3/M4
 B. eICU uniquepid -> independent patients / multi-stay / cluster-robust M1
 C. Clean ARD recode (drop 721/722, M47/M48, 711-719, bare 446) -> MIMIC clean ARD prevalence + ARD-adjusted FI OR
Incremental JSON save."""
import os, json, re, warnings, numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
from psycopg2.extras import execute_values

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
ROOT=ROOT; ANA=os.path.join(ROOT,"analysis"); OUT=os.path.join(ANA,"db_audit")
os.makedirs(OUT,exist_ok=True); CONN=dict(**_DB_AUTH)
def C(db):
    c=CONN.copy(); c["dbname"]=db; con=psycopg2.connect(**c)
    with con.cursor() as cur: cur.execute("SET statement_timeout='420s'")
    return con
RES={}
def save(): json.dump(RES,open(os.path.join(OUT,"rev3_db.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)

# ---------- A. MIMIC Charlson without age ----------
try:
    rev=pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv").dropna(subset=["fi_lab"]).copy(); rev["fi10"]=rev.fi_lab*10
    hids=tuple(int(x) for x in rev["hadm_id"].dropna().unique())
    con=C("mimiciv")
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE t(h bigint)"); execute_values(cur,"INSERT INTO t(h) VALUES %s",[(h,) for h in hids])
            cur.execute("""SELECT ch.hadm_id, ch.charlson_comorbidity_index cci, ch.rheumatic_disease rd, ch.age_score ag
                FROM mimiciv_derived.charlson ch JOIN t ON ch.hadm_id=t.h""")
            rows=cur.fetchall()
        con.commit()
    finally: con.close()
    ch=pd.DataFrame(rows,columns=["hadm_id","cci","rd","ag"])
    ch["cci_na_excl_rh"]=ch["cci"]-ch["ag"].fillna(0)-ch["rd"].fillna(0)
    rev=rev.merge(ch[["hadm_id","cci_na_excl_rh","ag","cci","rd"]],on="hadm_id",how="left")
    m=rev.dropna(subset=["sofa","oasis","sapsii","cci_na_excl_rh"]).copy()
    def orf(f):
        r=smf.logit(f,data=m).fit(disp=0); b,se=r.params["fi10"],r.bse["fi10"]
        return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"
    A={"n":len(m),
       "M2 +Charlson(no age)":orf("died ~ fi10 + age + male + cci_na_excl_rh"),
       "M3 +SOFA+OASIS":orf("died ~ fi10 + age + male + cci_na_excl_rh + sofa + oasis"),
       "M4 +vent/vaso/rrt":orf("died ~ fi10 + age + male + cci_na_excl_rh + sofa + oasis + vent + vaso + rrt"),
       "mean_age_score":round(float(ch["ag"].mean()),2),
       "note":"CCI total minus age_score minus rheumatic_disease"}
    RES["A_charlson_noage_mimic"]=A; print("A:",A)
except Exception as e:
    RES["A_charlson_noage_mimic"]={"ERROR":str(e)}; print("A ERR",e)
save()

# ---------- B. eICU uniquepid / cluster ----------
try:
    ana=pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy(); ana["stay_id"]=ana["stay_id"].astype(str); ana["fi10"]=ana.fi_lab*10
    ids=tuple(int(x) for x in ana["stay_id"])
    con=C("eicu")
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE t(s bigint)"); execute_values(cur,"INSERT INTO t(s) VALUES %s",[(i,) for i in ids])
            cur.execute("SELECT p.patientunitstayid, p.uniquepid FROM eicu_crd.patient p JOIN t ON p.patientunitstayid=t.s")
            rows=cur.fetchall()
        con.commit()
    finally: con.close()
    up=pd.DataFrame(rows,columns=["stay_id","uniquepid"]); up["stay_id"]=up["stay_id"].astype(str)
    ana=ana.merge(up,on="stay_id",how="left")
    nst=len(ana); npat=ana["uniquepid"].nunique()
    r=smf.logit("died ~ fi10 + age + male",data=ana).fit(disp=0,cov_type="cluster",cov_kwds={"groups":ana["uniquepid"]})
    b,se=r.params["fi10"],r.bse["fi10"]
    B={"stays":nst,"unique_patients":int(npat),"extra_multi":int(nst-npat),"pct_multi":round(100*(nst-npat)/nst,2),
       "M1_cluster_OR":f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"}
    RES["B_eicu_uniquepid"]=B; print("B:",B)
except Exception as e:
    RES["B_eicu_uniquepid"]={"ERROR":str(e)}; print("B ERR",e)
save()

# ---------- C. Clean ARD (MIMIC + NWICU counts; MIMIC ARD-adjusted FI OR) ----------
CLEAN=[("SLE",r"^(7100|7101)$",r"^(M32)"),("SSc",r"^(7101)$",r"^(M34)"),("Sjogren",r"^(7102)$",r"^(M350)"),
 ("PMDM",r"^(7103|7104)$",r"^(M33)"),("RA",r"^(7140)$",r"^(M05|M06)"),("ANCA",r"^(4464|44621)$",r"^(M313|M300|M3181)"),
 ("APS",r"^(28981|79579)$",r"^(D6861|M3181)"),("MCTD",r"^(7109)$",r"^(M351)"),("Behcet",r"^(1361)$",r"^(M352)"),
 ("GCA",r"^(4465)$",r"^(M315|M316)"),("PMR",r"^(725)$",r"^(M353)"),("JIA",r"^(6960|0993|7142)$",r"^(M07|M02|M08|M09)"),
 ("SpA(ASonly)",r"^(720)$",r"^(M45|M46)"),("OtherCTD",r"^(7100)$",r"^(M35|M36)")]  # dropped 721/722,M47/M48,711-719,bare446,VascNOS
def clean_flags(df):
    df=df.copy(); df["icd_code"]=df["icd_code"].astype(str).str.strip(); df["icd_version"]=pd.to_numeric(df["icd_version"],errors="coerce")
    anym=pd.Series(False,index=df.index)
    for _,r9,r10 in CLEAN:
        anym |= ((df.icd_version==9)&df.icd_code.str.match(r9,na=False)) | ((df.icd_version==10)&df.icd_code.str.match(r10,na=False))
    return set(df.loc[anym,"hadm_id"].unique())
def fetch_dx(db,schema,keycol,ids):
    con=C(db)
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE t(s bigint)"); execute_values(cur,"INSERT INTO t(s) VALUES %s",[(int(i),) for i in ids])
            cur.execute(f"""SELECT d.hadm_id, d.icd_code, d.icd_version FROM {schema}.diagnoses_icd d
                JOIN t ON d.hadm_id=t.s""")
            rows=cur.fetchall()
        con.commit()
    finally: con.close()
    return pd.DataFrame(rows,columns=["hadm_id","icd_code","icd_version"])
try:
    Cc={}
    for db,schema in [("mimiciv","mimiciv_hosp"),("nwicu","hosp")]:
        ana=pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
        hadm=ana["hadm_id"].dropna().unique()
        dx=fetch_dx(db,schema,"hadm_id",hadm); clean=clean_flags(dx)
        Cc[db+"_clean_ARD"]=len(clean)
        if db=="mimiciv":
            ana["ard_clean"]=ana["hadm_id"].isin(clean).astype(int)
            m=ana.dropna(subset=["fi_lab","died","age","male"]).copy(); m["fi10"]=m.fi_lab*10
            r=smf.logit("died ~ fi10 + age + male + ard_clean",data=m).fit(disp=0)
            b,se=r.params["fi10"],r.bse["fi10"]
            ba_,sea,pa=r.params["ard_clean"],r.bse["ard_clean"],r.pvalues["ard_clean"]
            Cc["mimic_FI_OR_adj_cleanARD"]=f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"
            Cc["mimic_cleanARD_OR"]=f"{np.exp(ba_):.3f} ({np.exp(ba_-1.96*sea):.3f}-{np.exp(ba_+1.96*sea):.3f})"
            Cc["mimic_cleanARD_prev_pct"]=round(100*m["ard_clean"].mean(),2)
    RES["C_clean_ARD"]=Cc; print("C:",Cc)
except Exception as e:
    RES["C_clean_ARD"]={"ERROR":str(e)}; print("C ERR",e)
save()
print("\nSAVED rev3_db.json")
