# -*- coding: utf-8 -*-
"""DB audit part 1: (a) ARD per-category counts among study cohort (3 DBs),
(b) NWICU/eICU lab-item presence for the 100%-missing analytes (RDW/MCV/INR/LDH/fibrinogen/CK),
(c) NWICU COVID diagnosis presence. Incremental save -> analysis/db_audit/."""
import os, json, re, traceback
import pandas as pd, psycopg2

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


ROOT=ROOT; ANA=os.path.join(ROOT,"analysis"); OUT=os.path.join(ANA,"db_audit")
os.makedirs(OUT,exist_ok=True)
CONN=dict(**_DB_AUTH)
def C(db):
    c=CONN.copy(); c["dbname"]=db; con=psycopg2.connect(**c)
    with con.cursor() as cur: cur.execute("SET statement_timeout='420s'")
    return con
RES={}

# ---- ARD categories (build_analysis.py, reused) ----
ARD=[
 ("SLE", r"^(7100|7101)$", r"^(M32)"),("SSc", r"^(7101)$", r"^(M34)"),
 ("Sjogren", r"^(7102)$", r"^(M350)"),("PMDM", r"^(7103|7104)$", r"^(M33)"),
 ("RA", r"^(7140)$", r"^(M05|M06)"),("ANCA", r"^(4464|44621)$", r"^(M313|M300|M3181)"),
 ("APS", r"^(28981|79579)$", r"^(D6861|M3181)"),("MCTD", r"^(7109)$", r"^(M351)"),
 ("Behcet", r"^(1361)$", r"^(M352)"),("GCA", r"^(4465)$", r"^(M315|M316)"),
 ("PMR", r"^(725)$", r"^(M353)"),("JIA_Reactive", r"^(6960|0993|7142)$", r"^(M07|M02|M08|M09)"),
 ("SpA", r"^(720|721|722)$", r"^(M45|M46|M47|M48)"),("OtherCTD", r"^(710|711|712|713|719)$", r"^(M35|M36)"),
 ("VascNOS", r"^(446)$", r"^(M31)"),
]
FLAWED={"SpA","OtherCTD","VascNOS"}  # groups containing non-autoimmune code ranges

def classify(df):
    """df: hadm_id, icd_code, icd_version. Return dict cat->set(hadm), plus invalid-only set."""
    df=df.copy()
    df["icd_code"]=df["icd_code"].astype(str).str.strip()
    df["icd_version"]=pd.to_numeric(df["icd_version"],errors="coerce")
    cats={}
    for name,r9,r10 in ARD:
        m9=(df.icd_version==9)&df.icd_code.str.match(r9,na=False)
        m10=(df.icd_version==10)&df.icd_code.str.match(r10,na=False)
        cats[name]=set(df.loc[m9|m10,"hadm_id"].unique())
    anyall=set().union(*cats.values()) if cats else set()
    clean=set().union(*[v for k,v in cats.items() if k not in FLAWED]) if any(k not in FLAWED for k in cats) else set()
    flawed_only=anyall-clean
    return cats,anyall,clean,flawed_only

def ard_mimic_like(db,schema,tag):
    coh=pd.read_csv(f"{ANA}/analysis_{db}.csv")
    hadm=tuple(int(x) for x in coh["hadm_id"].dropna().unique())
    con=C(db)
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_coh(hadm_id bigint)")
            from psycopg2.extras import execute_values
            execute_values(cur,"INSERT INTO tmp_coh(hadm_id) VALUES %s",[(h,) for h in hadm])
            cur.execute(f"""SELECT d.hadm_id, d.icd_code, d.icd_version
                FROM {schema}.diagnoses_icd d JOIN tmp_coh c USING(hadm_id)""")
            rows=cur.fetchall()
        con.commit()
    finally:
        con.close()
    df=pd.DataFrame(rows,columns=["hadm_id","icd_code","icd_version"])
    cats,anyall,clean,flawed=classify(df)
    out={"cohort_hadm":len(hadm),"dx_rows":len(df)}
    for name,_,_ in ARD:
        out[name]=len(cats[name])
    out["ARD_union_flagged"]=len(anyall)
    out["ARD_clean_only"]=len(clean)
    out["ARD_flawed_only(721/722|711-719|446)"]=len(flawed)
    print(f"[{tag}]",out)
    return out

def ard_eicu():
    coh=pd.read_csv(f"{ANA}/analysis_eicu.csv")
    ids=tuple(int(x) for x in coh["stay_id"].dropna().unique())
    con=C("eicu")
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE tmp_coh(sid bigint)")
            from psycopg2.extras import execute_values
            execute_values(cur,"INSERT INTO tmp_coh(sid) VALUES %s",[(h,) for h in ids])
            cur.execute("""SELECT d.patientunitstayid, d.icd9code, lower(d.diagnosisstring)
                FROM eicu_crd.diagnosis d JOIN tmp_coh c ON d.patientunitstayid=c.sid""")
            rows=cur.fetchall()
        con.commit()
    finally:
        con.close()
    df=pd.DataFrame(rows,columns=["sid","icd9code","dx"])
    res={"cohort_stays":len(ids),"dx_rows":len(df)}
    for name,r9,_ in ARD:
        m=df.icd9code.astype(str).str.match(r9,na=False)
        res[name]=int(df.loc[m,"sid"].nunique())
    res["ARD_icd9_union"]=int(df.loc[df.apply(lambda r:any(re.match(p,str(r.icd9code or '')) for _,p,_ in ARD),axis=1),"sid"].nunique())
    # text fallback (approx per reviewer)
    txt=df.dx.fillna("")
    textpat=r"lupus|rheumat|scleroderma|systemic sclerosis|sjogren|sjögren|myosit|vasculit|wegener|granulomat|behcet|antiphospholip|mixed connective|spondyloarth|psoriatic|giant cell|polymyalg|ankylosing"
    res["ARD_text_union"]=int(df.loc[txt.str.contains(textpat,na=False),"sid"].nunique())
    print("[eICU]",res); return res

def nwicu_items():
    con=C("nwicu")
    try:
        with con.cursor() as cur:
            # target itemids used in mapping
            ids=[100008,100017,100030,100055,100048,100085,51277,51250,100021,100002]
            cur.execute(f"SELECT itemid,count(*) FROM hosp.labevents WHERE itemid IN {tuple(ids)} GROUP BY itemid")
            r=cur.fetchall()
            # also how many distinct itemids total & top counts
            cur.execute("SELECT count(DISTINCT itemid) FROM hosp.labevents")
            nitem=cur.fetchone()[0]
        con.commit()
    finally:
        con.close()
    print("[NWICU itemids present]",r,"distinct itemids=",nitem)
    return {"present":{int(a):int(b) for a,b in r},"distinct_itemids":int(nitem)}

def eicu_labnames():
    con=C("eicu")
    try:
        with con.cursor() as cur:
            names=("RDW","MCV","PT - INR","LDH","fibrinogen","CPK","INR","PT")
            cur.execute("SELECT lower(labname),count(*) FROM eicu_crd.lab WHERE lower(labname) IN %s GROUP BY lower(labname)",(tuple(n.lower() for n in names),))
            r=cur.fetchall()
        con.commit()
    finally:
        con.close()
    print("[eICU labnames]",r); return {a:int(b) for a,b in r}

def nwicu_covid():
    con=C("nwicu")
    try:
        with con.cursor() as cur:
            cur.execute("""SELECT count(DISTINCT hadm_id) FROM hosp.diagnoses_icd
                WHERE (icd_version=10 AND icd_code LIKE 'U07%') OR (icd_version=9 AND icd_code LIKE '078%')""")
            n=cur.fetchone()[0]
            cur.execute("SELECT count(DISTINCT hadm_id) FROM hosp.diagnoses_icd")
            tot=cur.fetchone()[0]
        con.commit()
    finally:
        con.close()
    print(f"[NWICU COVID] covid_hadm={n} / total_hadm={tot}")
    return {"covid_hadm":int(n),"total_hadm":int(tot)}

for name,fn in [("ard_mimic",lambda:ard_mimic_like("mimiciv","mimiciv_hosp","MIMIC-IV")),
                ("ard_nwicu",lambda:ard_mimic_like("nwicu","hosp","NWICU")),
                ("ard_eicu",ard_eicu),("nwicu_items",nwicu_items),
                ("eicu_labnames",eicu_labnames),("nwicu_covid",nwicu_covid)]:
    try:
        RES[name]=fn()
    except Exception as e:
        RES[name]={"ERROR":str(e)}; print(name,"ERROR",e); traceback.print_exc()
    json.dump(RES,open(os.path.join(OUT,"db_audit1.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
print("\nSAVED",os.path.join(OUT,"db_audit1.json"))
