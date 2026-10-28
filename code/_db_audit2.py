# -*- coding: utf-8 -*-
"""DB audit part 2: (a) icu_los for cohort (3 DBs); (b) MIMIC early-death sensitivity done RIGHT
(exclude in-hospital death with icu_los<1d) + count of <24h deaths; (c) threshold >=20 subset;
(d) NWICU RDW/MCV missingness with CORRECT itemids (100008/100017) vs INR/LDH truly absent."""
import os, json, numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf

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
os.makedirs(OUT,exist_ok=True); CONN=dict(**_DB_AUTH)
def C(db):
    c=CONN.copy(); c["dbname"]=db; con=psycopg2.connect(**c)
    with con.cursor() as cur: cur.execute("SET statement_timeout='450s'")
    return con
def icu_los(db, table, key, expr):
    coh=pd.read_csv(f"{ANA}/analysis_{db}.csv")
    ids=tuple(int(x) for x in coh[key].dropna().unique())
    con=C(db)
    try:
        with con.cursor() as cur:
            cur.execute("CREATE TEMP TABLE t(sid bigint)")
            from psycopg2.extras import execute_values
            execute_values(cur,"INSERT INTO t(sid) VALUES %s",[(i,) for i in ids])
            cur.execute(f"SELECT c.sid, {expr} FROM t c JOIN {table} x ON x.{key}=c.sid")
            rows=cur.fetchall()
        con.commit()
    finally: con.close()
    df=pd.DataFrame(rows,columns=["sid","los"]).groupby("sid")["los"].max()
    print(f"[{db}] icu_los mapped {df.notna().sum()}/{len(ids)}")
    return df

RES={}
# (a) icu_los
mim_los = icu_los("mimiciv","mimiciv_icu.icustays","stay_id","EXTRACT(EPOCH FROM (x.outtime-x.intime))/86400.0")
nwi_los = icu_los("nwicu","icu.icustays","stay_id","EXTRACT(EPOCH FROM (x.outtime-x.intime))/86400.0")
RES["icu_los_mapped"]={"mimic":int(mim_los.notna().sum()),"nwicu":int(nwi_los.notna().sum())}

# (b) MIMIC early-death sensitivity (full cohort) + <24h death count
m=pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
m["fi10"]=m.fi_lab*10; m["los"]=m.stay_id.map(mim_los)
def orf(df,form="died ~ fi10 + age + male + ard"):
    r=smf.logit(form,data=df).fit(disp=0); b,se=r.params["fi10"],r.bse["fi10"]
    return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})",len(df)
full_or,_=orf(m)
died24 = m[(m.died==1)&(m.los<1)]
m_ex = m[~((m.died==1)&(m.los<1))]
ex_or,n_ex=orf(m_ex)
# threshold >=20 (valid subset from analysis cohort, all >=15)
m20=m[m.n_assessed>=20]; th20,_=orf(m20)
RES["mimic_sensitivity"]={
 "full":"n=%d OR=%s"%(len(m),full_or),
 "died_within24h_count":int(len(died24)),
 "died_within24h_pct":round(100*len(died24)/len(m),3),
 "exclude_death_lt24h":"n=%d OR=%s"%(n_ex,ex_or),
 "n_assessed_ge20":"n=%d OR=%s"%(len(m20),th20),
 "n_assessed_ge15":"n=%d OR=%s"%(len(m),full_or),
 "note_ge10":"cannot reconstruct from >=15 cohort (needs parent cohort rebuild)"}
print("[MIMIC sens]",RES["mimic_sensitivity"])

# NWICU early-death too
n=pd.read_csv(f"{ANA}/analysis_nwicu.csv").dropna(subset=["fi_lab"]).copy()
n["fi10"]=n.fi_lab*10; n["los"]=n.stay_id.map(nwi_los)
def orfn(df): 
    r=smf.logit("died ~ fi10 + age + male + ard",data=df).fit(disp=0); b,se=r.params["fi10"],r.bse["fi10"]
    return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"
nd24=n[(n.died==1)&(n.los<1)]
n_ex=n[~((n.died==1)&(n.los<1))]
RES["nwicu_sensitivity"]={"full_OR":orfn(n),"died_within24h":int(len(nd24)),"excl_OR":orfn(n_ex),"n_excl":len(n_ex)}
print("[NWICU sens]",RES["nwicu_sensitivity"])

# (d) NWICU RDW/MCV corrected missingness (itemid 100008=RDW,100017=MCV) in first 24h vs INR/LDH
con=C("nwicu")
try:
    coh=pd.read_csv(f"{ANA}/analysis_nwicu.csv"); ids=tuple(int(x) for x in coh["stay_id"].dropna().unique())
    with con.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t(sid bigint)")
        from psycopg2.extras import execute_values
        execute_values(cur,"INSERT INTO t(sid) VALUES %s",[(i,) for i in ids])
        cur.execute("""SELECT
           COUNT(DISTINCT CASE WHEN le.itemid=100008 THEN i.stay_id END) rdw,
           COUNT(DISTINCT CASE WHEN le.itemid=100017 THEN i.stay_id END) mcv,
           COUNT(DISTINCT CASE WHEN le.itemid=100030 THEN i.stay_id END) inr100030,
           COUNT(DISTINCT CASE WHEN le.itemid=100055 THEN i.stay_id END) ldh100055
           FROM t
           JOIN icu.icustays i ON i.stay_id=t.sid
           JOIN hosp.labevents le ON le.subject_id=i.subject_id AND le.hadm_id=i.hadm_id
             AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
             AND le.valuenum IS NOT NULL
             AND le.itemid IN (100008,100017,100030,100055)""")
        r=cur.fetchone()
    con.commit()
finally: con.close()
n_tot=len(ids)
RES["nwicu_rdw_mcv_corrected"]={"N":n_tot,"rdw_assessed(100008)":int(r[0]),"mcv_assessed(100017)":int(r[1]),
  "inr_assessed(100030)":int(r[2]),"ldh_assessed(100055)":int(r[3]),
  "rdw_miss%":round(100*(1-r[0]/n_tot),1),"mcv_miss%":round(100*(1-r[1]/n_tot),1)}
print("[NWICU RDW/MCV corrected]",RES["nwicu_rdw_mcv_corrected"])

# also list top NWICU lab itemids to locate true INR/LDH itemids
con=C("nwicu")
try:
    with con.cursor() as cur:
        cur.execute("SELECT itemid,count(*) FROM hosp.labevents GROUP BY itemid ORDER BY count(*) DESC LIMIT 40")
        top=cur.fetchall()
    con.commit()
finally: con.close()
RES["nwicu_top_itemids"]={int(a):int(b) for a,b in top}
print("[NWICU top40 itemids]",top)

json.dump(RES,open(os.path.join(OUT,"db_audit2.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
print("\nSAVED db_audit2.json")
