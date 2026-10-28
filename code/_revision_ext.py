import os, warnings
import numpy as np, pandas as pd, psycopg2
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

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
ROOT=ROOT; ANA=f"{ROOT}/analysis"; OUT=f"{ANA}/revision"
CONN=dict(**_DB_AUTH)
def q(db,sql):
    c=psycopg2.connect(dbname=db,**CONN)
    try: return pd.read_sql(sql,c)
    finally: c.close()

def fit_or(df,f):
    m=smf.logit(f,data=df).fit(disp=0)
    b=m.params.get("fi10");se=m.bse.get("fi10");p=m.pvalues.get("fi10")
    return dict(OR=round(np.exp(b),3),lo=round(np.exp(b-1.96*se),3),hi=round(np.exp(b+1.96*se),3),
                p=p,auc=round(roc_auc_score(df["died"],m.predict(df)),3),n=len(df))

# ---- eICU APACHE 调整 ----
eic=pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
eic["fi10"]=eic["fi_lab"]*10
sids=tuple(int(x) for x in eic["stay_id"].unique())
ap=q("eicu",f"""SELECT patientunitstayid AS stay_id, MAX(apachescore) AS apache, MAX(acutephysiologyscore) AS aps
                FROM eicu_crd.apachepatientresult WHERE patientunitstayid IN {sids} AND apachescore>=0 GROUP BY 1""")
eic["apache"]=eic["stay_id"].map(ap.set_index("stay_id")["apache"])
eic_cc=eic.dropna(subset=["apache"]).copy()
rows=[]
rows.append(dict(cohort="eICU",model="base age+sex+ard",**fit_or(eic_cc,"died ~ fi10 + age + male + ard")))
rows.append(dict(cohort="eICU",model="+APACHE IV",**fit_or(eic_cc,"died ~ fi10 + age + male + ard + apache")))
print(f"eICU with APACHE: {len(eic_cc)}/{len(eic)}, APACHE median={eic_cc.apache.median()}")

# ---- NWICU Charlson (Quan ICD-10 compact) ----
nwi=pd.read_csv(f"{ANA}/analysis_nwicu.csv").dropna(subset=["fi_lab"]).copy()
nwi["fi10"]=nwi["fi_lab"]*10
hids=tuple(int(x) for x in nwi["hadm_id"].unique())
dx=q("nwicu",f"SELECT hadm_id, icd_code, icd_version FROM hosp.diagnoses_icd WHERE hadm_id IN {hids}")
dx["icd_code"]=dx["icd_code"].astype(str).str.upper().str.replace(".","",regex=False)
# Quan 2005 ICD-10 前缀（去风湿项 M05/M06/M32/M34/M33 等不计入，以避免与 ARD 共线）
quan10={
 "mi":["I21","I22","I252"],"chf":["I099","I110","I130","I132","I255","I420","I425","I426","I427","I428","I429","I43","I50","P290"],
 "pvd":["I70","I71","I731","I738","I739","I771","I790","I792","K551","K558","K559","Z958","Z959"],
 "cevd":["G45","G46","H340","I60","I61","I62","I63","I64","I65","I66","I67","I68","I69"],
 "dementia":["F00","F01","F02","F03","F051","G30","G311"],
 "cpd":["I278","I279","J40","J41","J42","J43","J44","J45","J46","J47","J60","J61","J62","J63","J64","J65","J66","J67","J684","J701","J703"],
 "pud":["K25","K26","K27","K28"],
 "mld":["B18","K700","K701","K702","K703","K709","K713","K714","K715","K717","K73","K74","K760","K762","K763","K764","K768","K769","Z944"],
 "diab":["E100","E101","E106","E108","E109","E110","E111","E116","E118","E119","E120","E121","E126","E128","E129","E130","E131","E136","E138","E139","E140","E141","E146","E148","E149"],
 "diabc":["E102","E103","E104","E105","E107","E112","E113","E114","E115","E117","E122","E123","E124","E125","E127","E132","E133","E134","E135","E137","E142","E143","E144","E145","E147"],
 "hp":["G041","G114","G801","G802","G81","G82","G830","G831","G832","G833","G834","G839"],
 "renal":["I120","I131","N032","N033","N034","N035","N036","N037","N052","N053","N054","N055","N056","N057","N18","N19","N250","Z490","Z491","Z492","Z940","Z992"],
 "canc":["C0","C1","C2","C3","C40","C41","C43","C45","C46","C47","C48","C49","C5","C6","C70","C71","C72","C73","C74","C75","C76","C81","C82","C83","C84","C85","C88","C90","C91","C92","C93","C94","C95","C96","C97"],
 "msld":["I850","I859","I864","I982","K704","K711","K721","K729","K765","K766","K767"],
 "mst":["C77","C78","C79","C80"],"aids":["B20","B21","B22","B24"]}
w={"mi":1,"chf":1,"pvd":1,"cevd":1,"dementia":1,"cpd":1,"pud":1,"mld":1,"diab":1,"diabc":2,"hp":2,"renal":2,"canc":2,"msld":3,"mst":6,"aids":6}
def charlson(codes):
    s=0
    for cat,pre in quan10.items():
        if any(c.startswith(tuple(pre)) for c in codes): s+=w[cat]
    return s
cci=dx[dx.icd_version==10].groupby("hadm_id")["icd_code"].apply(lambda x:charlson(list(x)))
nwi["cci_excl_rheum"]=nwi["hadm_id"].map(cci).fillna(0)
rows.append(dict(cohort="NWICU",model="base age+sex+ard",**fit_or(nwi,"died ~ fi10 + age + male + ard")))
rows.append(dict(cohort="NWICU",model="+Charlson(-rheum)",**fit_or(nwi,"died ~ fi10 + age + male + ard + cci_excl_rheum")))
print(f"NWICU Charlson median={nwi.cci_excl_rheum.median()}")
ext=pd.DataFrame(rows); ext.to_csv(f"{OUT}/severity_adjust_ext_dbs.csv",index=False)
print("\n=== eICU/NWICU 严重程度调整 ==="); print(ext.to_string(index=False))

# ---- 纳入流程/attrition counts ----
# MIMIC: 老年(>=65) 首次ICU 住院总数
attr=[]
tot_m=q("mimiciv","""WITH firsticu AS (SELECT stay_id, subject_id, hadm_id, ROW_NUMBER() OVER(PARTITION BY subject_id ORDER BY intime) rn, intime
  FROM mimiciv_icu.icustays)
  SELECT COUNT(*) n FROM firsticu f JOIN mimiciv_hosp.patients p ON f.subject_id=p.subject_id
  JOIN mimiciv_hosp.admissions a ON f.hadm_id=a.hadm_id
  WHERE f.rn=1 AND (p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year) >= 65""").iloc[0,0]
attr.append(dict(db="MIMIC-IV",step="老年(>=65)首次ICU住院",n=int(tot_m)))
attr.append(dict(db="MIMIC-IV",step="FI-Lab可评估(>=15项)",n=45656))
attr.append(dict(db="MIMIC-IV",step="排除(FI不可评估)",n=int(tot_m)-45656))
ext2=pd.DataFrame(attr); ext2.to_csv(f"{OUT}/attrition_counts.csv",index=False)
print("\n=== 纳入流程(MIMIC) ==="); print(ext2.to_string(index=False))
