"""
Augment analysis_*.csv with severity covariates for revision figures:
  MIMIC-IV : first_day_sofa, oasis, sapsii, charlson(-rheum), vent/vaso/rrt
  eICU      : apachepatientresult (apachescore, apache_acs, predictedmortality)
  NWICU     : Charlson computed from diagnoses_icd (Quan 2005, rheumatic excluded)
Writes analysis/rev_{db}.csv with full covariate set.
"""
import psycopg2, pandas as pd, numpy as np, os, re
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
ANA = os.path.join(ROOT, 'analysis')
os.makedirs(f"{ANA}/rev", exist_ok=True)

def q(db, sql):
    c = psycopg2.connect(dbname=db, **PG)
    try: return pd.read_sql(sql, c)
    finally: c.close()

# ---- Quan 2005 ICD-10 Charlson mapping (rheumatologic categories deliberately excluded) ----
# (category -> [icd10 prefixes]) ; we EXCLUDE rheumatic disease (M05/M06/M32/M34/M33/M35/M31...)
QUAN = {
 "myocardial_infarction": ["I21","I22","I252"],
 "congestive_heart_failure": ["I099","I110","I130","I132","I255","I42","I43","I50","P290"],
 "peripheral_vascular": ["I70","I71","I731","I738","I739","I771","I79","I83","I85","K551","K558","K559","Z958","Z959"],
 "cerebrovascular": ["G45","G46","I60","I61","I62","I63","I64","I65","I66","I67","I68","I69","H340"],
 "dementia": ["F00","F01","F02","F03","G30","F338","F339"],
 "chronic_pulmonary": ["I27","J40","J41","J42","J43","J44","J45","J46","J47","J60","J61","J62","J63","J64","J65","J66","J67","J684","J701","J703"],
 "mild_liver": ["B18","K70","K713","K714","K715","K717","K73","K74","K760","K762","K763","K764","K768","K769","Z944"],
 "diabetes_no_comp": ["E100","E101","E106","E108","E109","E110","E111","E116","E118","E119"],
 "diabetes_comp": ["E102","E103","E104","E105","E107","E112","E113","E114","E115","E117"],
 "hemiplegia": ["G81","G82","K560","K561","K562","K563","K564","K565","K566","K568","K569"],
 "moderate_severe_renal": ["I12","I13","N18","N19","N250","Z49","Z940","Z992"],
 "malignancy": ["C00","C01","C02","C03","C04","C05","C06","C07","C08","C09","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C30","C31","C32","C33","C34","C37","C38","C39","C40","C41","C43","C45","C46","C47","C48","C49","C50","C51","C52","C53","C54","C55","C56","C57","C58","C60","C61","C62","C63","C64","C65","C66","C67","C68","C69","C70","C71","C72","C73","C74","C75","C76","C77","C78","C79","C80","C81","C82","C83","C84","C85","C88","C90","C91","C92","C93","C94","C95","C96","C97"],
 "metastatic": ["C77","C78","C79","C80"],
 "hiv": ["B20","B21","B22","B23","B24"],
 "moderate_severe_liver": ["K70","K711","K712","K716","K718","K719","K72","K76","K77"],
}
WEIGHT = {
 "myocardial_infarction":1,"congestive_heart_failure":1,"peripheral_vascular":1,"cerebrovascular":1,
 "dementia":1,"chronic_pulmonary":1,"mild_liver":1,"diabetes_no_comp":1,"diabetes_comp":2,
 "hemiplegia":2,"moderate_severe_renal":2,"malignancy":2,"metastatic":6,"hiv":6,
 "moderate_severe_liver":3,
}

def nwicu_charlson():
    a = pd.read_csv(f"{ANA}/analysis_nwicu.csv")
    hids = tuple(int(x) for x in a["hadm_id"].unique())
    d = q("nwicu", f"SELECT hadm_id, icd_code, icd_version FROM hosp.diagnoses_icd WHERE hadm_id IN {hids}")
    def score_for(hid):
        sub = d[d["hadm_id"]==hid]
        s = 0
        for _, r in sub.iterrows():
            code = str(r["icd_code"]); ver = r["icd_version"]
            if ver == 10:
                for cat, pres in QUAN.items():
                    if any(code.startswith(p) for p in pres):
                        s += WEIGHT[cat]; break
            else:
                # ICD-9: map a few common; rheumatologic (710/711/712/713/714/446/136/725/720/721/722) EXCLUDED
                try: icd9 = int(re.sub('[^0-9]','', code))
                except: continue
                if 410<=icd9<=414 or icd9==4292: s+=1
                elif icd9 in (428,): s+=1
                elif 440<=icd9<=444 or icd9 in (441,7939,4471,449,5571,5579,4439,440): s+=1
                elif 430<=icd9<=438: s+=1
                elif icd9 in (290,): s+=1
                elif 490<=icd9<=496 or 500<=icd9<=505 or 5064: s+=1
                elif icd9 in (5712,5714,5715,5716,5718,5719): s+=1
                elif icd9 in (25000,25001,25002,25003,25010,25011,25012,25013,25020,25021,25022,25023,25030,25031,25032,25033,25040,25041,25042,25043,25050,25051,25052,25053,25060,25061,25062,25063,25070,25071,25072,25073): s+=1
                elif icd9 in (25040,25041,25042,25043,25050,25051,25052,25053,25060,25061,25062,25063,25070,25071,25072,25073): s+=1
                elif icd9 in (342,343,3441,3442,3443,3444,3445,3446,3449): s+=2
                elif icd9 in (582,5830,5831,5832,5833,5834,5835,5836,5837,585,586,5880,5888,5889,V420,V451,V56): s+=2
                elif 140<=icd9<=195 or 200<=icd9<=208 or icd9==2386: s+=2
                elif icd9 in (196,197,198,199): s+=6
                elif icd9 in (420,421,422,423,424,425): s+=1
        return s
    a["cci_excl_rheum"] = a["hadm_id"].map(score_for).fillna(0).astype(int)
    return a

def main():
    # ---------- MIMIC-IV ----------
    m = pd.read_csv(f"{ANA}/analysis_mimiciv.csv")
    sids = tuple(int(x) for x in m["stay_id"].unique())
    hids = tuple(int(x) for x in m["hadm_id"].unique())
    sofa = q("mimiciv", f"SELECT stay_id, sofa FROM mimiciv_derived.first_day_sofa WHERE stay_id IN {sids}")
    oasis = q("mimiciv", f"SELECT stay_id, oasis FROM mimiciv_derived.oasis WHERE stay_id IN {sids}")
    saps = q("mimiciv", f"SELECT stay_id, sapsii FROM mimiciv_derived.sapsii WHERE stay_id IN {sids}")
    ch = q("mimiciv", f"SELECT hadm_id, charlson_comorbidity_index AS cci, rheumatic_disease FROM mimiciv_derived.charlson WHERE hadm_id IN {hids}")
    ch["cci_excl_rheum"] = (ch["cci"] - ch["rheumatic_disease"].fillna(0)).astype(int)
    vent = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS vent FROM mimiciv_derived.ventilation WHERE stay_id IN {sids} AND ventilation_status IN ('InvasiveVent','Tracheostomy')")
    vaso = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS vaso FROM mimiciv_derived.vasoactive_agent WHERE stay_id IN {sids}")
    rrt = q("mimiciv", f"SELECT DISTINCT stay_id, 1 AS rrt FROM mimiciv_derived.rrt WHERE stay_id IN {sids}")
    for tbl,col in [(sofa,"sofa"),(oasis,"oasis"),(saps,"sapsii")]:
        m[col]=m["stay_id"].map(tbl.set_index("stay_id")[col])
    m["cci_excl_rheum"]=m["hadm_id"].map(ch.set_index("hadm_id")["cci_excl_rheum"])
    for tbl,col in [(vent,"vent"),(vaso,"vaso"),(rrt,"rrt")]:
        m[col]=m["stay_id"].map(tbl.set_index("stay_id")[col]).fillna(0).astype(int)
    m["apache"]=np.nan
    m.to_csv(f"{ANA}/rev/rev_mimiciv.csv", index=False)
    print("MIMIC rev:", m.shape, "cci_excl missing", m["cci_excl_rheum"].isna().sum())

    # ---------- eICU ----------
    e = pd.read_csv(f"{ANA}/analysis_eicu.csv")
    sids = tuple(int(x) for x in e["stay_id"].unique())
    ap = q("eicu", f"SELECT patientunitstayid AS stay_id, apachescore, acutephysiologyscore AS aps, predictedhospitalmortality AS apache_pred FROM eicu_crd.apachepatientresult WHERE patientunitstayid IN {sids}")
    ap = ap.sort_values("apachescore").groupby("stay_id", as_index=False).last()
    e = e.merge(ap, on="stay_id", how="left")
    e = e.rename(columns={"apachescore":"apache","aps":"apsiii"})
    e["sofa"]=np.nan; e["oasis"]=np.nan; e["sapsii"]=np.nan; e["cci_excl_rheum"]=np.nan
    e["vent"]=np.nan; e["vaso"]=np.nan; e["rrt"]=np.nan
    e.to_csv(f"{ANA}/rev/rev_eicu.csv", index=False)
    print("eICU rev:", e.shape, "apache missing", e["apache"].isna().sum())

    # ---------- NWICU ----------
    n = nwicu_charlson()
    n["sofa"]=np.nan; n["oasis"]=np.nan; n["sapsii"]=np.nan
    n["apache"]=np.nan; n["apsiii"]=np.nan
    n["vent"]=np.nan; n["vaso"]=np.nan; n["rrt"]=np.nan
    n.to_csv(f"{ANA}/rev/rev_nwicu.csv", index=False)
    print("NWICU rev:", n.shape, "cci_excl missing", n["cci_excl_rheum"].isna().sum())

    print("AUGMENT DONE")

if __name__ == "__main__":
    main()
