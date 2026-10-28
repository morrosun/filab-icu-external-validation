"""
Direction B — Cohort extraction layer.
Builds the "elderly (>=65) autoimmune rheumatic disease (ARD) ICU" cohort
across three PostgreSQL databases: mimiciv, nwicu, eicu.
Outputs per-database patient-stay level CSVs + a combined summary.

ARD definition: ICD-9/10 prefix list (harmonized across MIMIC-IV & NWICU);
eICU additionally matched by diagnosis-string text (its icd9code is ICD-9 only).
"""
import os
import psycopg2
import pandas as pd

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



OUT = os.path.join(ROOT, 'cohort')
os.makedirs(OUT, exist_ok=True)

PG = dict(**_DB_AUTH)

# ---- ARD disease groups: (label, icd9_regex, icd10_regex) ----
# Broad autoimmune rheumatic disease battery. Narrow later if needed.
ARD = [
    ("SLE",              r"^(7100)$",                       r"^(M32)"),
    ("SystemicSclerosis",r"^(7101)$",                       r"^(M34)"),
    ("Sjogren",          r"^(7102)$",                       r"^(M350)"),
    ("Myositis",         r"^(7103|7104)$",                  r"^(M33)"),
    ("RA",               r"^(7140)$",                       r"^(M05|M06)"),
    ("ANCA_Vasculitis",  r"^(4464|44621)$",                 r"^(M313|M300|M3181)"),
    ("APS",              r"^(28981|79579)$",                r"^(D6861|M3181)"),
    ("MCTD",             r"^(7109)$",                       r"^(M351)"),
    ("Behcet",           r"^(1361)$",                       r"^(M352)"),
    ("RelapsingPolycho", r"^(73399|446?)$",                 r"^(M941)"),
    ("GCA_TAK",          r"^(4465)$",                       r"^(M315|M316)"),
    ("PMR",              r"^(725)$",                        r"^(M353)"),
    ("Psoriatic_IA",     r"^(6960)$",                       r"^(M07)"),
    ("Reactive_JIA",     r"^(0993|7142|6960)$",            r"^(M02|M08|M09)"),
    ("Spondylo_Arth",    r"^(720|721|722)$",                r"^(M45|M46|M47|M48)"),
    ("Other_CTD",        r"^(710|711|712|713|719)$",        r"^(M35|M36)"),
]
# eICU text fallback (diagnosisstring)
EICU_TEXT = [
    "lupus", "scleroderma", "systemic sclerosis", "sjogren", "sjögren",
    "dermatomyositis", "polymyositis", "myositis", "rheumatoid arthritis",
    "rheumatoid", "vasculitis", "wegener", "granulomatosis", "churg",
    "polyarteritis", "takayasu", "behcet", "antiphospholipid", "aps syndrome",
    "mixed connective", "spondyloarth", "psoriatic", "giant cell arteritis",
    "polymyalgia", "relapsing polychondritis", "ankylosing spondyl",
]


def icd_where(col_v="d.icd_version", col_c="d.icd_code"):
    parts = []
    for label, r9, r10 in ARD:
        parts.append(f"({col_v}=9 AND {col_c} ~ '{r9}')")
        parts.append(f"({col_v}=10 AND {col_c} ~ '{r10}')")
    return " OR ".join(parts)


def icd9_where(col_c="dx.icd9code"):
    """eICU diagnosis table is ICD-9 only (no version column)."""
    parts = [f"({col_c} ~ '{r9}')" for label, r9, r10 in ARD]
    return " OR ".join(parts)


def conn(db):
    return psycopg2.connect(dbname=db, **PG)


def extract_mimic_like(db, hosp_schema, icu_schema, died_expr="a.hospital_expire_flag"):
    """MIMIC-IV and NWICU share structure. Use base tables for robustness.
    died_expr: SQL expression evaluating to in-hospital death flag.
      MIMIC uses reliable hospital_expire_flag; NWICU's flag is broken
      (only 0.17% populated) so we fall back to patients.dod vs discharge.
    """
    where = icd_where()
    sql = f"""
    WITH ard_hadm AS (
        SELECT DISTINCT d.hadm_id
        FROM {hosp_schema}.diagnoses_icd d
        WHERE {where}
    ),
    base AS (
        SELECT
            i.stay_id, i.subject_id, i.hadm_id,
            i.intime AS icu_intime, i.outtime AS icu_outtime,
            a.admittime, a.dischtime, ({died_expr}) AS hospital_expire_flag,
            p.gender, a.race,
            (p.anchor_age + (EXTRACT(year FROM a.admittime)::int - p.anchor_year::int)) AS age
        FROM {icu_schema}.icustays i
        JOIN {hosp_schema}.admissions a ON i.hadm_id = a.hadm_id
        JOIN {hosp_schema}.patients p  ON i.subject_id = p.subject_id
        WHERE i.hadm_id IN (SELECT hadm_id FROM ard_hadm)
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY icu_intime) AS rn_icu
        FROM base
    )
    SELECT
        stay_id, subject_id, hadm_id,
        age, gender, race,
        hospital_expire_flag AS died,
        EXTRACT(EPOCH FROM (icu_outtime - icu_intime))/86400.0 AS icu_los_days,
        EXTRACT(EPOCH FROM (dischtime - admittime))/86400.0 AS hosp_los_days
    FROM ranked
    WHERE rn_icu = 1 AND age >= 65
    """
    with conn(db) as c:
        return pd.read_sql_query(sql, c)


def extract_eicu():
    where = icd9_where(col_c="dx.icd9code")
    text_or = " OR ".join([f"dx.diagnosisstring ILIKE '%{t}%'" for t in EICU_TEXT])
    sql = f"""
    WITH ard_stay AS (
        SELECT DISTINCT dx.patientunitstayid
        FROM eicu_crd.diagnosis dx
        WHERE {where} OR {text_or}
    ),
    base AS (
        SELECT
            p.patientunitstayid, p.uniquepid, p.patienthealthsystemstayid,
            NULLIF(regexp_replace(p.age, '[^0-9]', '', 'g'), '')::int AS age, p.gender,
            p.hospitaldischargestatus,
            p.hospitaladmitoffset, p.hospitaldischargeoffset, p.unitdischargeoffset
        FROM eicu_crd.patient p
        JOIN ard_stay a ON p.patientunitstayid = a.patientunitstayid
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY patienthealthsystemstayid ORDER BY unitdischargeoffset) AS rn
        FROM base
    )
    SELECT
        patientunitstayid AS stay_id,
        uniquepid AS subject_id,
        patienthealthsystemstayid AS hadm_id,
        age, gender,
        CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS died,
        NULL::float AS icu_los_days,
        (hospitaldischargeoffset - hospitaladmitoffset)/1440.0 AS hosp_los_days
    FROM ranked
    WHERE rn = 1 AND age >= 65 AND age < 300
    """
    with conn("eicu") as c:
        return pd.read_sql_query(sql, c)


def summarize(df, name):
    n = len(df)
    deaths = int(df["died"].sum())
    icu_los = df["icu_los_days"].dropna()
    return {
        "database": name,
        "n_stays": n,
        "n_deaths": deaths,
        "mortality_pct": round(100 * deaths / n, 1) if n else 0,
        "median_age": round(df["age"].median(), 1) if n else None,
        "median_icu_los": round(icu_los.median(), 1) if len(icu_los) else None,
        "pct_female": round(100 * (df["gender"].astype(str).str.lower().isin(["f", "female"]).mean()), 1) if n else None,
    }


if __name__ == "__main__":
    mimic = extract_mimic_like("mimiciv", "mimiciv_hosp", "mimiciv_icu")
    mimic["database"] = "MIMIC-IV"
    # NWICU hospital_expire_flag is unreliable -> derive in-hospital death from dod
    nw_died = "(CASE WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime + INTERVAL '1 day' THEN 1 ELSE 0 END)"
    nwicu = extract_mimic_like("nwicu", "hosp", "icu", died_expr=nw_died)
    nwicu["database"] = "NWICU"
    eicu = extract_eicu()
    eicu["database"] = "eICU"

    for df, nm in [(mimic, "mimiciv"), (nwicu, "nwicu"), (eicu, "eicu")]:
        df.to_csv(os.path.join(OUT, f"cohort_{nm}.csv"), index=False)
        print(f"[{nm}] rows={len(df)} deaths={int(df['died'].sum())} median_age={df['age'].median():.1f}")

    summary = pd.DataFrame([summarize(mimic, "MIMIC-IV"),
                            summarize(nwicu, "NWICU"),
                            summarize(eicu, "eICU")])
    summary.to_csv(os.path.join(OUT, "cohort_summary.csv"), index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
