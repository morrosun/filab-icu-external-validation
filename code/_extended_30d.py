"""
Recompute 30-day mortality and 30-day readmission for MIMIC-IV and NWICU,
restricted to the FI-evaluable analysis cohort. FI-Lab per 0.1 OR adjusted for
age + male + ard (base covariates, consistent with revised manuscript 3.3).
eICU has no post-discharge follow-up -> in-hospital only.
Output: analysis/revision/extended_outcomes_30d.csv
"""
import psycopg2, pandas as pd, numpy as np, statsmodels.formula.api as smf
import os
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



ANA = os.path.join(ROOT, 'analysis')
OUT = os.path.join(ROOT, 'analysis', 'revision')

def conn(db):
    return psycopg2.connect(dbname=db, **_DB_AUTH)

def fit_or(df, outcome):
    d = df.dropna(subset=[outcome, "fi10", "age", "male", "ard"]).copy()
    d[outcome] = d[outcome].astype(int)
    if d[outcome].sum() < 10 or d[outcome].nunique() < 2:
        return None
    m = smf.logit(f"{outcome} ~ fi10 + age + male + ard", data=d).fit(disp=0)
    p = m.predict(d)
    auc = roc_auc_score(d[outcome], p)
    ci = m.conf_int().loc["fi10"]
    return dict(n=int(m.nobs), events=int(d[outcome].sum()),
                rate=round(d[outcome].mean()*100, 1),
                OR=round(np.exp(m.params["fi10"]), 3),
                lo=round(np.exp(ci[0]), 3), hi=round(np.exp(ci[1]), 3),
                p=m.pvalues["fi10"], auc=round(auc, 3))

def mimic_like(db, hsp, icu):
    a = pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy()
    a["stay_id"] = a["stay_id"].astype(str)
    a["fi10"] = a["fi_lab"] * 10
    # 用分析队列中的权威院内死亡定义(died)做生存者排除，避免 hsp.hospital_expire_flag 缺失导致再入院n失真
    hadm = tuple(a["hadm_id"].dropna().astype(int).astype(str).unique())
    subj = tuple(a["subject_id"].dropna().astype(int).astype(str).unique())
    with conn(db) as c:
        # index admission times + ICU intime + dod
        idx = pd.read_sql_query(f"""
            SELECT i.stay_id, i.hadm_id, i.subject_id, i.intime,
                   ad.admittime, ad.dischtime,
                   p.dod
            FROM {icu}.icustays i
            JOIN {hsp}.admissions ad ON i.hadm_id=ad.hadm_id
            JOIN {hsp}.patients p ON i.subject_id=p.subject_id
            WHERE i.hadm_id IN {hadm}""", c)
        # all admissions per subject (for readmission)
        alladm = pd.read_sql_query(f"""
            SELECT subject_id, hadm_id, admittime, dischtime
            FROM {hsp}.admissions WHERE subject_id IN {subj}""", c)
    idx["stay_id"] = idx["stay_id"].astype(str)
    for col in ["intime", "admittime", "dischtime", "dod"]:
        idx[col] = pd.to_datetime(idx[col], errors="coerce")
    alladm["admittime"] = pd.to_datetime(alladm["admittime"], errors="coerce")
    alladm["dischtime"] = pd.to_datetime(alladm["dischtime"], errors="coerce")

    # 30-day mortality from ICU admission
    idx["death30"] = ((idx["dod"].notna()) &
                      (idx["dod"] <= idx["intime"] + pd.Timedelta(days=30))).astype(int)

    # 30-day readmission: among IN-HOSPITAL SURVIVORS (analysis 'died'==0),
    # any later admission with admittime in (index dischtime, dischtime+30d]
    a_idx = a[["stay_id", "died"]].drop_duplicates("stay_id")
    idx = idx.merge(a_idx, on="stay_id", how="left")
    surv = idx[idx["died"] == 0].copy()
    readm = {}
    adm_by_subj = {s: g for s, g in alladm.groupby("subject_id")}
    for _, r in surv.iterrows():
        g = adm_by_subj.get(r["subject_id"])
        flag = 0
        if g is not None:
            later = g[(g["admittime"] > r["dischtime"]) &
                      (g["admittime"] <= r["dischtime"] + pd.Timedelta(days=30))]
            flag = int(len(later) > 0)
        readm[r["stay_id"]] = flag
    idx["readmit30"] = idx["stay_id"].map(readm)  # NaN for non-survivors (died==1)

    m = a.merge(idx[["stay_id", "death30", "readmit30"]], on="stay_id", how="left")
    return m

def main():
    rows = []
    for db, hsp, icu, label in [("mimiciv", "mimiciv_hosp", "mimiciv_icu", "MIMIC-IV"),
                                 ("nwicu", "hosp", "icu", "NWICU")]:
        m = mimic_like(db, hsp, icu)
        print(f"\n===== {label} (n={len(m)}) =====")
        # in-hospital
        r_in = fit_or(m, "died"); r_in["db"] = label; r_in["outcome"] = "In-hospital mortality"; rows.append(r_in)
        r_d30 = fit_or(m, "death30"); r_d30["db"] = label; r_d30["outcome"] = "30-day mortality"; rows.append(r_d30)
        r_re = fit_or(m, "readmit30")
        if r_re: r_re["db"] = label; r_re["outcome"] = "30-day readmission"; rows.append(r_re)
        for r in [r_in, r_d30, r_re]:
            if r: print(r)
    # eICU in-hospital only
    e = pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
    e["fi10"] = e["fi_lab"] * 10
    r = fit_or(e, "died"); r["db"] = "eICU"; r["outcome"] = "In-hospital mortality"; rows.append(r)
    print("\neICU in-hospital:", r)

    df = pd.DataFrame(rows)[["db", "outcome", "n", "events", "rate", "OR", "lo", "hi", "p", "auc"]]
    df["p_fmt"] = df["p"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    df.to_csv(f"{OUT}/extended_outcomes_30d.csv", index=False)
    print("\nSaved extended_outcomes_30d.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
