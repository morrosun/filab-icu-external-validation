# -*- coding: utf-8 -*-
"""第七轮：NWICU 死亡日期逐规则重算（以实际日期为准，不再沿用旧缓存口径）"""
import json
import os
import pandas as pd
import psycopg2

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


PG = dict(**_DB_AUTH)
OUT = "analysis/db_audit/nwicu_dod_flags.csv"

def conn(db):
    return psycopg2.connect(dbname=db, **PG)

a = pd.read_csv("analysis/analysis_nwicu.csv")
a["stay_id"] = a["stay_id"].astype(str)
keep = a.dropna(subset=["fi_lab"])[["stay_id", "subject_id", "died"]].copy()
print("FI-assessable stays:", len(keep), "cached died=1:", int(keep.died.sum()))

q = """
SELECT i.stay_id::text AS stay_id, i.hadm_id::text AS hadm_id,
       a.admittime, a.dischtime, p.dod,
       a.hospital_expire_flag
FROM icu.icustays i
JOIN hosp.admissions a ON i.hadm_id = a.hadm_id
JOIN hosp.patients  p ON i.subject_id = p.subject_id
"""
with conn("nwicu") as c:
    d = pd.read_sql_query(q, c)
for col in ["admittime", "dischtime", "dod"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")
print("raw rows:", len(d))

m = keep.merge(d, on="stay_id", how="left")
print("merged:", len(m), "missing dates:", int(m.admittime.isna().sum()))

adm, dis, dod = m.admittime, m.dischtime, m.dod
rules = {
    "A_current_no_lower_plus1d": dod.notna() & (dod <= dis + pd.Timedelta(days=1)),
    "B_no_lower_to_disch":       dod.notna() & (dod <= dis),
    "C_clean_adm_to_disch":      dod.notna() & (dod >= adm) & (dod <= dis),
    "D_clean_adm_to_disch_plus1d": dod.notna() & (dod >= adm) & (dod <= dis + pd.Timedelta(days=1)),
    "E_dod_after_disch_anytime": dod.notna() & (dod > dis + pd.Timedelta(days=1)),
}
res = {}
for k, v in rules.items():
    res[k] = int(v.sum())
    print(f"{k:30} {res[k]:6d}")

neg = dod.notna() & (dod < adm)
print("\ndod < admittime:", int(neg.sum()))
gap = (adm - dod)[neg].dt.total_seconds() / 86400.0
print("  gap days: median %.2f  p90 %.2f  max %.2f" % (gap.median(), gap.quantile(.9), gap.max()))
print("  <=0.5d:", int((gap <= 0.5).sum()), " 0.5-1d:", int(((gap > 0.5) & (gap <= 1)).sum()),
      " 1-2d:", int(((gap > 1) & (gap <= 2)).sum()), " 2-7d:", int(((gap > 2) & (gap <= 7)).sum()),
      " >7d:", int((gap > 7).sum()))
# 同日入院同日死亡（dod == admittime date）是否算清洁
print("  dod == admittime(date):", int((dod.dt.normalize() == adm.dt.normalize()).sum()))

# 51 例中 dod 是否落在同一患者其他住院内
oth = d[d.dod.notna()][["subject_id" if "subject_id" in d.columns else "stay_id", "hadm_id", "admittime", "dischtime"]].copy() if False else None
q2 = """
SELECT i.subject_id::text AS subject_id, i.hadm_id::text AS hadm_id, a.admittime, a.dischtime
FROM icu.icustays i JOIN hosp.admissions a ON i.hadm_id = a.hadm_id
"""
with conn("nwicu") as c:
    ad = pd.read_sql_query(q2, c)
ad["admittime"] = pd.to_datetime(ad.admittime, errors="coerce")
ad["dischtime"] = pd.to_datetime(ad.dischtime, errors="coerce")
mm = keep.merge(d[["stay_id", "hadm_id", "admittime", "dischtime", "dod"]], on="stay_id", how="left")
bad = mm[dod.reindex(mm.index).notna()] if False else mm[mm.dod.notna() & (mm.dod < mm.admittime)]
n_in_other = 0
for _, r in bad.iterrows():
    sub = ad[(ad.subject_id == str(r["subject_id"])) & (ad.hadm_id != str(r["hadm_id"]))]
    dd = pd.Timestamp(r["dod"])
    if ((sub.admittime <= dd) & (sub.dischtime >= dd)).any():
        n_in_other += 1
print("  of these, dod falls inside another admission of same patient:", n_in_other, "/", len(bad))

# 保存逐例标记（清洁规则 + 强制排除明显错误）
m["died_A"] = rules["A_current_no_lower_plus1d"].astype(int)
m["died_C"] = rules["C_clean_adm_to_disch"].astype(int)
m["dod_before_adm"] = neg.astype(int)
m[["stay_id", "died", "died_A", "died_C", "dod_before_adm", "admittime", "dischtime", "dod"]].to_csv(OUT, index=False)
print("\nwrote", OUT)
print("cached died == died_A ?", bool((m["died"].fillna(0).astype(int) == m["died_A"]).all()))
json.dump({"rules": res, "n": int(len(m)),
           "gap_median_d": round(float(gap.median()), 2),
           "gap_p90_d": round(float(gap.quantile(.9)), 2),
           "gap_max_d": round(float(gap.max()), 1),
           "gap_le_0_5": int((gap <= 0.5).sum()),
           "gap_0_5_1": int(((gap > 0.5) & (gap <= 1)).sum()),
           "gap_1_2": int(((gap > 1) & (gap <= 2)).sum()),
           "gap_2_7": int(((gap > 2) & (gap <= 7)).sum()),
           "gap_gt_7": int((gap > 7).sum()),
           "in_other_admission": n_in_other,
           "dod_eq_adm_date": int((dod.dt.normalize() == adm.dt.normalize()).sum())},
          open("analysis/db_audit/nwicu_dod_rules.json", "w"), indent=2)
print("json written")
