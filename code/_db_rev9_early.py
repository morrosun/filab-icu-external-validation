# -*- coding: utf-8 -*-
"""第九轮：早期死亡敏感性按真实死亡时间（deathtime − ICU intime < 24 h）重算。
对照：旧（错误）口径 = died & icu_los < 1 天。
"""
import json
import os
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


PG = dict(**_DB_AUTH)

q = """
SELECT i.stay_id::text AS stay_id, i.subject_id::text AS subject_id,
       i.intime, i.outtime, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
"""
with psycopg2.connect(dbname="mimiciv", **PG) as c:
    d = pd.read_sql_query(q, c)
for col in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
    d[col] = pd.to_datetime(d[col], errors="coerce")

a = pd.read_csv("analysis/analysis_mimiciv.csv")
a["stay_id"] = a["stay_id"].astype(str)
d["stay_id"] = d["stay_id"].astype(str)
m = a.dropna(subset=["fi_lab"]).merge(
    d[["stay_id", "intime", "outtime", "deathtime", "hospital_expire_flag"]], on="stay_id", how="left")
m["fi10"] = m["fi_lab"] * 10
m["los_d"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 86400.0
m["death_minus_intime_d"] = (m["deathtime"] - m["intime"]).dt.total_seconds() / 86400.0

print("[deaths] died=%d ; deathtime available=%d" % (int(m.died.sum()), int(m.deathtime.notna().sum())))

# 旧（错误）口径
old_flag = (m["died"] == 1) & (m["los_d"] < 1)
# 正确口径：死亡发生在入 ICU 后 24 h 内（需要 deathtime）
new_flag = (m["died"] == 1) & m["death_minus_intime_d"].notna() & (m["death_minus_intime_d"] < 1)
# 死亡但无 deathtime（未捕获）者无法判定早死
undetermined = (m["died"] == 1) & m["deathtime"].isna()

print("[old rule] died & icu_los<1d  n=%d (%.2f%%)" % (int(old_flag.sum()), old_flag.mean() * 100))
print("[new rule] died & deathtime-intime<1d  n=%d (%.2f%%)" % (int(new_flag.sum()), new_flag.mean() * 100))
print("[undetermined] died but no deathtime  n=%d" % int(undetermined.sum()))
print("[gap stats] deathtime-intime among deaths: median %.2f d, p10 %.2f, p90 %.2f" % (
    m.loc[m.died == 1, "death_minus_intime_d"].median(),
    m.loc[m.died == 1, "death_minus_intime_d"].quantile(.1),
    m.loc[m.died == 1, "death_minus_intime_d"].quantile(.9)))


def fit(df, y="died"):
    dd = df.dropna(subset=[y, "fi10", "age", "male"]).copy()
    mod = smf.glm(f"{y} ~ fi10 + age + male", data=dd, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": dd["subject_id"].values})
    b, se = mod.params["fi10"], mod.bse["fi10"]
    return dict(n=int(len(dd)), deaths=int(dd[y].sum()),
                OR=round(float(np.exp(b)), 3),
                lo=round(float(np.exp(b - 1.96 * se)), 3),
                hi=round(float(np.exp(b + 1.96 * se)), 3),
                auc=round(float(roc_auc_score(dd[y], mod.predict(dd))), 3))


res = {}
res["primary"] = fit(m)
res["exclude_early_correct"] = fit(m[~new_flag])
res["exclude_early_old_invalid"] = fit(m[~old_flag])
res["counts"] = dict(old_rule=int(old_flag.sum()), new_rule=int(new_flag.sum()),
                     undetermined=int(undetermined.sum()),
                     old_pct=round(float(old_flag.mean() * 100), 2),
                     new_pct=round(float(new_flag.mean() * 100), 2))
for k, v in res.items():
    print(k, v)

json.dump(res, open("analysis/db_audit/rev9_early_death.json", "w"), indent=2)
print("WROTE rev9_early_death.json")
