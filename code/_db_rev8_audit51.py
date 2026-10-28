# -*- coding: utf-8 -*-
"""第八轮核查：51 例「死亡日期早于入院」逐条留痕。
输出 analysis/db_audit/nwicu_dod_51_audit.csv + nwicu_dod_51_summary.json
字段：raw dod / raw admittime / 日期化后的 dod 与 admittime / 旧规则(时刻比较)判定 / 新规则(日期比较)判定
"""
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
f = pd.read_csv("analysis/db_audit/nwicu_dod_flags.csv")
f["stay_id"] = f["stay_id"].astype(str)
for c in ["admittime", "dischtime", "dod"]:
    f[c] = pd.to_datetime(f[c], errors="coerce")

with psycopg2.connect(dbname="nwicu", **PG) as c:
    it = pd.read_sql_query("SELECT stay_id::text AS stay_id, intime, outtime FROM icu.icustays", c)
it["stay_id"] = it["stay_id"].astype(str)
for c in ["intime", "outtime"]:
    it[c] = pd.to_datetime(it[c], errors="coerce")
f = f.merge(it, on="stay_id", how="left")

f["adm_date"] = f["admittime"].dt.normalize()
f["dis_date"] = f["dischtime"].dt.normalize()
f["dod_date"] = f["dod"].dt.normalize()

# 旧（上一版）判定：把 dod（日期，零点）与 admittime（时刻）直接比较
f["old_rule_flag"] = (f["dod"] < f["admittime"]) & f["dod"].notna()
# 新判定：日期层面
f["same_calendar_date"] = (f["dod_date"] == f["adm_date"])
f["new_rule_flag"] = (f["dod_date"] < f["adm_date"]) & f["dod"].notna()
# 旧版"严格区间"（按时刻）为何得到 1,391：dod >= admittime
f["strict_ts_ok"] = f["dod"].notna() & (f["dod"] >= f["admittime"]) & (f["dod"] <= f["dischtime"])

bad = f[f["old_rule_flag"]].copy()
bad["gap_days"] = (bad["admittime"] - bad["dod"]).dt.total_seconds() / 86400.0
bad["dod_date_minus_adm_date_days"] = (bad["adm_date"] - bad["dod_date"]).dt.days
bad["classified"] = bad.apply(
    lambda r: ("同日死亡（日期戳零点造成的假阳性）" if r["same_calendar_date"]
               else "真实早于入院日（结局不确定，剔除）"), axis=1)

cols = ["stay_id", "admittime", "dod", "adm_date", "dod_date", "gap_days",
        "dod_date_minus_adm_date_days", "same_calendar_date", "old_rule_flag",
        "new_rule_flag", "strict_ts_ok", "classified"]
bad[cols].sort_values("gap_days").to_csv("analysis/db_audit/nwicu_dod_51_audit.csv", index=False)

summ = dict(
    n_old_flag=int(len(bad)),
    n_same_calendar_date=int(bad["same_calendar_date"].sum()),
    n_truly_before_adm_date=int(bad["new_rule_flag"].sum()),
    truly_bad_rows=bad[bad["new_rule_flag"]][["stay_id", "admittime", "dod",
                                              "dod_date_minus_adm_date_days"]].astype(str).to_dict("records"),
    # 计数闭合校验
    died_no_lower_bound=int((f["dod"].notna() & (f["dod"] <= f["dischtime"])).sum()),
    died_strict_timestamp=int(f["strict_ts_ok"].sum()),
    died_date_level=int((f["dod"].notna() & (f["dod_date"] >= f["adm_date"]) & (f["dod_date"] <= f["dis_date"])).sum()),
    gap_stats=dict(median=round(float(bad["gap_days"].median()), 3),
                   p90=round(float(bad["gap_days"].quantile(.9)), 3),
                   max=round(float(bad["gap_days"].max()), 3)),
)
json.dump(summ, open("analysis/db_audit/nwicu_dod_51_summary.json", "w"), indent=2, ensure_ascii=False)

print("[51 例分解] 旧判定 51 = 同日死亡 %d + 真实早于入院日 %d" %
      (summ["n_same_calendar_date"], summ["n_truly_before_adm_date"]))
print("[闭合] 无下界 %d ; 时刻严格区间 %d ; 日期层面 %d" %
      (summ["died_no_lower_bound"], summ["died_strict_timestamp"], summ["died_date_level"]))
print("[差值] median %.2f d  p90 %.2f d  max %.2f d" %
      (summ["gap_stats"]["median"], summ["gap_stats"]["p90"], summ["gap_stats"]["max"]))
print("[真实异常]")
for r in summ["truly_bad_rows"]:
    print("   ", r)
print("WROTE nwicu_dod_51_audit.csv / nwicu_dod_51_summary.json")
