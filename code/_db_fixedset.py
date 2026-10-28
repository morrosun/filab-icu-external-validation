# -*- coding: utf-8 -*-
"""三库共同高覆盖固定项目集敏感性分析
固定集 = 三库缺失率均 <=20% 的 11 项常规化验:
  sodium, potassium, chloride, bicarbonate, BUN, creatinine,
  glucose, calcium, hematocrit, hemoglobin, platelet
覆盖规则(预设): 11 项中 >=9 项已检测 (>=82%) 才计算固定集 FI。
输出: analysis/db_audit/fixedset.json  + 控制台摘要
"""
import os, json, sys, time
import numpy as np, pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

_DB_AUTH = dict(
    host=os.environ.get("FILAB_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("FILAB_DB_PORT", "5432")),
    user=os.environ.get("FILAB_DB_USER", "postgres"),
    password=os.environ.get("FILAB_DB_PASSWORD", ""),
)


OUT = "analysis/db_audit/fixedset.json"
os.makedirs("analysis/db_audit", exist_ok=True)
CONN = dict(**_DB_AUTH)

FIXED = {
    # name: (mimic_first_day_lab_col, nwicu_itemid, eicu_labname, lo, hi)
    "sodium":      ("sodium",      100010, "sodium",            135.0, 145.0),
    "potassium":   ("potassium",   100011, "potassium",           3.5,   5.5),
    "chloride":    ("chloride",    100012, "chloride",           98.0, 110.0),
    "bicarbonate": ("bicarbonate", 100013, "bicarbonate",        22.0,  29.0),
    "bun":         ("bun",         100004, "BUN",                 7.0,  20.0),
    "creatinine":  ("creatinine",  100002, "creatinine",          0.5,   1.3),
    "glucose":     ("glucose",     100001, "glucose",            70.0, 140.0),
    "calcium":     ("calcium",     100003, "calcium",             8.5,  10.5),
    "hematocrit":  ("hematocrit",  100006, "Hct",                36.0,  50.0),
    "hemoglobin":  ("hemoglobin",  100007, "Hgb",                10.0,  17.0),
    "platelet":    ("platelets",   100014, "platelets x 1000",  150.0, 450.0),
}
ITEMS = list(FIXED.keys())
MIN_FIXED = 9   # >=9/11 assessed

def conn(db):
    c = dict(CONN); c["dbname"] = db
    return psycopg2.connect(**c)

def chunks(seq, n=1500):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def q(db, sql, ids=None, chunk=True):
    """按 id 分批查询后拼接；sql 必须含 {IDLIST} 占位"""
    if not chunk:
        with conn(db) as c:
            with c.cursor() as cur:
                cur.execute("SET statement_timeout='600s'")
            return pd.read_sql_query(sql.format(IDLIST=ids), c)
    parts = []
    for ch in chunks(ids):
        lit = "(" + ",".join("'%s'" % x for x in ch) + ")"
        with conn(db) as c:
            with c.cursor() as cur:
                cur.execute("SET statement_timeout='600s'")
            parts.append(pd.read_sql_query(sql.format(IDLIST=lit), c))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def fi_mimic_fixed(ids):
    cols = ", ".join([f"{FIXED[a][0]}_min AS {a}_min, {FIXED[a][0]}_max AS {a}_max" for a in ITEMS])
    sql = f"SELECT stay_id, {cols} FROM mimiciv_derived.first_day_lab WHERE stay_id IN {{IDLIST}}"
    # mimiciv first_day_lab stay_id 是 bigint → 用数字字面量
    parts = []
    ids = list(ids)
    for ch in chunks(ids, 1500):
        lit = "(" + ",".join(str(int(x)) for x in ch) + ")"
        with conn("mimiciv") as c:
            with c.cursor() as cur:
                cur.execute("SET statement_timeout='600s'")
            parts.append(pd.read_sql_query(sql.format(IDLIST=lit), c))
    df = pd.concat(parts, ignore_index=True)
    df["stay_id"] = df["stay_id"].astype(str)
    return df

def fi_nwicu_fixed(ids):
    itemids = tuple(FIXED[a][1] for a in ITEMS)
    case = ", ".join([f"MAX(CASE WHEN le.itemid={FIXED[a][1]} THEN le.valuenum END) {a}_max, "
                      f"MIN(CASE WHEN le.itemid={FIXED[a][1]} THEN le.valuenum END) {a}_min" for a in ITEMS])
    sql = f"""
        SELECT i.stay_id, {case}
        FROM icu.icustays i
        JOIN hosp.labevents le ON le.subject_id=i.subject_id AND le.hadm_id=i.hadm_id
         AND le.charttime BETWEEN i.intime AND i.intime+interval '24 hour'
         AND le.valuenum IS NOT NULL AND le.itemid IN {itemids}
        WHERE i.stay_id IN {{IDLIST}} GROUP BY i.stay_id"""
    df = q("nwicu", sql, ids)
    df["stay_id"] = df["stay_id"].astype(str)
    return df

def fi_eicu_fixed(ids):
    names = tuple(FIXED[a][2] for a in ITEMS)
    case = ", ".join([f"MAX(CASE WHEN labname='{FIXED[a][2]}' THEN labresult END) {a}_max, "
                      f"MIN(CASE WHEN labname='{FIXED[a][2]}' THEN labresult END) {a}_min" for a in ITEMS])
    nl = "(" + ",".join("'%s'" % n for n in names) + ")"
    sql = f"""
        SELECT patientunitstayid AS stay_id, {case}
        FROM eicu_crd.lab
        WHERE patientunitstayid IN {{IDLIST}} AND labresultoffset BETWEEN 0 AND 1440
          AND labresult IS NOT NULL AND labname IN {nl}
        GROUP BY patientunitstayid"""
    df = q("eicu", sql, ids)
    df["stay_id"] = df["stay_id"].astype(str)
    return df

def compute_fixed(L):
    defs = []
    for a in ITEMS:
        lo, hi = FIXED[a][3], FIXED[a][4]
        mn, mx = L.get(f"{a}_min"), L.get(f"{a}_max")
        assessed = mn.notna() | mx.notna()
        defs.append(((mn < lo) | (mx > hi)).fillna(False).where(assessed))
    M = pd.concat(defs, axis=1)
    an = M.notna().sum(axis=1)
    dn = M.sum(axis=1, min_count=1).fillna(0)
    fi = np.full(len(L), np.nan)
    m = (an >= MIN_FIXED).values
    fi[m] = (dn[m] / an[m]).values
    return pd.Series(fi, index=L.index), an

def calib(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    auc = roc_auc_score(y, p); brier = brier_score_loss(y, p)
    lp = np.log(np.clip(p, 1e-9, 1-1e-9) / (1 - np.clip(p, 1e-9, 1-1e-9)))
    X = sm.add_constant(lp)
    cm = sm.Logit(y, X).fit(disp=0)
    return dict(n=int(len(y)), deaths=int(y.sum()), auc=round(float(auc), 4),
                brier=round(float(brier), 4),
                oe=round(float(y.mean() / p.mean()), 4),
                cal_int=round(float(cm.params[0]), 4),
                cal_slope=round(float(cm.params[1]), 4))

def main():
    res = {"min_fixed": MIN_FIXED, "items": ITEMS}
    data = {}
    for db in ["mimiciv", "nwicu", "eicu"]:
        t0 = time.time()
        base = pd.read_csv(f"analysis/analysis_{db}.csv")
        base["stay_id"] = base["stay_id"].astype(str)
        ids = list(base["stay_id"].unique())
        print(f"[{db}] {len(ids)} stays, 提取 11 项...", flush=True)
        if db == "mimiciv":
            L = fi_mimic_fixed(ids)
        elif db == "nwicu":
            L = fi_nwicu_fixed(ids)
        else:
            L = fi_eicu_fixed(ids)
        fi, na = compute_fixed(L)
        L = L.copy(); L["fi_fixed"] = fi.values; L["n_fixed"] = na.values
        d = base.merge(L[["stay_id", "fi_fixed", "n_fixed"]], on="stay_id", how="left")
        data[db] = d
        ok = d["fi_fixed"].notna()
        res[db] = {
            "n_all": int(len(d)),
            "n_fixed_computable": int(ok.sum()),
            "pct_computable": round(float(ok.mean() * 100), 2),
            "n_assessed_median": float(d.loc[ok, "n_fixed"].median()),
            "n_assessed_p10": float(d.loc[ok, "n_fixed"].quantile(0.10)),
            "fi_median": round(float(d.loc[ok, "fi_fixed"].median()), 4),
            "fi_iqr": [round(float(d.loc[ok, "fi_fixed"].quantile(0.25)), 4),
                       round(float(d.loc[ok, "fi_fixed"].quantile(0.75)), 4)],
            "corr_with_fullFI": round(float(d[["fi_fixed", "fi_lab"]].corr().iloc[0, 1]), 4),
            "secs": round(time.time() - t0, 1),
        }
        print(f"[{db}] 完成 {res[db]}", flush=True)

    # ---- 模型: M1 died ~ fi10 + age + male (固定集) ----
    res["models"] = {}
    for db, d in data.items():
        dd = d.dropna(subset=["fi_fixed", "died", "age", "male"]).copy()
        dd["fi10f"] = dd["fi_fixed"] * 10
        m = smf.logit("died ~ fi10f + age + male", data=dd).fit(disp=0)
        b, se = m.params["fi10f"], m.bse["fi10f"]
        res["models"][db] = {
            "n": int(m.nobs),
            "OR": round(float(np.exp(b)), 4),
            "lo": round(float(np.exp(b - 1.96 * se)), 4),
            "hi": round(float(np.exp(b + 1.96 * se)), 4),
            "p": float(m.pvalues["fi10f"]),
            "auc_apparent": round(float(roc_auc_score(dd["died"], m.predict(dd))), 4),
            "coef": {k: round(float(v), 6) for k, v in m.params.items()},
        }
        print(f"[{db}] M1(fixed) OR={res['models'][db]['OR']} "
              f"({res['models'][db]['lo']}-{res['models'][db]['hi']}) "
              f"AUC={res['models'][db]['auc_apparent']}", flush=True)

    # ---- 冻结系数外验: MIMIC 固定集系数 -> NWICU/eICU ----
    res["ext"] = {}
    c = res["models"]["mimiciv"]["coef"]
    for db in ["mimiciv", "nwicu", "eicu"]:
        d = data[db].dropna(subset=["fi_fixed", "died", "age", "male"]).copy()
        lp = c["Intercept"] + c["fi10f"] * d["fi_fixed"] * 10 + c["age"] * d["age"] + c["male"] * d["male"]
        p = 1 / (1 + np.exp(-lp))
        res["ext"][db] = calib(d["died"], p)
        print(f"[{db}] ext(fixed) {res['ext'][db]}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nWROTE", OUT)

if __name__ == "__main__":
    main()
