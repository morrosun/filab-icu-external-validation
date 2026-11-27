# -*- coding: utf-8 -*-
"""Stage 7: harmonized-panel comparisons. Single source for every number in
Supplementary Tables S7, S9 and S11 and for the common-sample paragraph in Results 3.4.

(A) Coverage and correlation of the harmonized 14-item panel vs the 30-item FI-Lab.
(B) Common sample (both indices computable): 30-item coefficients from the full
    MIMIC-IV first-day cohort (= Supplementary Table S3); 14-item coefficients refitted
    on the MIMIC-IV harmonized-panel cohort (>=12 of 14 items). Frozen-coefficient
    discrimination/calibration (patient-level bootstrap, B=1000) and paired dAUC.
(C) 14/14 fully observed panel: cohort-specific cluster-robust ORs and frozen
    performance, with coefficients from the MIMIC-IV 14/14 subset (= Supplementary S11).
(D) NWICU death-definition comparison (primary vs no lower bound), cluster-robust ORs.

Outputs: analysis/stage7/stage7_common_sample.json
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from rev_common import (ANA, OUT, save_json, read_analysis, read_fixed14,
                        performance, paired_delta_auc, cluster_column, expit, fit_or)

STAGE = os.path.join(OUT, "stage7")
os.makedirs(STAGE, exist_ok=True)


def lp(coef, fi10, age, male):
    return coef["Intercept"] + coef["fi10"] * fi10 + coef["age"] * age + coef["male"] * male


RES = {"stage": "7"}

# ---------------------------------------------------------------- coefficients
m30 = read_analysis("mimiciv").dropna(subset=["fi10", "died", "age", "male"]).copy()
fit30 = smf.logit("died ~ fi10 + age + male", data=m30).fit(disp=0)
COEF30 = {k: float(v) for k, v in fit30.params.items()}
print("[coef30]", {k: round(v, 6) for k, v in COEF30.items()}, "n", int(fit30.nobs), flush=True)

m14 = read_fixed14("mimiciv")
m14_panel = m14.dropna(subset=["fi14", "died", "age", "male"]).copy()
m14_panel["fi10_14"] = m14_panel["fi14"] * 10
fit14 = smf.logit("died ~ fi10_14 + age + male", data=m14_panel).fit(disp=0)
COEF14 = {k.replace("fi10_14", "fi10"): float(v) for k, v in fit14.params.items()}
COEF14_SE = {k.replace("fi10_14", "fi10"): float(v) for k, v in fit14.bse.items()}
print("[coef14 >=12]", {k: round(v, 6) for k, v in COEF14.items()}, "n", int(fit14.nobs), flush=True)

m14_full = m14[(m14["n14"] == 14) & m14["died"].notna()].copy()
m14_full["fi10_14"] = m14_full["fi14"] * 10
fit14f = smf.logit("died ~ fi10_14 + age + male", data=m14_full).fit(disp=0)
COEF14F = {k.replace("fi10_14", "fi10"): float(v) for k, v in fit14f.params.items()}
print("[coef14 14/14]", {k: round(v, 6) for k, v in COEF14F.items()}, "n", int(fit14f.nobs), flush=True)

RES.update(coef30={k: round(v, 6) for k, v in COEF30.items()},
           coef14={k: round(v, 6) for k, v in COEF14.items()},
           coef14_se={k: round(v, 6) for k, v in COEF14_SE.items()},
           coef14_n=int(fit14.nobs),
           coef14_full={k: round(v, 6) for k, v in COEF14F.items()},
           coef14_full_n=int(fit14f.nobs))

# ---------------------------------------------------------------- (A)(B)
RES["coverage"], RES["corr"], RES["common"] = {}, {}, {}
for db in ["mimiciv", "nwicu", "eicu"]:
    a = read_analysis(db)
    f = read_fixed14(db)
    a_ok = a.dropna(subset=["fi_lab"])
    fa = f.merge(a_ok[["stay_id"]], on="stay_id", how="inner")
    ok14 = fa["fi14"].notna()
    RES["coverage"][db] = dict(n_cohort=int(len(a_ok)), n14=int(ok14.sum()),
                               pct=round(float(ok14.mean() * 100), 1),
                               fi14_median=round(float(fa.loc[ok14, "fi14"].median()), 4))

    d = a.merge(f[["stay_id", "fi14"]], on="stay_id", how="inner")
    d = d.dropna(subset=["fi_lab", "fi14", "died", "age", "male"]).copy()
    RES["corr"][db] = dict(
        n=int(len(d)),
        pearson=round(float(d[["fi_lab", "fi14"]].corr().iloc[0, 1]), 4),
        spearman=round(float(d[["fi_lab", "fi14"]].corr(method="spearman").iloc[0, 1]), 4),
        mad=round(float((d["fi14"] - d["fi_lab"]).abs().mean()), 4))

    d["fi10_14"] = d["fi14"] * 10
    d["_pid"] = cluster_column(db, d).values
    mm = smf.logit("died ~ fi10_14 + age + male", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["_pid"].values)[0]})
    b, se = float(mm.params["fi10_14"]), float(mm.bse["fi10_14"])
    local14 = dict(OR=round(float(np.exp(b)), 4), lo=round(float(np.exp(b - 1.96 * se)), 4),
                   hi=round(float(np.exp(b + 1.96 * se)), 4))

    y = d["died"].astype(int).values
    p30 = np.asarray(expit(lp(COEF30, d["fi_lab"].values * 10, d["age"].values, d["male"].values)))
    p14 = np.asarray(expit(lp(COEF14, d["fi14"].values * 10, d["age"].values, d["male"].values)))
    g = d["_pid"].values
    ext30 = performance(y, p30, g=g, B=1000)
    ext14 = performance(y, p14, g=g, B=1000)
    dauc = paired_delta_auc(y, p14, p30, g=g, B=1000)
    RES["common"][db] = dict(n=int(len(d)), deaths=int(y.sum()), local14=local14,
                             ext30=ext30, ext14=ext14, dAUC=dauc["delta_auc"],
                             dAUC_lo=dauc.get("delta_auc_lo"), dAUC_hi=dauc.get("delta_auc_hi"))
    print(f"[common {db}] n={len(d)} deaths={int(y.sum())} OR={local14['OR']} "
          f"AUC30={ext30['auc']} AUC14={ext14['auc']} dAUC={dauc['delta_auc']}", flush=True)

# ---------------------------------------------------------------- (C) 14/14 panel
RES["panel14of14"] = {}
for db in ["mimiciv", "nwicu", "eicu"]:
    d = read_fixed14(db)
    d = d[(d["n14"] == 14) & d["died"].notna()].copy()
    d["_pid"] = cluster_column(db, d).values
    d["fi10_14"] = d["fi14"] * 10
    mm = smf.logit("died ~ fi10_14 + age + male", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["_pid"].values)[0]})
    b, se = float(mm.params["fi10_14"]), float(mm.bse["fi10_14"])
    p = np.asarray(expit(lp(COEF14F, d["fi14"].values * 10, d["age"].values, d["male"].values)))
    perf = performance(d["died"].values.astype(int), p, g=d["_pid"].values, B=1000)
    RES["panel14of14"][db] = dict(
        n=int(len(d)), deaths=int(d["died"].sum()),
        OR=round(float(np.exp(b)), 4), lo=round(float(np.exp(b - 1.96 * se)), 4),
        hi=round(float(np.exp(b + 1.96 * se)), 4), perf=perf)
    print(f"[14/14 {db}] n={len(d)} OR={np.exp(b):.4f} AUC={perf['auc']}", flush=True)

# ---------------------------------------------------------------- (D) NWICU death definitions
RES["nwicu_definitions"] = {}
primary = read_analysis("nwicu").dropna(subset=["fi10", "age", "male", "died"]).copy()
primary["_pid"] = cluster_column("nwicu", primary).values
r1, _ = fit_or(primary, "died ~ fi10 + age + male", "fi10", cluster="_pid")
RES["nwicu_definitions"]["primary"] = dict(n=int(len(primary)), deaths=int(primary["died"].sum()),
                                           OR=r1["OR"], lo=r1["lo"], hi=r1["hi"])
raw = pd.read_csv(os.path.join(ANA, "analysis_nwicu.csv"))
raw["stay_id"] = raw["stay_id"].astype(str)
raw["fi10"] = raw["fi_lab"] * 10
raw = raw.dropna(subset=["fi10", "age", "male", "died"]).copy()
raw["_pid"] = cluster_column("nwicu", raw).values
r2, _ = fit_or(raw, "died ~ fi10 + age + male", "fi10", cluster="_pid")
RES["nwicu_definitions"]["no_lower_bound"] = dict(n=int(len(raw)), deaths=int(raw["died"].sum()),
                                                  OR=r2["OR"], lo=r2["lo"], hi=r2["hi"])
# Death-definition variants (labels precomputed in db_audit/nwicu_final_labels.csv)
lab = pd.read_csv(os.path.join(ANA, "db_audit", "nwicu_final_labels.csv"))
lab["stay_id"] = lab["stay_id"].astype(str)
base = pd.read_csv(os.path.join(ANA, "analysis_nwicu.csv"))
base["stay_id"] = base["stay_id"].astype(str)
base["fi10"] = base["fi_lab"] * 10
VARIANTS = {"primary": "died_clean", "no_lower_bound": "died_nolower",
            "no_lower_bound_1d": "died_nolower_1d", "lower_bound_1d": "died_clean_1d"}
for key, col in VARIANTS.items():
    # keep died_clean as its own column in every branch; _lab carries the variant outcome
    # died_clean is kept as its own column in every branch; _lab is a copy of the variant outcome
    lab_v = lab[["stay_id", "died_clean"]].assign(_lab=lab[col].values)
    d = base.merge(lab_v, on="stay_id", how="left")
    # primary and 1-day lower-bound variants exclude the 3 pre-admission records (died_clean is NaN)
    if key in ("primary", "lower_bound_1d"):
        d = d.dropna(subset=["died_clean"])
    d = d.dropna(subset=["_lab", "fi10", "age", "male"]).copy()
    d["_lab"] = d["_lab"]
    d["died"] = d["_lab"].astype(int)
    d["_pid"] = cluster_column("nwicu", d).values
    r, _ = fit_or(d, "died ~ fi10 + age + male", "fi10", cluster="_pid")
    RES["nwicu_definitions"][key] = dict(n=int(len(d)), deaths=int(d["died"].sum()),
                                         OR=r["OR"], lo=r["lo"], hi=r["hi"], label_col=col)
    print("[nwicu variant]", key, RES["nwicu_definitions"][key], flush=True)
print("[nwicu defs]", RES["nwicu_definitions"], flush=True)

save_json(RES, os.path.join(STAGE, "stage7_common_sample.json"))
print("WROTE", os.path.join(STAGE, "stage7_common_sample.json"))
