# -*- coding: utf-8 -*-
"""Shared statistics for BMC Geriatrics revision analyses.

All resampling is patient-level (cluster = subject_id / uniquepid).
Every stage imports this module so that metric definitions are identical.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ZY = os.path.dirname(REV)
ANA = os.path.join(ZY, "analysis")
OUT = os.path.join(REV, "analysis")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print("WROTE", path, flush=True)


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -35, 35)))


def logit_of(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _fit_slope(y, lp):
    """Two-parameter calibration regression y ~ alpha + beta*lp (Newton/IRLS)."""
    X = np.column_stack([np.ones(len(lp)), np.asarray(lp, float)])
    b = np.array([0.0, 1.0])
    for _ in range(60):
        p = expit(X @ b)
        w = np.clip(p * (1 - p), 1e-12, None)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-10 * np.eye(2), X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(b[0]), float(b[1])


def _fit_citl(y, lp):
    """One-parameter fixed-slope calibration: y ~ alpha + offset(lp)."""
    a = 0.0
    for _ in range(60):
        p = expit(np.asarray(lp, float) + a)
        step = float(np.sum(y - p) / max(np.sum(p * (1 - p)), 1e-12))
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def cluster_indices(g):
    g = np.asarray(g)
    uniq, inv = np.unique(g, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uniq))]
    return uniq, idx


def resample_indices(idx, B, rng):
    """B replicates of patient-level bootstrap index arrays."""
    n_groups = len(idx)
    for _ in range(B):
        pick = rng.integers(0, n_groups, n_groups)
        yield np.concatenate([idx[k] for k in pick])


def qci(vals, lo=2.5, hi=97.5):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return None, None
    return round(float(np.percentile(vals, lo)), 4), round(float(np.percentile(vals, hi)), 4)


def performance(y, p, g=None, B=1000, seed=20261126, bootstrap_calibration=True):
    """Discrimination + calibration with patient-level bootstrap intervals.

    AUC/CITL/slope intervals: percentile bootstrap by patient.
    O:E interval: Poisson approximation (unchanged from the original paper).
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    lp = logit_of(p)
    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    alpha, slope = _fit_slope(y, lp)
    citl = _fit_citl(y, lp)
    obs, exp = float(y.sum()), float(p.sum())
    oe = obs / exp if exp > 0 else np.nan
    oe_lo = float(np.exp(np.log(oe) - 1.96 / np.sqrt(obs))) if obs > 0 else None
    oe_hi = float(np.exp(np.log(oe) + 1.96 / np.sqrt(obs))) if obs > 0 else None
    out = dict(
        n=int(len(y)), deaths=int(obs),
        auc=round(auc, 4), brier=round(brier, 4),
        oe=round(float(oe), 4),
        oe_lo=round(oe_lo, 4) if oe_lo is not None else None,
        oe_hi=round(oe_hi, 4) if oe_hi is not None else None,
        citl=round(citl, 4), slope=round(slope, 4),
        interval_method="patient-level percentile bootstrap (AUC, CITL, slope); Poisson (O:E)",
    )
    if g is not None and B:
        rng = np.random.default_rng(seed)
        _, idx = cluster_indices(g)
        ba, bs, bc = [], [], []
        for ii in resample_indices(idx, B, rng):
            yy, pp = y[ii], p[ii]
            if yy.min() == yy.max():
                continue
            ba.append(roc_auc_score(yy, pp))
            a_b, s_b = _fit_slope(yy, logit_of(pp))
            if np.isfinite(s_b):
                bs.append(s_b)
            bc.append(_fit_citl(yy, logit_of(pp)))
        out["auc_lo"], out["auc_hi"] = qci(ba)
        if bootstrap_calibration:
            out["slope_lo"], out["slope_hi"] = qci(bs)
            out["citl_lo"], out["citl_hi"] = qci(bc)
            out["bootstrap_replicates"] = int(len(ba))
    return out


def paired_delta_auc(y, p_base, p_full, g=None, B=1000, seed=20261126):
    y = np.asarray(y, int)
    p_base = np.asarray(p_base, float)
    p_full = np.asarray(p_full, float)
    point = roc_auc_score(y, p_full) - roc_auc_score(y, p_base)
    out = dict(delta_auc=round(float(point), 4))
    if g is not None and B:
        rng = np.random.default_rng(seed)
        _, idx = cluster_indices(g)
        vals = []
        for ii in resample_indices(idx, B, rng):
            yy = y[ii]
            if yy.min() == yy.max():
                continue
            vals.append(roc_auc_score(yy, p_full[ii]) - roc_auc_score(yy, p_base[ii]))
        out["delta_auc_lo"], out["delta_auc_hi"] = qci(vals)
    return out


def paired_delta_brier(y, p_base, p_full, g=None, B=1000, seed=20261126):
    y = np.asarray(y, int)
    p_base = np.asarray(p_base, float)
    p_full = np.asarray(p_full, float)
    point = brier_score_loss(y, p_full) - brier_score_loss(y, p_base)
    out = dict(delta_brier=round(float(point), 4))
    if g is not None and B:
        rng = np.random.default_rng(seed)
        _, idx = cluster_indices(g)
        vals = []
        for ii in resample_indices(idx, B, rng):
            yy = y[ii]
            if yy.min() == yy.max():
                continue
            vals.append(brier_score_loss(yy, p_full[ii]) - brier_score_loss(yy, p_base[ii]))
        out["delta_brier_lo"], out["delta_brier_hi"] = qci(vals)
    return out


def fit_or(data, formula, term="fi10", cluster=None):
    import re
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    needed = [c for c in tokens if c in data.columns]
    if cluster is not None and cluster in data.columns:
        needed.append(cluster)
    d = data.dropna(subset=needed).copy()
    kw = {}
    if cluster is not None and cluster in d.columns:
        codes = pd.factorize(d[cluster].values)[0]
        kw = dict(cov_type="cluster", cov_kwds={"groups": codes})
    m = smf.logit(formula, data=d).fit(disp=0, **kw)
    b, se = float(m.params[term]), float(m.bse[term])
    p = float(m.pvalues[term])
    res = dict(
        n=int(m.nobs), term=term,
        OR=round(float(np.exp(b)), 3),
        lo=round(float(np.exp(b - 1.96 * se)), 3),
        hi=round(float(np.exp(b + 1.96 * se)), 3),
        p=(0.0 if p < 1e-300 else round(p, 8)),
        auc_apparent=round(float(roc_auc_score(d["died"], m.predict(d))), 4),
        se_type=("cluster-robust" if kw else "model-based"),
    )
    return res, m


def frozen_predict(coef, df, fi_col="fi_lab", age_col="age", male_col="male", fi10=False):
    fi = np.asarray(df[fi_col], float) * (1 if fi10 else 10)
    lp = (float(coef.get("Intercept", 0.0))
          + float(coef.get("fi10", 0.0)) * fi
          + float(coef.get("age", 0.0)) * np.asarray(df[age_col], float)
          + float(coef.get("male", 0.0)) * np.asarray(df[male_col], float))
    return expit(lp)


def cluster_column(db, df):
    """Return a patient-level grouping Series (never NaN)."""
    if "subject_id" in df.columns and df["subject_id"].notna().any():
        return df["subject_id"].astype(str)
    return pd.Series(np.arange(len(df)), index=df.index)


def apply_nwicu_label_exclusion(d, db):
    """NWICU: drop the 3 hospitalizations whose death date precedes admission
    (died_clean is NaN). Raw `died` is 1 for these rows, so they must not be
    kept with the raw label. Same rule as stage3 and stage4."""
    if db != "nwicu":
        return d
    lab = pd.read_csv(os.path.join(ANA, "db_audit", "nwicu_final_labels.csv"),
                      usecols=["stay_id", "died_clean"])
    lab["stay_id"] = lab["stay_id"].astype(str)
    d = d.merge(lab, on="stay_id", how="left")
    d = d.dropna(subset=["died_clean"]).copy()
    d["died"] = d["died_clean"].astype(int)
    return d.drop(columns=["died_clean"])


def read_analysis(db):
    d = pd.read_csv(os.path.join(ANA, f"analysis_{db}.csv"))
    d["stay_id"] = d["stay_id"].astype(str)
    d["fi10"] = d["fi_lab"] * 10
    return apply_nwicu_label_exclusion(d, db)


def read_fixed14(db):
    d = pd.read_csv(os.path.join(ANA, "db_audit", f"fixed_fi14_{db}.csv"))
    d["stay_id"] = d["stay_id"].astype(str)
    d["fi10"] = d["fi14"] * 10
    return apply_nwicu_label_exclusion(d, db)
