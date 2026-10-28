# -*- coding: utf-8 -*-
"""Parent-cohort >=10/15/20 FI-Lab sensitivity. The analysis CSV already holds all parent stays
(n_assessed for everyone; fi_lab NaN when <15). For stays with 10<=n_assessed<15 we recompute FI
(deficit/assessed) from their labs (light: a few hundred/thousand stays), then run thresholds."""
import os, json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
import build_analysis as ba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


ROOT=ROOT; ANA=os.path.join(ROOT,"analysis"); OUT=os.path.join(ANA,"db_audit")
os.makedirs(OUT,exist_ok=True); RES={}

def deficits(L):
    cols=[]
    for a in ba.ANALYTES:
        lo,hi=ba.D[a][3],ba.D[a][4]
        mn=L.get(f"{a}_min"); mx=L.get(f"{a}_max")
        if mn is None: continue
        assessed=mn.notna()|mx.notna()
        cols.append(((mn<lo)|(mx>hi)).fillna(False).where(assessed))
    M=pd.concat(cols,axis=1); an=M.notna().sum(axis=1); dn=M.sum(axis=1,min_count=1).fillna(0)
    return an,dn
def orci(df,form="died ~ fi10 + age + male + ard"):
    r=smf.logit(form,data=df).fit(disp=0); b,se=r.params["fi10"],r.bse["fi10"]
    return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})",len(df)

def run(db,tag,fi_fn):
    ana=pd.read_csv(f"{ANA}/analysis_{db}.csv"); ana["stay_id"]=ana["stay_id"].astype(str)
    ana["n_assessed"]=pd.to_numeric(ana["n_assessed"],errors="coerce")
    band=(ana.n_assessed>=10)&(ana.n_assessed<15)
    ids=tuple(int(x) for x in ana.loc[band,"stay_id"])
    print(f"[{tag}] total={len(ana)} band(10-14)={int(band.sum())} extracting...")
    fi_map={}
    if ids:
        L=fi_fn(ids); L["stay_id"]=L["stay_id"].astype(str)
        an2,dn2=deficits(L); fv=np.where(an2>0,dn2/an2,np.nan)
        fi_map=dict(zip(L["stay_id"],fv))
    ana.loc[band,"fi_lab"]=ana.loc[band,"stay_id"].map(fi_map)
    an=ana["n_assessed"]
    dist={"total":int(len(ana)),"n_ge10":int((an>=10).sum()),"n_ge15":int((an>=15).sum()),
          "n_ge20":int((an>=20).sum()),"n_10to14":int(band.sum()),"n_lt10":int((an<10).sum())}
    for T in [10,15,20]:
        d=ana[an>=T].dropna(subset=["fi_lab","died","age","male"]).copy(); d["fi10"]=d["fi_lab"]*10
        dist[f">={T}"]=f"n={len(d)} OR={orci(d)[0]}"
    print(f"[{tag}]",dist); return dist

RES["mimic"]=run("mimiciv","MIMIC-IV",ba.fi_mimic)
try: RES["nwicu"]=run("nwicu","NWICU",ba.fi_nwicu)
except Exception as e: RES["nwicu"]={"ERROR":str(e)}; print("NWICU err",e)
json.dump(RES,open(os.path.join(OUT,"parent_threshold.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
print("\n",json.dumps(RES,ensure_ascii=False,indent=1,default=str))
