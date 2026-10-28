# -*- coding: utf-8 -*-
"""Compute ARD-excluded supplementary numbers (S4/S5/S6/S7) -> JSON for builder."""
import os, json, warnings, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT=ROOT; ANA=f"{ROOT}/analysis"

ana={db:pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
rev={db:pd.read_csv(f"{ANA}/rev/rev_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
red={db:pd.read_csv(f"{ROOT}/filab/filab_reduced_{db}.csv") for db in ["mimiciv","nwicu","eicu"]}
out={db:pd.read_csv(f"{ANA}/outcomes_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu"]}
for d in ana.values(): d["fi10"]=d["fi_lab"]*10
for d in rev.values(): d["fi10"]=d["fi_lab"]*10

def orc(m,term="fi10"):
    b,se,p=m.params[term],m.bse[term],m.pvalues[term]
    return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}–{np.exp(b+1.96*se):.3f})", ("<0.001" if p<0.001 else f"{p:.3f}")
def auc_ci(y,p,B=1000):
    y=np.asarray(y);p=np.asarray(p);n=len(y);bs=[]
    for _ in range(B):
        idx=np.random.randint(0,n,n)
        try: bs.append(roc_auc_score(y[idx],p[idx]))
        except: pass
    return roc_auc_score(y,p),np.percentile(bs,2.5),np.percentile(bs,97.5)

J={}

# ---- S4: non-ARD MIMIC derivation coefs (died~fi10+age+male) ----
mm=smf.logit("died ~ fi10 + age + male",data=ana["mimiciv"][ana["mimiciv"].ard==0]).fit(disp=0)
rows=[["Intercept",f"{mm.params['Intercept']:.4f}",f"{mm.bse['Intercept']:.4f}","—","<0.001"]]
labels={"fi10":"fi10 (FI-Lab, per 0.1 increase)","age":"age (per year)","male":"male"}
for t in ["fi10","age","male"]:
    b,se,p=mm.params[t],mm.bse[t],mm.pvalues[t]
    rows.append([labels[t],f"{b:.4f}",f"{se:.4f}",f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}–{np.exp(b+1.96*se):.3f})",("<0.001" if p<0.001 else f"{p:.3f}")])
J["S4"]=rows

# ---- S5 (old S6): threshold + early-death, MIMIC non-ARD ----
raw=ana["mimiciv"][ana["mimiciv"].ard==0].copy()
fl=pd.read_csv(f"{ROOT}/filab/filab_mimiciv.csv").set_index("stay_id")["icu_los_days"]
raw["icu_los_days"]=raw["stay_id"].map(fl)
s6=[]
for thr in [10,15,20]:
    d=raw[raw["n_assessed"]>=thr]
    m=smf.logit("died ~ fi10 + age + male",data=d).fit(disp=0)
    orc_s,p_=orc(m); s6.append([f"n_assessed>={thr}",len(d),orc_s,p_])
d=raw[~((raw["died"]==1)&(raw["icu_los_days"]<1))]
m=smf.logit("died ~ fi10 + age + male",data=d).fit(disp=0)
orc_s,p_=orc(m); s6.append(["exclude ICU death <24h",len(d),orc_s,p_])
J["S5"]=s6

# ---- S6 (old S7): 30-day outcomes, non-ARD ----
s7=[]
def inrow(db,name):
    d=ana[db][ana[db].ard==0].copy()
    m=smf.logit("died ~ fi10 + age + male",data=d).fit(disp=0)
    o,p_=orc(m); auc=roc_auc_score(d.died,m.predict(d))
    s7.append([name,"In-hospital mortality",f"{len(d):,}",f"{int(d.died.sum()):,} ({d.died.mean()*100:.1f})",o,p_,f"{auc:.3f}"])
def outrow(db,name,col,label):
    d=out[db][out[db].ard==0].copy(); d["fi10"]=d.fi_lab*10
    dd=d.dropna(subset=[col])
    if col=="readmit30": dd=dd[dd.died==0]
    m=smf.logit(f"{col} ~ fi10 + age + male",data=dd).fit(disp=0)
    o,p_=orc(m); auc=roc_auc_score(dd[col],m.predict(dd))
    s7.append([name,label,f"{len(dd):,}",f"{int(dd[col].sum()):,} ({dd[col].mean()*100:.1f})",o,p_,f"{auc:.3f}"])
inrow("mimiciv","MIMIC-IV"); outrow("mimiciv","MIMIC-IV","mort30","30-day mortality"); outrow("mimiciv","MIMIC-IV","readmit30","30-day readmission")
inrow("nwicu","NWICU"); outrow("nwicu","NWICU","mort30","30-day mortality*"); outrow("nwicu","NWICU","readmit30","30-day readmission")
inrow("eicu","eICU")
J["S6"]=s7

# ---- S7A (old S8 A): reduced 26-var model, MIMIC non-ARD ----
r=rev["mimiciv"]; r=r[r.ard==0].copy()
rmap=pd.read_csv(f"{ROOT}/filab/filab_reduced_mimiciv.csv").set_index("stay_id")["fi_lab_reduced"]
r["fi_red"]=r["stay_id"].map(rmap)
rcc=r.dropna(subset=["sofa","oasis","sapsii","cci_excl_rheum","fi_red"]).copy()
rcc["fi10r"]=rcc["fi_red"]*10
# original (30-var) on same cc
o_models=[("M1: age+sex","died ~ fi10 + age + male"),
          ("M2: +Charlson(-rheum)","died ~ fi10 + age + male + cci_excl_rheum"),
          ("M3: +SOFA+OASIS","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
          ("M4: +vent+vaso+rrt","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt")]
r_models=[(lab,f.replace("fi10","fi10r")) for lab,f in o_models]
s8a=[]
for (lab,fo),(_,fr) in zip(o_models,r_models):
    mo=smf.logit(fo,data=rcc).fit(disp=0); mr=smf.logit(fr,data=rcc).fit(disp=0)
    oo,_=orc(mo); ro,_=orc(mr,"fi10r")
    aucr=roc_auc_score(rcc.died,mr.predict(rcc))
    s8a.append([lab,oo,ro,("<0.001" if mr.pvalues["fi10r"]<0.001 else f"{mr.pvalues['fi10r']:.3f}"),f"{aucr:.3f}"])
J["S7A"]=s8a

# ---- S7B (old S8 B): reduced external validation, non-ARD ----
def load_red_nonard(db):
    b=ana[db].merge(red[db][["stay_id","fi_lab_reduced","n_assessed_reduced"]],on="stay_id",how="left")
    b=b[b.ard==0].copy(); b["fi_lab"]=b["fi_lab_reduced"]; b["fi10"]=b["fi_lab"]*10
    return b.dropna(subset=["fi_lab"]).copy()
rmim=load_red_nonard("mimiciv")
mdr=smf.logit("died ~ age + male + fi10",data=rmim).fit(disp=0); cf=mdr.params
def calibb(df):
    lp=cf["Intercept"]+cf["age"]*df.age+cf["male"]*df.male+cf["fi10"]*df.fi10
    p=1/(1+np.exp(-lp)); y=df.died.astype(int).values
    auc,alo,ahi=auc_ci(y,p.values); brier=brier_score_loss(y,p)
    O=y.sum(); oe=y.mean()/p.mean(); oe_lo=np.exp(np.log(oe)-1.96/np.sqrt(O)); oe_hi=np.exp(np.log(oe)+1.96/np.sqrt(O))
    Xo=np.ones((len(y),1)); cmo=sm.Logit(y,Xo,offset=lp.values).fit(disp=0); citl=cmo.params[0]
    cl,ch=citl-1.96*cmo.bse[0],citl+1.96*cmo.bse[0]
    Xc=sm.add_constant(lp.values); cm=sm.Logit(y,Xc).fit(disp=0); sl=cm.params[1]; sl_lo,sl_hi=sl-1.96*cm.bse[1],sl+1.96*cm.bse[1]
    return [f"{auc:.3f} ({alo:.3f}–{ahi:.3f})",f"{brier:.3f}",f"{oe:.3f} ({oe_lo:.3f}–{oe_hi:.3f})",f"{citl:.3f} ({cl:.3f}–{ch:.3f})",f"{sl:.3f} ({sl_lo:.3f}–{sl_hi:.3f})"]
s8b=[]
for db,nm in [("mimiciv","MIMIC-IV (apparent)"),("nwicu","NWICU"),("eicu","eICU")]:
    df=load_red_nonard(db); s8b.append([nm]+calibb(df))
J["S7B"]=s8b

json.dump(J,open(f"{ROOT}/manuscript/supp_nonard_numbers.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("=== S4 ==="); [print(r) for r in J["S4"]]
print("\n=== S5 threshold/leakage ==="); [print(r) for r in J["S5"]]
print("\n=== S6 30-day ==="); [print(r) for r in J["S6"]]
print("\n=== S7A reduced model ==="); [print(r) for r in J["S7A"]]
print("\n=== S7B reduced external ==="); [print(r) for r in J["S7B"]]
print("\nJSON saved.")
