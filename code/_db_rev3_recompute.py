# -*- coding: utf-8 -*-
"""Round-3 review recompute (from CSVs; no re-query):
 - Full-cohort MIMIC M1-M4 WITHOUT ARD (died~fi10+age+male; +CCI; +SOFA+OASIS; +vent/vaso/rrt)
 - External validation: derive died~fi10+age+male (FI-Lab+age+sex), apply NWICU/eICU (AUC/O:E/CITL/slope/Brier)
 - Clustering: unique patients, multi-stay counts (3 DBs); cluster-robust SE for M1; first-stay-per-patient M1
 - eICU APACHE increment: age+male+APACHE vs +FI-Lab (dAUC)
"""
import os, json, warnings, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore"); np.random.seed(42)
ROOT=ROOT; ANA=os.path.join(ROOT,"analysis"); OUT=os.path.join(ANA,"db_audit")
os.makedirs(OUT,exist_ok=True); RES={}
ana={db:pd.read_csv(f"{ANA}/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
rev={db:pd.read_csv(f"{ANA}/rev/rev_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
for d in ana.values(): d["fi10"]=d.fi_lab*10
for d in rev.values(): d["fi10"]=d.fi_lab*10

def orci(m,term="fi10"):
    b,se=m.params[term],m.bse[term]; return f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"
def auc_ci(y,p,B=1000):
    y=np.asarray(y);p=np.asarray(p);n=len(y);bs=[]
    for _ in range(B):
        i=np.random.randint(0,n,n)
        try: bs.append(roc_auc_score(y[i],p[i]))
        except: pass
    return roc_auc_score(y,p),np.percentile(bs,2.5),np.percentile(bs,97.5)

# ---- (1) MIMIC full-cohort M1-M4 WITHOUT ARD ----
m=rev["mimiciv"].dropna(subset=["sofa","oasis","sapsii","cci_excl_rheum"]).copy()
models=[("M1 age+sex","died ~ fi10 + age + male"),
        ("M2 +Charlson","died ~ fi10 + age + male + cci_excl_rheum"),
        ("M3 +SOFA+OASIS","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
        ("M4 +vent/vaso/rrt","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt")]
M=[]
for lab,f in models:
    r=smf.logit(f,data=m).fit(disp=0); auc,lo,hi=auc_ci(m.died,r.predict(m))
    M.append(dict(model=lab,OR=orci(r),P=("<0.001" if r.pvalues["fi10"]<0.001 else f"{r.pvalues['fi10']:.3f}"),AUC=f"{auc:.3f} ({lo:.3f}-{hi:.3f})"))
RES["MIMIC_M1_M4_noARD"]=M; print("M1-M4 (no ARD):"); [print("  ",x) for x in M]

# ---- (2) External validation: FI-Lab + age + sex ----
md=smf.logit("died ~ fi10 + age + male",data=ana["mimiciv"]).fit(disp=0); cf=md.params
RES["derive_coef_agessex"]={k:round(v,4) for k,v in cf.items()}
def calib(df):
    lp=cf["Intercept"]+cf["age"]*df.age+cf["male"]*df.male+cf["fi10"]*df.fi10
    p=1/(1+np.exp(-lp)); y=df.died.astype(int).values
    auc,alo,ahi=auc_ci(y,p); brier=brier_score_loss(y,p); O=y.sum(); oe=y.mean()/p.mean()
    oe_lo=np.exp(np.log(oe)-1.96/np.sqrt(O)); oe_hi=np.exp(np.log(oe)+1.96/np.sqrt(O))
    cmo=sm.Logit(y,np.ones((len(y),1)),offset=lp.values).fit(disp=0); citl=cmo.params[0]; ci=cmo.conf_int()[0]
    Xc=sm.add_constant(lp.values); cm=sm.Logit(y,Xc).fit(disp=0); s_ci=cm.conf_int()[1]
    return dict(n=len(y),deaths=int(O),AUC=f"{auc:.3f} ({alo:.3f}-{ahi:.3f})",Brier=round(brier,3),
                OE=f"{oe:.3f} ({oe_lo:.3f}-{oe_hi:.3f})",CITL=f"{citl:.3f} ({ci[0]:.3f}-{ci[1]:.3f})",
                slope=f"{cm.params[1]:.3f} ({s_ci[0]:.3f}-{s_ci[1]:.3f})")
E={}
for db,nm in [("mimiciv","MIMIC-IV apparent"),("nwicu","NWICU"),("eicu","eICU")]:
    E[nm]=calib(ana[db].copy())
RES["external_agessex"]=E; print("\nExternal (FI-Lab+age+sex):"); [print("  ",k,v) for k,v in E.items()]

# ---- (3) Clustering ----
C={}
for db in ["mimiciv","nwicu","eicu"]:
    d=ana[db]; nst=len(d); npat=d["subject_id"].nunique(); multi=nst-npat
    C[db]=dict(stays=nst,unique_patients=int(npat),extra_multi_stay=int(multi),pct_multi=round(100*multi/nst,2))
RES["clustering"]=C; print("\nClustering:",C)
# cluster-robust SE for M1 (MIMIC) using subject_id
r1=smf.logit("died ~ fi10 + age + male",data=ana["mimiciv"]).fit(disp=0,cov_type="cluster",cov_kwds={"groups":ana["mimiciv"]["subject_id"]})
b,se=r1.params["fi10"],r1.bse["fi10"]
RES["M1_clusterrobust_mimic"]=f"OR={np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})"
print("M1 cluster-robust (MIMIC):",RES["M1_clusterrobust_mimic"])
# first-stay-per-patient (MIMIC)
d=ana["mimiciv"].sort_values("stay_id").drop_duplicates("subject_id",keep="first")
r=smf.logit("died ~ fi10 + age + male",data=d).fit(disp=0)
RES["M1_firststay_perpatient_mimic"]=dict(n=len(d),OR=orci(r))
print("M1 first-stay-per-patient (MIMIC):",RES["M1_firststay_perpatient_mimic"])

# ---- (4) eICU APACHE increment ----
e=rev["eicu"].dropna(subset=["apache","fi10"]).copy()
print(f"\neICU APACHE n={len(e)}")
if len(e)>100 and e["apache"].std()>0:
    b=smf.logit("died ~ age + male + apache",data=e).fit(disp=0)
    f=smf.logit("died ~ age + male + apache + fi10",data=e).fit(disp=0)
    aucb=roc_auc_score(e.died,b.predict(e)); aucf=roc_auc_score(e.died,f.predict(e))
    RES["eicu_apache_increment"]=dict(n=len(e),AUC_apache=f"{aucb:.3f}",AUC_apache_fi=f"{aucf:.3f}",
        dAUC=f"{aucf-aucb:+.4f}",fi_OR_added=orci(f))
    print("eICU APACHE increment:",RES["eicu_apache_increment"])
else:
    RES["eicu_apache_increment"]="apache unavailable/insufficient"

json.dump(RES,open(os.path.join(OUT,"rev3_recompute.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
print("\nSAVED rev3_recompute.json")
