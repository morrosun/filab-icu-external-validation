# -*- coding: utf-8 -*-
"""ARD-excluded sensitivity: recompute ALL manuscript numbers on non-ARD cohort."""
import os, warnings, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)


warnings.filterwarnings("ignore")
np.random.seed(42)
ROOT=ROOT

ana={db:pd.read_csv(f"{ROOT}/analysis/analysis_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
rev={db:pd.read_csv(f"{ROOT}/analysis/rev/rev_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"]}
out={db:pd.read_csv(f"{ROOT}/analysis/outcomes_{db}.csv").dropna(subset=["fi_lab"]).copy() for db in ["mimiciv","nwicu","eicu"] if __import__("os").path.exists(f"{ROOT}/analysis/outcomes_{db}.csv")}
for d in ana.values(): d["fi10"]=d["fi_lab"]*10
for d in rev.values(): d["fi10"]=d["fi_lab"]*10

def fmt_med(s):
    s=pd.to_numeric(s,errors="coerce").dropna(); return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"

print("="*80); print("TABLE 1  Baseline (NON-ARD)"); print("="*80)
for db,name in [("mimiciv","MIMIC-IV"),("nwicu","NWICU"),("eicu","eICU")]:
    d=ana[db][ana[db].ard==0]
    print(f"{name}: N={len(d)}, age={fmt_med(d.age)}, male={int(d.male.sum())} ({d.male.mean()*100:.1f}%), "
          f"FI-Lab={fmt_med(d.fi_lab)}, assessed={fmt_med(d.n_assessed)}, deaths={int(d.died.sum())} ({d.died.mean()*100:.1f}%)")

def auc_ci(y,p,B=1000):
    y=np.asarray(y);p=np.asarray(p);n=len(y);bs=[]
    for _ in range(B):
        idx=np.random.randint(0,n,n)
        try: bs.append(roc_auc_score(y[idx],p[idx]))
        except: pass
    return roc_auc_score(y,p),np.percentile(bs,2.5),np.percentile(bs,97.5)
def oe_ci(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);O=y.sum();base=y.mean()/p.mean()
    se=1.96/np.sqrt(O);return base,np.exp(np.log(base)-se),np.exp(np.log(base)+se)

print("\n"+"="*80); print("TABLE 2  MIMIC non-ARD hierarchical models (per 0.1 FI-Lab)"); print("="*80)
m=rev["mimiciv"]; m=m[m.ard==0].copy()
mcc=m.dropna(subset=["sofa","oasis","sapsii","cci_excl_rheum"]).copy()
print(f"MIMIC non-ARD complete-case w/ severity: {len(mcc)} / {len(m)}")
models=[("M1: age+sex","died ~ fi10 + age + male"),
        ("M2: +Charlson(-rheum)","died ~ fi10 + age + male + cci_excl_rheum"),
        ("M3: +SOFA+OASIS","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis"),
        ("M4: +vent+vaso+rrt","died ~ fi10 + age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt")]
for name,f in models:
    mm=smf.logit(f,data=mcc).fit(disp=0)
    b,se,p=mm.params["fi10"],mm.bse["fi10"],mm.pvalues["fi10"]
    pred=mm.predict(mcc); auc,lo,hi=auc_ci(mcc.died,pred)
    pf="<0.001" if p<0.001 else f"{p:.3f}"
    print(f"  {name:26} OR={np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})  P={pf}  AUC={auc:.3f} ({lo:.3f}-{hi:.3f})")

print("\n"+"="*80); print("TABLE 3  Frozen external validation (derivation on MIMIC non-ARD: died~age+male+fi10)"); print("="*80)
md=smf.logit("died ~ age + male + fi10",data=ana["mimiciv"][ana["mimiciv"].ard==0]).fit(disp=0)
coef=md.params
print("Derivation coef:",{k:round(v,4) for k,v in coef.items()})
def calib(name,df):
    lp=coef["Intercept"]+coef["age"]*df.age+coef["male"]*df.male+coef["fi10"]*df.fi10
    p=1/(1+np.exp(-lp));y=df.died.astype(int).values
    auc,alo,ahi=auc_ci(y,p);brier=brier_score_loss(y,p);oe,olo,ohi=oe_ci(y,p)
    citl=smf.logit("died~1",data=df,offset=pd.Series(lp.values,index=df.index)).fit(disp=0)
    ci=citl.conf_int().loc["Intercept"]
    Xc=sm.add_constant(lp.values);cm=sm.Logit(y,Xc).fit(disp=0);sci=cm.conf_int()
    print(f"  {name:22} AUC={auc:.3f} ({alo:.3f}-{ahi:.3f})  Brier={brier:.3f}  O:E={oe:.3f} ({olo:.3f}-{ohi:.3f})  "
          f"CITL={citl.params['Intercept']:.3f} ({ci[0]:.3f}-{ci[1]:.3f})  slope={cm.params[1]:.3f} ({sci[1,0]:.3f}-{sci[1,1]:.3f})")
for db,name in [("mimiciv","MIMIC-IV (apparent)"),("nwicu","NWICU"),("eicu","eICU")]:
    calib(name, ana[db][ana[db].ard==0].copy())

print("\n"+"="*80); print("INCREMENTAL VALUE (MIMIC non-ARD): base severity vs +FI-Lab"); print("="*80)
base="died ~ age + male + cci_excl_rheum + sofa + oasis + vent + vaso + rrt"
mb=smf.logit(base,data=mcc).fit(disp=0); mf=smf.logit(base+" + fi10",data=mcc).fit(disp=0)
yb=mcc.died.values;pb=mb.predict(mcc).values;pf2=mf.predict(mcc).values
a1=roc_auc_score(yb,pb);a2=roc_auc_score(yb,pf2)
lr=2*(mf.llf-mb.llf);lrp=chi2.sf(lr,1)
# bootstrap delta AUC
dbs=[]
for _ in range(1000):
    idx=np.random.randint(0,len(yb),len(yb))
    try: dbs.append(roc_auc_score(yb[idx],pf2[idx])-roc_auc_score(yb[idx],pb[idx]))
    except: pass
b1=brier_score_loss(yb,pb);b2=brier_score_loss(yb,pf2)
bb=[]
for _ in range(1000):
    idx=np.random.randint(0,len(yb),len(yb))
    try: bb.append(brier_score_loss(yb[idx],pf2[idx])-brier_score_loss(yb[idx],pb[idx]))
    except: pass
print(f"  AUC base={a1:.3f} -> full={a2:.3f} (dAUC {a2-a1:+.3f}, 95%CI {np.percentile(dbs,2.5):+.4f} to {np.percentile(dbs,97.5):+.4f})")
print(f"  LR chi2={lr:.1f}, P={lrp:.2e}")
print(f"  Brier base={b1:.3f} -> full={b2:.3f} (diff {b2-b1:+.4f}, 95%CI {np.percentile(bb,2.5):+.4f} to {np.percentile(bb,97.5):+.4f})")

print("\n"+"="*80); print("30-DAY OUTCOMES (NON-ARD): mort30 & readmit30, OR per 0.1 + AUC"); print("="*80)
for db,name in [("mimiciv","MIMIC-IV"),("nwicu","NWICU"),("eicu","eICU")]:
    if db not in out: 
        print(f"  {name}: no outcomes file"); continue
    d=out[db]; d=d[d.ard==0].copy(); d["fi10"]=d.fi_lab*10
    for col in ["mort30","readmit30"]:
        if col not in d: continue
        dd=d.dropna(subset=[col])
        if col=="readmit30":
            dd=dd[dd.died==0]  # survivors only (competing risk)
        if dd[col].sum()<5 or len(dd)<30:
            print(f"  {name} {col}: too few (n={len(dd)}, ev={int(dd[col].sum())})"); continue
        mm=smf.logit(f"{col} ~ fi10 + age + male",data=dd).fit(disp=0)
        b,se,p=mm.params["fi10"],mm.bse["fi10"],mm.pvalues["fi10"]
        auc=roc_auc_score(dd[col],mm.predict(dd))
        pf="<0.001" if p<0.001 else f"{p:.3f}"
        print(f"  {name:8} {col:9}: n={len(dd)} ev={int(dd[col].sum())} ({dd[col].mean()*100:.1f}%) OR={np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f}) P={pf} AUC={auc:.3f}")

print("\nDONE")
