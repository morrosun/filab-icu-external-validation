# -*- coding: utf-8 -*-
"""
Revision figures (English labels, Times New Roman for SCI submission).
Restructured into 4 main multi-panel figures + 1 supplementary figure:
  fig1_flow.png/.pdf            -> Figure 1  STROBE flow (3 databases side-by-side)
  fig2_distribution_or.png/.pdf -> Figure 2  Panel A FI-Lab distribution; Panel B OR-decay forest
  fig3_validation_calibration.png/.pdf -> Figure 3 Panel A external AUC forest; B/C/D calibration per DB
  fig4_ard_dca.png/.pdf         -> Figure 4  Panel A ARD vs non-ARD AUC; Panel B DCA
  figS1_missingness.png/.pdf    -> Supplementary Figure S1  per-lab missingness heatmap

NOTE: figure numbers ("Figure X") and figure titles are NOT drawn on the images
anymore -- the Word/PDF caption is the single source of truth, so numbers on the
image can never disagree with the text. Only panel letters (A/B/C/D) remain.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of code/)
for _d in ("cohort", "filab", "analysis", "analysis/revision",
             "analysis/db_audit", "figures", "figures/revision"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)



plt.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"stix",
    "font.size":11,"axes.titlesize":12,"axes.labelsize":11,
    "axes.linewidth":1.1,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9.5,
    "figure.dpi":150,"savefig.dpi":300,"axes.titlesize":12.5,
    "axes.unicode_minus":False,
})
ANA=os.path.join(ROOT, 'analysis')
REV=f"{ANA}/revision"
FIG=os.path.join(ROOT, 'figures', 'revision')
os.makedirs(FIG, exist_ok=True)
OUT=[]

DER=dict(intercept=-6.754470955511114, age=0.032683277156335025,
         male=-0.03610865483247025, ard=-0.006681934923211605, fi10=0.4954576741870491)
def lp(df, coef):
    fi10=df["fi_lab"]*10
    return (coef["intercept"]+coef["age"]*df["age"]+coef["male"]*df["male"]
            +coef["ard"]*df["ard"]+coef["fi10"]*fi10)
def pred(df, coef): return 1/(1+np.exp(-lp(df,coef)))

def calib(df, coef, nb=10):
    p=pred(df,coef).values; y=df["died"].values.astype(float)
    order=np.argsort(p); ps=p[order]; ys=y[order]
    edges=np.linspace(0,1,nb+1); xs=[]; obs=[]
    for i in range(nb):
        m=(ps>=edges[i])&(ps<edges[i+1]) if i<nb-1 else (ps>=edges[i])&(ps<=edges[i+1])
        if m.sum()>0:
            xs.append(ps[m].mean()); obs.append(ys[m].mean())
    xs=np.array(xs); obs=np.array(obs)
    d=df.copy(); d["lp"]=lp(d,coef); d["y"]=d["died"]
    mm=smf.logit("y ~ lp", data=d).fit(disp=0)
    return xs, obs, mm.params["lp"], mm.params["Intercept"], p

def net_benefit(y,p,thr):
    y=np.asarray(y);p=np.asarray(p);n=len(y)
    out=[]
    for t in thr:
        idx=p>=t
        tp=((idx)&(y==1)).sum(); fp=((idx)&(y==0)).sum()
        nb=tp/n-fp/n*(t/(1-t))
        prev=y.mean(); nb_all=prev-(1-prev)*(t/(1-t))
        out.append((t,nb,nb_all))
    return pd.DataFrame(out,columns=["thr","nb","nb_all"])

# ---- load augmented cohorts (for severity-adjusted MIMIC model) ----
mim=pd.read_csv(f"{ANA}/rev/rev_mimiciv.csv").dropna(subset=["fi_lab","sofa","oasis","cci_excl_rheum"]).copy()
eic=pd.read_csv(f"{ANA}/rev/rev_eicu.csv").dropna(subset=["fi_lab","apache"]).copy()
nwi=pd.read_csv(f"{ANA}/rev/rev_nwicu.csv").dropna(subset=["fi_lab","cci_excl_rheum"]).copy()
for d in (mim,eic,nwi): d["fi10"]=d["fi_lab"]*10

# ---- load full FI-evaluable cohorts (for frozen-coefficient figures) ----
mim_f=pd.read_csv(f"{ANA}/analysis_mimiciv.csv").dropna(subset=["fi_lab"]).copy()
eic_f=pd.read_csv(f"{ANA}/analysis_eicu.csv").dropna(subset=["fi_lab"]).copy()
nwi_f=pd.read_csv(f"{ANA}/analysis_nwicu.csv").dropna(subset=["fi_lab"]).copy()
for d in (mim_f,eic_f,nwi_f): d["fi10"]=d["fi_lab"]*10

mfull=smf.logit("died ~ fi10+age+male+ard+cci_excl_rheum+sofa+oasis", data=mim).fit(disp=0)

# calibration per db (frozen-coefficient model)
cal={}
for name,df in [("MIMIC-IV",mim_f),("NWICU",nwi_f),("eICU",eic_f)]:
    xs,obs,slope,inter,pp=calib(df,DER,nb=10)
    cal[name]=dict(xs=xs,obs=obs,slope=slope,inter=inter,
                   auc=roc_auc_score(df["died"],pp),brier=brier_score_loss(df["died"],pp))

# DCA on MIMIC apparent
thr=np.round(np.arange(0.02,0.76,0.02),2)
dca=net_benefit(mim["died"].values, mfull.predict(mim), thr)
dca.to_csv(f"{REV}/dca_plot_mimiciv.csv",index=False)
_rows=[]
for name in cal:
    for x,o in zip(cal[name]['xs'],cal[name]['obs']):
        _rows.append(dict(db=name,x=round(float(x),4),y=round(float(o),4)))
pd.DataFrame(_rows).to_csv(f"{REV}/calibration_points.csv",index=False)

COL={"MIMIC-IV":"#4C72B0","NWICU":"#DD8452","eICU":"#55A868"}

def savefig(fig, base, pdf=True):
    fig.savefig(f"{FIG}/{base}.png",bbox_inches="tight",pad_inches=0.15)
    if pdf:
        fig.savefig(f"{FIG}/{base}.pdf",bbox_inches="tight",pad_inches=0.15)
    plt.close(fig)
    OUT.append(f"{FIG}/{base}.png")

def panel_letter(ax, letter):
    ax.text(0.02,0.98,letter,transform=ax.transAxes,fontsize=14,fontweight="bold",
            va="top",ha="left")

# ============================================================ Figure 1: STROBE flow (3 databases side-by-side)
db_info = [
    ("MIMIC-IV", 46566, 45656, 910),
    ("NWICU",    13921, 10819, 5771),
    ("eICU",     84713, 56544, 28169),
]
fig,ax=plt.subplots(figsize=(10.5,7.0))
ax.axis("off"); ax.set_xlim(0,11); ax.set_ylim(-0.4,8.2)
# three columns, left edges and centres
cols=[dict(xL=0.5,xR=3.5,cx=2.0),dict(xL=4.0,xR=7.0,cx=5.5),dict(xL=7.5,xR=10.5,cx=9.0)]
def box1(x,y,w,h,text,fc,fs=9.0):
    ax.add_patch(plt.Rectangle((x,y),w,h,fill=True,fc=fc,ec="black",lw=1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,wrap=True)
for (name,total,ev,ex),c in zip(db_info,cols):
    evp=ev/total*100; exp=ex/total*100
    xL,xR,cx=c["xL"],c["xR"],c["cx"]
    W=xR-xL
    # top: total population
    box1(xL,6.4,W,0.95,f"{name}\nelderly (>=65) first ICU\n{total:,}", "#DCE6F1", fs=9.5)
    # stub arrow (in the gap, never over a box)
    ax.annotate("",xy=(cx,5.95),xytext=(cx,6.4),arrowprops=dict(arrowstyle="->",color="black",lw=1))
    # lower boxes: evaluable (left) and excluded (right)
    bw=W/2-0.15
    box1(xL,4.45,bw,0.95,f"FI-Lab\nevaluable\n{ev:,}\n({evp:.1f}%)", "#E2EFDA", fs=8.3)
    box1(xR-bw,4.45,bw,0.95,f"FI-Lab not\nevaluable\n{ex:,}\n({exp:.1f}%)", "#F2DCDB", fs=8.3)
    # fork arrows from stub end to each lower box (kept in the gaps)
    ax.annotate("",xy=(xL+bw/2,5.4),xytext=(cx-0.45,5.95),
                arrowprops=dict(arrowstyle="->",color="black",lw=1))
    ax.annotate("",xy=(xR-bw/2,5.4),xytext=(cx+0.45,5.95),
                arrowprops=dict(arrowstyle="->",color="black",lw=1))
savefig(fig,"fig1_flow")
OUT.append(f"{FIG}/fig1_flow.pdf")

# ============================================================ Figure 2: distribution (A) + OR-decay forest (B)
hir=pd.read_csv(f"{REV}/hierarchical_models_mimiciv.csv")
labels=["M1\nAge + Sex","M2\n+ARD","M3\n+Charlson\n(excl. rheum)","M4\n+SOFA+OASIS","M5\n+Vent/Vaso/RRT"]
fig=plt.figure(figsize=(12,4.7))
gs=GridSpec(1,2,figure=fig,width_ratios=[1.45,1],wspace=0.28)
gsA=gs[0].subgridspec(1,3,wspace=0.32)
for j,(name,df) in enumerate([("MIMIC-IV",mim_f),("NWICU",nwi_f),("eICU",eic_f)]):
    ax=fig.add_subplot(gsA[0,j])
    ax.hist(df["fi_lab"].dropna(),bins=30,color=COL[name],edgecolor="white",linewidth=0.3)
    ax.set_title(name,fontsize=11); ax.set_xlabel("FI-Lab"); ax.set_ylabel("Number of patients" if j==0 else "")
    med=df["fi_lab"].median(); ax.axvline(med,color="#C44E52",ls="--",lw=1.2)
    ax.text(0.97,0.92,f"Median {med:.2f}",transform=ax.transAxes,ha="right",fontsize=9,color="#C44E52")
    panel_letter(ax,"A")
axB=fig.add_subplot(gs[1]); yy=np.arange(len(hir))
axB.errorbar(hir["OR"],yy,xerr=[hir["OR"]-hir["lo"],hir["hi"]-hir["OR"]],fmt="o",color="#4C72B0",capsize=3,ms=5)
for i,r in hir.iterrows():
    axB.text(r["hi"]+0.015,i,f"{r['OR']:.2f}",va="center",fontsize=8.5)
axB.set_yticks(yy); axB.set_yticklabels(labels,fontsize=9); axB.invert_yaxis()
axB.margins(y=0.14)
axB.axvline(1.0,color="grey",ls=":",lw=0.8)
xr=max(1.0, hir["hi"].max()*1.18)
axB.set_xlim(0.9,xr)
axB.set_xlabel("FI-Lab per 0.1 increase: OR (95% CI)")
axB.set_title("Attenuation of FI-Lab OR after severity adjustment (MIMIC-IV)",fontsize=11,pad=8)
panel_letter(axB,"B")
savefig(fig,"fig2_distribution_or")
OUT.append(f"{FIG}/fig2_distribution_or.pdf")

# ============================================================ Figure 3: external AUC forest (A) + calibration B/C/D
ext=pd.read_csv(f"{REV}/external_validation_frozen.csv")
elab=["MIMIC-IV\n(derivation)","NWICU\n(frozen external)","eICU\n(frozen external)"]
fig,axes=plt.subplots(2,2,figsize=(11,8.6))
axA=axes[0,0]
axA.scatter(ext["auc"],np.arange(len(ext)),s=45,color="#55A868",zorder=3)
for i,r in ext.iterrows(): axA.text(r["auc"]+0.004,i,f"{r['auc']:.3f}",va="center",fontsize=9)
axA.set_yticks(np.arange(len(ext))); axA.set_yticklabels(elab,fontsize=9); axA.invert_yaxis()
axA.axvline(0.5,color="grey",ls=":",lw=0.8); axA.set_xlim(0.45,0.85)
axA.set_xlabel("In-hospital mortality AUC (frozen-coefficient external validation)")
axA.set_title("A. Cross-database external validation AUC",fontsize=11)
panel_letter(axA,"A")
pos=[(0,1),(1,0),(1,1)]; names=["MIMIC-IV","NWICU","eICU"]
for (r,c),name in zip(pos,names):
    ax=axes[r,c]; c0=cal[name]
    ax.plot([0,1],[0,1],"k--",lw=1,label="Ideal calibration")
    ax.plot(c0["xs"],c0["obs"],"o-",color=COL[name],label=f"{name}\n(slope {c0['slope']:.2f}, CITL {c0['inter']:.2f})",ms=5)
    ax.set_xlabel("Predicted probability of death"); ax.set_ylabel("Observed proportion of death")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8,loc="upper left")
    tag={0:"B",1:"C",2:"D"}[names.index(name)]
    ax.set_title(f"{tag}. Calibration — {name}",fontsize=11)
    panel_letter(ax,tag)
savefig(fig,"fig3_validation_calibration")
OUT.append(f"{FIG}/fig3_validation_calibration.pdf")

# ============================================================ Figure 4: ARD vs non-ARD AUC (A) + DCA (B)
sub=pd.read_csv(f"{REV}/ard_subgroup_performance.csv")
piv=sub.pivot(index="cohort",columns="group",values="auc")
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
axA=axes[0]; yy=np.arange(len(piv))
axA.errorbar(piv["ARD"],yy,fmt="o",color="#C44E52",label="ARD",ms=7)
axA.errorbar(piv["non-ARD"],yy,fmt="s",color="#4C72B0",label="non-ARD",ms=7)
for i,idx in enumerate(piv.index):
    a=piv.loc[idx,"ARD"]; n=piv.loc[idx,"non-ARD"]
    if abs(a-n)<0.03:
        left,right=(a,n) if a<=n else (n,a); li=a<=n
        axA.text((a if li else n)-0.006,i,f"{(a if li else n):.3f}",va="center",ha="right",fontsize=8,color="#C44E52" if li else "#4C72B0")
        axA.text((n if li else a)+0.006,i,f"{(n if li else a):.3f}",va="center",ha="left",fontsize=8,color="#4C72B0" if li else "#C44E52")
    else:
        axA.text(a+0.004,i,f"{a:.3f}",va="center",fontsize=8,color="#C44E52")
        axA.text(n+0.004,i,f"{n:.3f}",va="center",fontsize=8,color="#4C72B0")
axA.set_yticks(yy); axA.set_yticklabels(piv.index,fontsize=9); axA.invert_yaxis()
axA.axvline(0.5,color="grey",ls=":",lw=0.8); axA.set_xlim(0.45,0.85); axA.set_xlabel("In-hospital mortality AUC")
axA.legend(fontsize=9,loc="lower right"); axA.set_title("A. ARD vs non-ARD subgroups",fontsize=11)
panel_letter(axA,"A")
axB=axes[1]
axB.plot(dca["thr"],dca["nb"],"-",color="#4C72B0",label="Model (FI-Lab + severity)")
axB.plot(dca["thr"],dca["nb_all"],"-",color="#C44E52",label="Treat all")
axB.plot(dca["thr"],[0]*len(dca),"k:",lw=0.8,label="Treat none")
axB.set_xlabel("Risk threshold"); axB.set_ylabel("Net benefit (per patient)")
axB.set_title("B. Decision curve (MIMIC-IV)",fontsize=11); axB.legend(fontsize=8.5,loc="upper right")
panel_letter(axB,"B")
savefig(fig,"fig4_ard_dca")
OUT.append(f"{FIG}/fig4_ard_dca.pdf")

# ============================================================ Supplementary Figure S1: missingness heatmap
miss=pd.read_csv(f"{REV}/per_lab_missingness.csv").set_index("item")
order=["bun","creatinine","sodium","potassium","chloride","glucose","calcium","bicarbonate",
       "platelet","hematocrit","hemoglobin","aniongap","rdw","mcv",
       "albumin","bilirubin_tot","alt","ast","alp","total_protein","magnesium","phosphate",
       "pt","ptt","inr","lactate","fibrinogen","ld_ldh","ck_cpk","bilirubin_dir"]
M=miss.loc[order,["MIMIC-IV_miss%","NWICU_miss%","eICU_miss%"]].values
fig,ax=plt.subplots(figsize=(5.6,9))
im=ax.imshow(M,cmap="YlOrRd",aspect="auto",vmin=0,vmax=100)
ax.set_xticks([0,1,2]); ax.set_xticklabels(["MIMIC-IV","NWICU","eICU"],fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order,fontsize=7.5)
ax.set_title("Per-laboratory missingness (%)",fontsize=11)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.ax.tick_params(labelsize=7)
savefig(fig,"figS1_missingness")
OUT.append(f"{FIG}/figS1_missingness.pdf")

print("FIGURES:", OUT)
