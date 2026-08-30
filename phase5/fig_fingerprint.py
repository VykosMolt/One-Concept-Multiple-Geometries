import json, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
models=["olmo2_1b","gemma2_2b","qwen25_3b","olmo2_7b"]; fams=["C_harmonic","D_chord","E_modulation"]; ex=["A_win40","B_any","D_doc"]
fig,axes=plt.subplots(2,2,figsize=(13,8))
for r,(view,f) in enumerate((("spelled","results/phase5/fingerprint/wikipedia.json"),("neutral","results/phase5/fingerprint/wikipedia_neutral.json"))):
    J=json.load(open(f))
    for c,(stat,title,lo,hi) in enumerate((("resid_r","residual Spearman(C_resid, Q_resid)",-0.2,0.7),("dcv","ΔCV = LOO(theory+corpus) − LOO(theory)",-0.05,0.1))):
        M=np.full((len(fams)*len(ex),len(models)),np.nan); P=np.full_like(M,np.nan)
        for j,m in enumerate(models):
            for i,(fa,e) in enumerate([(fa,e) for fa in fams for e in ex]):
                k=f"{m}|{fa}|{e}"
                if k in J: M[i,j]=J[k][stat]; P[i,j]=J[k]["resid_p" if stat=="resid_r" else "dcv_p"]
        ax=axes[r][c]; im=ax.imshow(M,cmap="viridis",vmin=lo,vmax=hi,aspect="auto"); ax.set_xticks(range(len(models))); ax.set_xticklabels(models,fontsize=8); ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"{fa[:1]}×{e}" for fa in fams for e in ex],fontsize=7); ax.set_title(f"{view} view: {title}",fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j,i,f"{M[i,j]:+.2f}\np={P[i,j]:.2f}",ha="center",va="center",fontsize=6,color="w" if M[i,j]<(hi+lo)/2 else "k")
        plt.colorbar(im,ax=ax,fraction=0.04)
fig.suptitle("Phase V — Wikipedia conditional statistics vs model next-key behaviour (rows: context family × extraction family)",fontsize=9); fig.tight_layout(); os.makedirs("figures/phase5",exist_ok=True); fig.savefig("figures/phase5/fingerprint_wikipedia.png",dpi=120); print("ok")
