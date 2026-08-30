import json, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags=["olmo2_1b","gemma2_2b","qwen25_3b","olmo2_7b"]; fams=["B_enharmonic","C_harmonic","D_chord","E_modulation"]
fig,axes=plt.subplots(1,3,figsize=(14,3.6))
for c,(stat,title,cmap,lo,hi) in enumerate((("eci","ECI (twins close = low)","viridis_r",0,1),("cl","circle|line","viridis",-0.2,0.6),("lc","line|circle","viridis",-0.2,0.6))):
    M=np.full((len(fams),len(tags)),np.nan)
    for j,t in enumerate(tags):
        J=json.load(open(f"results/phase2/behavior/{t}_fit.json"))
        for i,f in enumerate(fams):
            rows=[v for k,v in J.items() if k.startswith(f) and k.endswith("|total")]; M[i,j]=np.mean([r[stat] for r in rows])
    ax=axes[c]; im=ax.imshow(M,cmap=cmap,vmin=lo,vmax=hi,aspect="auto"); ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags,fontsize=8); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams,fontsize=8); ax.set_title(f"behaviour (total log-prob): {title}",fontsize=9)
    for i in range(len(fams)):
        for j in range(len(tags)): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8,color="w")
    plt.colorbar(im,ax=ax,fraction=0.04)
fig.tight_layout(); fig.savefig("figures/phase2/behavior15.png",dpi=120); print("ok")
