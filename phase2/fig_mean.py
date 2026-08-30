import json, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags=["olmo2_1b","gemma2_2b","qwen25_3b","olmo2_7b"]; fams=["A_spelling","B_enharmonic","C_harmonic","D_chord","E_modulation","F_generic"]
fig,axes=plt.subplots(2,3,figsize=(15,7))
for r,(pos,src) in enumerate((("last","{t}_symbol.json"),("mean","{t}_symbol_v2_mean.json"))):
    for c,(stat,key,better) in enumerate((("ECI best","eci","low"),("circle|line best","circle_given_line","high"),("line|circle best","line_given_circle","high"))):
        M=np.full((len(fams),len(tags)),np.nan); P=np.full_like(M,np.nan)
        for j,t in enumerate(tags):
            f=f"results/phase2/geometry/"+src.format(t=t)
            if not os.path.exists(f): continue
            J=json.load(open(f))
            for i,fam in enumerate(fams):
                s=J[f"{fam}__{pos}"]["summary"][key]; M[i,j]=s["best"]; P[i,j]=s["p_min_free"] if key=="eci" else s["p_max_free"]
        ax=axes[r][c]; im=ax.imshow(M,cmap="viridis_r" if better=="low" else "viridis",aspect="auto",vmin=(0 if better=="low" else -0.2),vmax=(0.6 if better=="low" else 0.6))
        ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags,fontsize=8); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams,fontsize=8); ax.set_title(f"{stat} [key token: {pos}]",fontsize=9)
        for i in range(len(fams)):
            for j in range(len(tags)): ax.text(j,i,f"{M[i,j]:.2f}\np={P[i,j]:.2f}",ha="center",va="center",fontsize=6,color="w")
        plt.colorbar(im,ax=ax,fraction=0.04)
fig.suptitle("Key-name geometry: last token (top) vs span mean (bottom). The 'line' at the last token disappears under span-mean.",fontsize=9); fig.tight_layout(); fig.savefig("figures/phase2/keytoken_last_vs_mean.png",dpi=120); print("ok")
