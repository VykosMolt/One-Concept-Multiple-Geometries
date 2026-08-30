import numpy as np, sys
sys.path.insert(0,'.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from phase2.keys15 import KEYS15
Z=np.load("results/phase2/corpus/conditional15.npz"); classes=["all","other","chord","signature","modulation"]
fig,axes=plt.subplots(1,len(classes),figsize=(3.6*len(classes),3.6))
for ax,c in zip(axes,classes):
    M=Z[c]; Sm=M+M.T; E=np.outer(Sm.sum(1),Sm.sum(0))/max(Sm.sum(),1); A=np.log((Sm+0.5)/(E+0.5)); np.fill_diagonal(A,np.nan)
    im=ax.imshow(A,cmap="viridis"); ax.set_title(f"corpus log-association, cue={c} (n={int(M.sum())})",fontsize=8)
    ax.set_xticks(range(15)); ax.set_xticklabels(KEYS15,fontsize=6); ax.set_yticks(range(15)); ax.set_yticklabels(KEYS15,fontsize=6)
    for a,b in ((0,12),(1,13),(2,14)): ax.add_patch(plt.Rectangle((b-0.5,a-0.5),1,1,fill=False,edgecolor="r",lw=1)); ax.add_patch(plt.Rectangle((a-0.5,b-0.5),1,1,fill=False,edgecolor="r",lw=1))
fig.suptitle("15 spellings in line order; red boxes = enharmonic pairs (bright = periodic identity, dark = open line)",fontsize=9); fig.tight_layout(); fig.savefig("figures/phase2/corpus_conditional15.png",dpi=120); print("ok")
