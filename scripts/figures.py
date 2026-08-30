"""Summary figures. Usage: python scripts/figures.py <model tags comma-separated>"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures/summary", exist_ok=True)
rep = json.load(open("results/corpus/wiki/report.json"))["all"]
tags = sys.argv[1].split(",")
# Fig 1: corpus kernels, months vs keys (PMI), in semitone and fifths order
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
km = np.array(rep["months@pmi"]["kappa"]); axes[0].plot(range(12), km, "o-"); axes[0].set_title("Months: PMI kernel κ(d) (Wikipedia)"); axes[0].set_xlabel("d (months)")
kk = np.array(rep["major_canon@pmi"]["kappa"]); kmin = np.array(rep["minor_canon@pmi"]["kappa"])
axes[1].plot(range(12), kk, "o-", label="major"); axes[1].plot(range(12), kmin, "s--", label="minor"); axes[1].set_title("Keys: PMI κ(d), semitone order"); axes[1].set_xlabel("d (semitones)"); axes[1].legend()
fo = [(7 * d) % 12 for d in range(12)]
axes[2].plot(range(12), kk[fo], "o-", label="major"); axes[2].plot(range(12), kmin[fo], "s--", label="minor"); axes[2].set_title("Keys: PMI κ(d'), fifths order"); axes[2].set_xlabel("d' (fifths)"); axes[2].legend()
fig.tight_layout(); fig.savefig("figures/summary/fig1_corpus_kernels.png", dpi=130); plt.close(fig)
# Fig 2: corpus M* matrices for keys (semitone order and fifths order) and months
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
M = np.array(rep["major_canon@pmi"]["M"]); labs = rep["major_canon@pmi"]["labels"]
im = axes[0].imshow(M, cmap="viridis"); axes[0].set_xticks(range(12)); axes[0].set_xticklabels(labs, fontsize=7); axes[0].set_yticks(range(12)); axes[0].set_yticklabels(labs, fontsize=7); axes[0].set_title("Keys PMI, semitone order"); plt.colorbar(im, ax=axes[0])
o = [(7 * i) % 12 for i in range(12)]
im = axes[1].imshow(M[np.ix_(o, o)], cmap="viridis"); axes[1].set_xticks(range(12)); axes[1].set_xticklabels([labs[i] for i in o], fontsize=7); axes[1].set_yticks(range(12)); axes[1].set_yticklabels([labs[i] for i in o], fontsize=7); axes[1].set_title("Keys PMI, fifths order"); plt.colorbar(im, ax=axes[1])
Mm = np.array(rep["months"]["M"]); im = axes[2].imshow(Mm, cmap="viridis"); axes[2].set_title("Months M*"); plt.colorbar(im, ax=axes[2])
fig.tight_layout(); fig.savefig("figures/summary/fig2_corpus_matrices.png", dpi=130); plt.close(fig)
# Fig 3+4 per model: layerwise P1/P5 raw & black-projected; RSA decomposition
for tag in tags:
    f = f"results/multictx/{tag}/analysis.json"
    if not os.path.exists(f): continue
    A = json.load(open(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, fam in zip(axes, ("major", "minor")):
        if fam not in A: continue
        rows = [r for r in A[fam]["last"] if r]
        L = [r["layer"] for r in rows]
        ax.plot(L, [r["profile"][4] for r in rows], "o-", color="C3", label="P5 raw")
        ax.plot(L, [r["profile"][0] for r in rows], "o-", color="C0", label="P1 raw")
        ax.plot(L, [r["profile_black"][4] for r in rows], "s--", color="C3", label="P5 black-projected")
        ax.plot(L, [r["profile_black"][0] for r in rows], "s--", color="C0", label="P1 black-projected")
        ax.axhline(2 / 11, color="k", ls=":", lw=0.8); ax.set_ylim(0, 0.5); ax.set_title(f"{tag} {fam} keys, last concept token (24-ctx avg)"); ax.set_xlabel("layer"); ax.set_ylabel("share of centered energy")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"figures/summary/fig3_{tag}_P1P5.png", dpi=130); plt.close(fig)
print("figures written")
