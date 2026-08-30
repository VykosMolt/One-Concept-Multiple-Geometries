import sys, json, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pf.families import PC_CANON_MAJOR
tags = sys.argv[1].split(",")
rep = json.load(open("results/corpus/wiki/report.json"))["all"]["major_canon@pmi"]; Cp = np.array(rep["M"])
x = np.arange(12); fp = (7 * x) % 12; order = np.argsort(fp)  # fifths order C G D A E B F# Db Ab Eb Bb F
labels = [PC_CANON_MAJOR[i] for i in order]
mats = [("corpus PMI (Wikipedia)", Cp)]
for t in tags:
    P = json.load(open(f"results/predictive/{t}.json")); L = np.array(P["modulates_to"]["L"]); mats.append((f"{t}: log P(next key | 'modulates from x major to')", (L + L.T) / 2))
fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4.2))
for ax, (name, M) in zip(np.atleast_1d(axes), mats):
    Mo = M[np.ix_(order, order)].copy(); np.fill_diagonal(Mo, np.nan)
    im = ax.imshow(Mo, cmap="viridis"); ax.set_xticks(range(12)); ax.set_xticklabels(labels, fontsize=8); ax.set_yticks(range(12)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(name, fontsize=9); ax.axvline(6.5, color="w", lw=0.8, ls="--"); ax.axhline(6.5, color="w", lw=0.8, ls="--"); plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Rows/cols in circle-of-fifths order (C G D A E B F# | Db Ab Eb Bb F). Circle ⇒ bright corners (B~Db); line ⇒ dark corners.", fontsize=9)
fig.tight_layout(); fig.savefig("figures/summary/fig5_circle_vs_line_matrices.png", dpi=130); print("ok")
