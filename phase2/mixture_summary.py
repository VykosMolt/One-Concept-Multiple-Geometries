"""Track 9 view: for every model × family × position, layer trajectories of (i) ECI, (ii) circle|line, (iii) line|circle,
(iv) CV R² of orthography-only and the increments for circle / line; plus a heatmap of best corrected values.
Usage: python phase2/mixture_summary.py tag1,tag2,... [spelling]"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from phase2.contexts import FAMILIES
tags = sys.argv[1].split(","); spelling = sys.argv[2] if len(sys.argv) > 2 else "symbol"; fams = list(FAMILIES)
avail = [t for t in tags if os.path.exists(f"results/phase2/geometry/{t}_{spelling}.json")]
J = {t: json.load(open(f"results/phase2/geometry/{t}_{spelling}.json")) for t in avail}
# heatmaps: rows = family, cols = model; values = best ECI (lower = more enharmonic collapse), best circle|line, best line|circle, at key token
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for r, pos in enumerate(("last", "final")):
    for c, (stat, key, better) in enumerate((("ECI (best over layers)", "eci", "low"), ("circle|line (best)", "circle_given_line", "high"), ("line|circle (best)", "line_given_circle", "high"))):
        M = np.full((len(fams), len(avail)), np.nan); P = np.full_like(M, np.nan)
        for j, t in enumerate(avail):
            for i, fam in enumerate(fams):
                s = J[t][f"{fam}__{pos}"]["summary"][key]; M[i, j] = s["best"]; P[i, j] = s["p_min_free"] if key == "eci" else s["p_max_free"]
        ax = axes[r][c]; im = ax.imshow(M, cmap="viridis_r" if better == "low" else "viridis", aspect="auto", vmin=(0 if better == "low" else -0.2), vmax=(0.6 if better == "low" else 0.6))
        ax.set_xticks(range(len(avail))); ax.set_xticklabels(avail, fontsize=8); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8); ax.set_title(f"{stat} [{pos}]", fontsize=9)
        for i in range(len(fams)):
            for j in range(len(avail)):
                ax.text(j, i, f"{M[i,j]:.2f}\np={P[i,j]:.2f}", ha="center", va="center", fontsize=6, color="w" if (M[i,j] < 0.3 if better == "low" else M[i,j] > 0.3) else "k")
        plt.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("15-key geometry: p = max-over-layers relabeling p (free)", fontsize=9); fig.tight_layout(); fig.savefig(f"figures/phase2/heatmap15_{spelling}.png", dpi=120)
# trajectories: CV R² increments per family (key token)
fig, axes = plt.subplots(1, len(avail), figsize=(4.5 * len(avail), 3.4), squeeze=False)
for j, t in enumerate(avail):
    ax = axes[0][j]
    for fam, col in zip(fams, ["C0", "C1", "C2", "C3", "C4", "C7"]):
        pl = J[t][f"{fam}__last"]["per_layer"]; L = [i for i, x in enumerate(pl) if x]
        ax.plot(L, [pl[i]["cv_circle"] - pl[i]["cv_ortho"] for i in L], "-", color=col, label=f"{fam} Δcircle")
        ax.plot(L, [pl[i]["cv_line"] - pl[i]["cv_ortho"] for i in L], "--", color=col, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{t}: ΔCV R² over orthography (solid circle, dashed line), key token", fontsize=8); ax.set_xlabel("layer")
    if j == 0: ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"figures/phase2/cv_increments_{spelling}.png", dpi=120)
# text table of the same
print(f"{'model':10s} {'family':13s} | key token: ECI best(p) | circle|line best(p) | line|circle best(p) | max ΔCV circle / line over layers")
for t in avail:
    for fam in fams:
        s = J[t][f"{fam}__last"]["summary"]; pl = [x for x in J[t][f"{fam}__last"]["per_layer"] if x]
        dc = max(x["cv_circle"] - x["cv_ortho"] for x in pl); dl = max(x["cv_line"] - x["cv_ortho"] for x in pl)
        print(f"{t:10s} {fam:13s} | {s['eci']['best']:.2f} ({s['eci']['p_min_free']:.3f}) | {s['circle_given_line']['best']:+.2f} ({s['circle_given_line']['p_max_free']:.3f}) | {s['line_given_circle']['best']:+.2f} ({s['line_given_circle']['p_max_free']:.3f}) | {dc:+.3f} / {dl:+.3f}")
print("figures written")
