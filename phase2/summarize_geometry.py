"""Aggregate geometry fits across models/families: table + figure of ECI, circle|line, line|circle (best layer with
corrected p, and final layer) per family and position. Usage: python phase2/summarize_geometry.py tag1,tag2,... [spelling]"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from phase2.contexts import FAMILIES
tags = sys.argv[1].split(","); spelling = sys.argv[2] if len(sys.argv) > 2 else "symbol"
fams = list(FAMILIES)
rows = []
print(f"{'model':10s} {'family':13s} pos   | ECI best (p_free/p_glyph) final | circle|line best (p) final | line|circle best (p) final | ΔCV: circle line both (at best circle|line layer)")
for t in tags:
    f = f"results/phase2/geometry/{t}_{spelling}.json"
    if not os.path.exists(f): continue
    J = json.load(open(f))
    for fam in fams:
        for pos in ("last", "final"):
            r = J[f"{fam}__{pos}"]; s = r["summary"]; bl = s["circle_given_line"]["best_layer"]; st = r["per_layer"][bl] or {}
            dcv = (st.get("cv_circle", np.nan) - st.get("cv_ortho", np.nan), st.get("cv_line", np.nan) - st.get("cv_ortho", np.nan), st.get("cv_both", np.nan) - st.get("cv_ortho", np.nan))
            print(f"{t:10s} {fam:13s} {pos:5s} | {s['eci']['best']:.2f} @L{s['eci']['best_layer']:<2d} ({s['eci']['p_min_free']:.3f}/{s['eci']['p_min_glyph']:.3f}) {s['eci']['final']:.2f} | "
                  f"{s['circle_given_line']['best']:+.2f} @L{bl:<2d} ({s['circle_given_line']['p_max_free']:.3f}) {s['circle_given_line']['final']:+.2f} | "
                  f"{s['line_given_circle']['best']:+.2f} @L{s['line_given_circle']['best_layer']:<2d} ({s['line_given_circle']['p_max_free']:.3f}) {s['line_given_circle']['final']:+.2f} | {dcv[0]:+.2f} {dcv[1]:+.2f} {dcv[2]:+.2f}")
            rows.append((t, fam, pos, s))
# figure: per model, per family, layer curves of ECI, circle|line, line|circle at the key token
fig, axes = plt.subplots(len(tags), 2, figsize=(12, 3.2 * len(tags)), squeeze=False)
for a, t in enumerate(tags):
    f = f"results/phase2/geometry/{t}_{spelling}.json"
    if not os.path.exists(f): continue
    J = json.load(open(f))
    for b, pos in enumerate(("last", "final")):
        ax = axes[a][b]
        for fam, col in zip(fams, ["C0", "C1", "C2", "C3", "C4", "C7"]):
            pl = J[f"{fam}__{pos}"]["per_layer"]; L = [i for i, x in enumerate(pl) if x]
            ax.plot(L, [pl[i]["eci"] for i in L], "-", color=col, label=f"{fam} ECI")
            ax.plot(L, [pl[i]["circle_given_line"] for i in L], ":", color=col, alpha=0.8)
        ax.axhline(0.5, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3); ax.set_ylim(-0.6, 1.0); ax.set_title(f"{t} [{pos}] — solid: ECI (0.5 = null, low = enharmonic pairs close); dotted: circle|line", fontsize=8)
        if a == 0 and b == 0: ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); os.makedirs("figures/phase2", exist_ok=True); fig.savefig(f"figures/phase2/geometry15_{spelling}.png", dpi=120); print("figure written")
