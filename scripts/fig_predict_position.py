"""Per-layer partial fifths and circle/line at the predicting position vs at the concept token. Usage: tags comma."""
import sys, os, json, re, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags = sys.argv[1].split(",")
fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 3.8), squeeze=False)
for ax, t in zip(axes[0], tags):
    f = f"results/predict_position/{t}.json"
    if not os.path.exists(f): continue
    J = json.load(open(f))
    for c, ls in (("then_key", "-"), ("modulates_to", "--"), ("next_song", ":"), ("ctrl_nonpredicting", "-.")):
        if c not in J: continue
        rows = J[c]; L = [r["layer"] for r in rows]
        ax.plot(L, [r["partial_fifths"] for r in rows], ls, color="C3" if c != "ctrl_nonpredicting" else "C7", label=f"partial fifths [{c}]")
        if c == "ctrl_nonpredicting": continue
        ax.plot(L, [r["circle_given_line"] for r in rows], ls, color="C2", alpha=0.8, label=f"circle|line [{c}]" if c == "then_key" else None)
        ax.plot(L, [r["line_given_circle"] for r in rows], ls, color="C0", alpha=0.8, label=f"line|circle [{c}]" if c == "then_key" else None)
    # concept-token partial fifths (major, last) for comparison
    dec = f"results/multictx/{t}_decompose_major_last.txt"
    if os.path.exists(dec):
        v = [(int(m.group(1)), float(m.group(2))) for m in (re.match(r"\s+(\d+)\s+fifths=([+-][\d.]+)", l) for l in open(dec)) if m]
        if v: ax.plot([a for a, b in v], [b for a, b in v], "k.-", alpha=0.6, label="partial fifths, key-name token (24 ctx)")
    ax.axhline(0.62, color="C3", lw=0.6, ls="-.", label="corpus PMI partial fifths (0.62)"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{t}: predicting position vs key-name token", fontsize=9); ax.set_xlabel("layer"); ax.set_ylim(-0.6, 0.8)
    ax.legend(fontsize=6, loc="lower left")
fig.tight_layout(); fig.savefig("figures/summary/fig6_predicting_position.png", dpi=130); print("ok")
