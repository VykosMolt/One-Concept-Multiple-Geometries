import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
models = ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]; fams = ["C_harmonic", "D_chord", "E_modulation"]; exs = ["A_win40", "B_any", "D_doc"]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, (nm, title) in enumerate((("wikipedia_v3_neutral", "neutralized view (12 target classes)"), ("wikipedia_v3", "spelled view (15 targets)"))):
    J = json.load(open(f"results/phase5/fingerprint/{nm}.json"))
    for col, (key, lab) in enumerate((("dkl", "held-out ΔKL = KL(theory) − KL(theory+corpus), nats/row"), ("r2gain", "held-out within-row ΔR² (theory+corpus vs theory)"))):
        ax = axes[row, col]; x = 0
        for m in models:
            for fam in fams:
                for ex in exs:
                    v = J[f"{m}|{fam}|{ex}"]; val = v[key]; p = v[key + "_p"]
                    ax.bar(x, val, color={"C_harmonic": "C0", "D_chord": "C2", "E_modulation": "C3"}[fam], alpha={"A_win40": 1.0, "B_any": 0.6, "D_doc": 0.35}[ex], edgecolor="k" if p < .05 else "none", linewidth=1.2); x += 1
                x += 0.5
            x += 1.5
        ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{title}: {lab}", fontsize=9); ax.set_xticks([4.75 + 12.5 * i for i in range(4)]); ax.set_xticklabels(models)
        if row == 0 and col == 0: ax.text(0.01, 0.97, "colour: C harmonic / D chord / E modulation; opacity: A_win40 / B_any / D_doc; black edge: relabeling p < .05", transform=ax.transAxes, fontsize=7, va="top")
fig.tight_layout(); fig.savefig("figures/phase5/heldout_gain_wikipedia.png", dpi=120); print("ok")
