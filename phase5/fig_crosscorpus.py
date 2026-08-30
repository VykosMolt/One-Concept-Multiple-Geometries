"""Cross-corpus figure: held-out ΔKL per model for each corpus (solid) vs its size-matched Wikipedia baseline (hatched).
Usage: python phase5/fig_crosscorpus.py [suffix _neutral|''] [corpora...]"""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
suf = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else ""; corpora = sys.argv[2:] or ["olmomix_wiki", "olmomix_dclm", "dolmino_dclm"]
models = ["olmo2_1b", "olmo2_7b", "gemma2_2b", "qwen25_3b"]; cells = [("E_modulation", "A_win40"), ("E_modulation", "D_doc"), ("C_harmonic", "A_win40"), ("C_harmonic", "D_doc")]
fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.8), sharey=True)
for ax, (fam, ex) in zip(axes, cells):
    x = 0; ticks = []
    for c in corpora:
        Jc = json.load(open(f"results/phase5/fingerprint/{c}{suf}.json")); Jb = json.load(open(f"results/phase5/fingerprint/wikipedia_thin_{c}{suf}.json"))
        for i, m in enumerate(models):
            v = Jc.get(f"{m}|{fam}|{ex}"); w = Jb.get(f"{m}|{fam}|{ex}")
            if v: ax.bar(x, v["dkl"], width=0.4, color=f"C{i}", edgecolor="k" if v["dkl_p"] < .05 else "none")
            if w: ax.bar(x + 0.4, w["dkl"], width=0.4, color=f"C{i}", alpha=0.45, hatch="//", edgecolor="k" if w["dkl_p"] < .05 else "none")
            x += 1
        ticks.append((x - 2.3, c)); x += 1
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([c for _, c in ticks], fontsize=8); ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{fam} × {ex}", fontsize=9)
axes[0].set_ylabel(f"held-out ΔKL (nats/row), {'neutral' if suf else 'spelled'} view"); axes[0].text(0.01, 0.97, "solid: corpus; hatched: Wikipedia thinned to the same pair mass\ncolours: OLMo-1B, OLMo-7B, Gemma-2B, Qwen-3B; black edge p<.05", transform=axes[0].transAxes, fontsize=7, va="top")
fig.tight_layout(); fig.savefig(f"figures/phase5/crosscorpus{suf}.png", dpi=120); print("ok")
