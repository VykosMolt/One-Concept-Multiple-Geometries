import json, numpy as np, sys
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags = ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, t in zip(axes, tags):
    J = json.load(open(f"results/decouple/{t}.json"))
    for setname, col in (("canonical", "C0"), ("decoupled", "C3")):
        rows = J[setname]["rows"]; L = [r["layer"] for r in rows]
        ax.plot(L, [r["rsa_black"] for r in rows], "-", color=col, label=f"{setname}: RSA with black-key block")
        ax.plot(L, [r["rsa_glyph"] for r in rows], "--", color=col, label=f"{setname}: RSA with accidental-glyph block")
    ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.4, 1); ax.set_title(t, fontsize=9); ax.set_xlabel("layer")
    if t == tags[0]: ax.legend(fontsize=6)
fig.suptitle("Respelling C,E,F,B as B#,Fb,E#,Cb: the block follows the accidental glyph, not the black keys (12 contexts, last token)", fontsize=9)
fig.tight_layout(); fig.savefig("figures/summary/fig7_orthographic_decoupling.png", dpi=130); print("ok")
