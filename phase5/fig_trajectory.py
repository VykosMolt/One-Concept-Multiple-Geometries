import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = json.load(open("results/phase5/ckpt_fingerprint.json"))
order = ["stage1-step300-tokens1B", "stage1-step10000-tokens21B", "stage1-step23100-tokens49B", "stage1-step50000-tokens105B", "stage1-step140000-tokens294B", "stage1-step480000-tokens1007B", "stage1-step950000-tokens1993B", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B", "stage2-ingredient1-step23852-tokens51B", "stage2-ingredient2-step23852-tokens51B", "main"]
lab = ["1B", "21B", "49B", "105B", "294B", "1T", "2T", "4T", "S2 i3", "S2 i1", "S2 i2", "released"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for fam, col in (("E_modulation", "C3"), ("C_harmonic", "C0")):
    R = {r["rev"]: r for r in rows if r["fam"] == fam}
    axes[0].plot(range(12), [R[o]["lc"] for o in order], "o-", color=col, label=f"{fam} line|circle"); axes[0].plot(range(12), [R[o]["cl"] for o in order], "s--", color=col, alpha=0.6, label=f"{fam} circle|line")
    axes[1].plot(range(12), [R[o]["resid"]["D_doc"][0] for o in order], "o-", color=col, label=f"{fam} × D_doc"); axes[1].plot(range(12), [R[o]["resid"]["A_win40"][0] for o in order], "s--", color=col, alpha=0.6, label=f"{fam} × A_win40")
    axes[2].plot(range(12), [R[o]["asym"] for o in order], "o-", color=col, label=f"{fam} twin asymmetry (nats)")
for ax, t in zip(axes, ("behaviour line vs circle (controlled partials)", "neutral-view residual corpus–model correspondence (rich nuisance)", "enharmonic-twin target asymmetry")):
    ax.set_xticks(range(12)); ax.set_xticklabels(lab, fontsize=8); ax.set_xlabel("OLMo-2-0425-1B checkpoint (training tokens)"); ax.set_title(t, fontsize=9); ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/phase5/checkpoint_trajectory.png", dpi=120); print("ok")
