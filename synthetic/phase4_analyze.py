"""Phase-IV analysis: tolerance from balanced runs; per-r/code final metrics; N_equiv(r) in exposures and steps; curves vs
steps and vs cumulative exposure; unique-state control; representation-vs-behaviour timing; aligned−permuted paired tests.
Usage: python -m synthetic.phase4_analyze"""
import json, glob, os, numpy as np
from scipy.stats import ttest_1samp, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = {}
for d in glob.glob("results/phase4/runs/*/*_s*"):
    if not os.path.exists(f"{d}/trajectory.json"): continue
    c = json.load(open(f"{d}/config.json")); T = json.load(open(f"{d}/trajectory.json"))["trajectory"]; runs[(d.split("/")[3], c["manip"], c["r"], c["code"], c["seed"])] = T
RS = sorted({k[2] for k in runs if k[1] == "alias" and k[0] == "primary"}, reverse=True)
# tolerance from balanced (r=0.5) primary runs: 3 x 95th percentile of final latent JS
bal = [runs[k][-1]["twin_jsz"] for k in runs if k[0] == "primary" and k[2] == 0.5]
TOL = 3 * float(np.quantile(bal, 0.95)) if bal else np.nan
print(f"tolerance for equivalence (3 x q95 of final latent twin JS in balanced runs, n={len(bal)}): {TOL:.5f}")
def n_equiv(T, key="twin_jsz"):
    """first checkpoint after which the metric stays below TOL; returns (exposures, steps) or (nan, nan)"""
    for i, t in enumerate(T):
        if all(u[key] < TOL for u in T[i:]): return t["exposure_rare"], t["step"]
    return float("nan"), float("nan")
print(f"\n=== PRIMARY (alias rarity), final checkpoint (12k steps), mean over seeds ===")
print(f"{'r':>6s} {'code':9s} | exposures | KL global common rare unique | twin JSz  JS15 | Q twin dist_rel cos eci | N_equiv exposures (steps) per seed")
summ = {}
for r in RS:
    for code in ("aligned", "permuted"):
        ks = sorted([k for k in runs if k[0] == "primary" and k[2] == r and k[3] == code], key=lambda k: k[4])
        if not ks: continue
        F = [runs[k][-1] for k in ks]; ne = [n_equiv(runs[k]) for k in ks]
        summ[(r, code)] = {"kl_global": np.mean([f["kl_global"] for f in F]), "kl_common": np.mean([f["kl_common"] for f in F]), "kl_rare": np.mean([f["kl_rare"] for f in F]), "twin_jsz": [f["twin_jsz"] for f in F], "twin_js15": np.mean([f["twin_js15"] for f in F]),
                          "dist_rel": np.mean([f["hidden"][-1]["twin_dist_rel"] for f in F]), "cos": np.mean([f["hidden"][-1]["twin_cos"] for f in F]), "eci": np.mean([f["hidden"][-1]["eci"] for f in F]), "n_equiv": ne, "exposure": np.mean([f["exposure_rare"] for f in F])}
        s = summ[(r, code)]
        print(f"{r:6g} {code:9s} | {s['exposure']:9.0f} | {s['kl_global']:.4f} {s['kl_common']:.4f} {s['kl_rare']:.4f} {np.mean([f['kl_unique_ctrl'] for f in F]):.4f} | {np.mean(s['twin_jsz']):.4f} {s['twin_js15']:.4f} | {s['dist_rel']:.2f} {s['cos']:.2f} {s['eci']:.2f} | " + " ".join(f"{e:.0f}({st:.0f})" if not np.isnan(e) else "never" for e, st in ne))
print("\n=== paired aligned − permuted (final latent JS; N_equiv exposures) ===")
for r in RS:
    a, p = summ.get((r, "aligned")), summ.get((r, "permuted"))
    if not a or not p: continue
    d = np.array(a["twin_jsz"]) - np.array(p["twin_jsz"]); ne_a = np.array([x[0] for x in a["n_equiv"]]); ne_p = np.array([x[0] for x in p["n_equiv"]]); dn = ne_a - ne_p; ok = ~np.isnan(dn)
    print(f"r={r:g}: Δ final JSz {d.mean():+.4f} ({int((d>0).sum())}/{len(d)} positive, t p={ttest_1samp(d,0).pvalue if d.std()>0 else float('nan'):.3f}); Δ N_equiv exposures {np.nanmean(dn):+.0f} ({int((dn[ok]>0).sum())}/{int(ok.sum())} positive; aligned never: {int(np.isnan(ne_a).sum())}, permuted never: {int(np.isnan(ne_p).sum())})")
print("\n=== EXTENDED runs (48k steps) ===")
for r in (0.003, 0.001):
    for code in ("aligned", "permuted"):
        ks = [k for k in runs if k[0] == "extended" and k[2] == r and k[3] == code]
        if not ks: continue
        F = [runs[k][-1] for k in ks]; ne = [n_equiv(runs[k]) for k in ks]
        print(f"r={r:g} {code:9s} n={len(ks)}: exposures {np.mean([f['exposure_rare'] for f in F]):.0f} | KL global {np.mean([f['kl_global'] for f in F]):.4f} rare {np.mean([f['kl_rare'] for f in F]):.4f} | JSz {np.mean([f['twin_jsz'] for f in F]):.4f} | Q dist_rel {np.mean([f['hidden'][-1]['twin_dist_rel'] for f in F]):.2f} | N_equiv " + " ".join(f"{e:.0f}({st:.0f})" if not np.isnan(e) else "never" for e, st in ne))
print("\n=== UNIQUE-STATE CONTROL: rare unique state's row KL vs rare alias's row KL at matched exposure (final, 12k) ===")
for r in (0.03, 0.01, 0.001):
    for code in ("aligned", "permuted"):
        ku = [k for k in runs if k[0] == "control" and k[2] == r and k[3] == code]; ka = [k for k in runs if k[0] == "primary" and k[2] == r and k[3] == code]
        if not ku or not ka: continue
        Fu = [runs[k][-1] for k in ku]; Fa = [runs[k][-1] for k in ka]
        print(f"r={r:g} {code:9s}: unique-control row KL {np.mean([f['kl_unique_ctrl'] for f in Fu]):.4f} (exposures {np.mean([f['exposure_unique_ctrl'] for f in Fu]):.0f}) vs rare-alias row KL {np.mean([f['kl_rare'] for f in Fa]):.4f} (exposures {np.mean([f['exposure_rare'] for f in Fa]):.0f}); others KL {np.mean([f['kl_others'] for f in Fu]):.4f}")
# figures: JSz vs steps and vs exposure, per r (mean over seeds), aligned solid / permuted dashed
fig, axes = plt.subplots(2, 3, figsize=(16, 8)); cols = plt.cm.viridis(np.linspace(0, 1, len(RS)))
for c, (xkey, xl) in enumerate((("step", "training step"), ("exposure_rare", "cumulative rare-alias source exposures"))):
    for row, (ykey, yl, ylog) in enumerate((("twin_jsz", "latent-class twin JS", True), ("kl_rare", "rare-alias row KL", True))):
        ax = axes[row][c]
        for ri, r in enumerate(RS):
            for code, ls in (("aligned", "-"), ("permuted", "--")):
                ks = [k for k in runs if k[0] in ("primary", "extended") and k[1] == "alias" and k[2] == r and k[3] == code]
                if not ks: continue
                for k in ks:
                    T = runs[k]; ax.plot([t[xkey] + (1 if xkey == "exposure_rare" else 0) for t in T], [max(t[ykey], 1e-6) for t in T], ls, color=cols[ri], alpha=0.35, label=f"r={r:g} {code}" if k[4] == 0 and k[0] == "primary" else None)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.axhline(TOL, color="k", lw=0.6, ls=":")
        if row == 0 and c == 0: ax.legend(fontsize=5, ncol=2)
ax = axes[0][2]
for ri, r in enumerate(RS):
    for code, ls in (("aligned", "-"), ("permuted", "--")):
        ks = [k for k in runs if k[0] == "primary" and k[2] == r and k[3] == code]
        for k in ks:
            T = runs[k]; ax.plot([t["exposure_rare"] + 1 for t in T], [t["hidden"][-1]["twin_dist_rel"] for t in T], ls, color=cols[ri], alpha=0.35)
ax.set_xscale("log"); ax.set_xlabel("cumulative rare-alias exposures"); ax.set_ylabel("Q twin distance / mean non-twin distance (last layer)"); ax.set_title("representation collapse", fontsize=9)
ax = axes[1][2]
for ri, r in enumerate(RS):
    for code, ls in (("aligned", "-"), ("permuted", "--")):
        ks = [k for k in runs if k[0] == "primary" and k[2] == r and k[3] == code]
        for k in ks:
            T = runs[k]; ax.plot([t["step"] for t in T], [max(t["kl_global"], 1e-6) for t in T], ls, color=cols[ri], alpha=0.35)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("global KL"); ax.set_title("global convergence", fontsize=9)
fig.suptitle("Phase IV: sparse source aliases. Colour = rarity r (yellow = rarest); solid = aligned code, dashed = permuted; dotted = equivalence tolerance", fontsize=9)
fig.tight_layout(); os.makedirs("figures/phase4", exist_ok=True); fig.savefig("figures/phase4/curves.png", dpi=120)
json.dump({"tol": TOL, "summary": {f"{r}|{code}": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in s.items()} for (r, code), s in summ.items()}}, open("results/phase4/summary.json", "w"), indent=1, default=float)
print("figure written")
