"""Layerwise Fourier spectra for extracted hidden states. Produces results/spectra/<tag>/*.json + figures."""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import mode_energies, paired_vector, parseval_ok, conjugate_symmetry_ok, permutation_null
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tag = sys.argv[1]
indir = f"results/hidden/{tag}"; outdir = f"results/spectra/{tag}"; figdir = f"figures/{tag}"
os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
NPERM = 2000


def analyze_H(H):
    """H: (L+1, n, d) -> per-layer normalized paired profile, raw energies, null quantiles."""
    Lp1, n, d = H.shape
    prof, raw, null95, null50, zsc = [], [], [], [], []
    for l in range(Lp1):
        Hl = H[l]
        if np.isnan(Hl).any():
            prof.append([np.nan] * (n // 2)); raw.append([np.nan] * (n // 2)); null95.append([np.nan] * (n // 2)); null50.append([np.nan]*(n//2)); zsc.append([np.nan]*(n//2)); continue
        assert parseval_ok(Hl)[0] and conjugate_symmetry_ok(Hl)
        E = mode_energies(Hl); v = paired_vector(E); p = v / v.sum()
        nv = permutation_null(Hl, n=NPERM); npf = nv / nv.sum(1, keepdims=True)
        prof.append(p.tolist()); raw.append(v.tolist())
        null95.append(np.quantile(npf, 0.95, axis=0).tolist()); null50.append(np.median(npf, axis=0).tolist())
        zsc.append(((p - npf.mean(0)) / npf.std(0)).tolist())
    return {"profile": prof, "raw": raw, "null95": null95, "null50": null50, "z": zsc}


summary = {}
for f in sorted(glob.glob(f"{indir}/*.npz")):
    name = os.path.basename(f)[:-4]
    z = np.load(f, allow_pickle=True)
    n = None
    if name.endswith("__embed"):
        H = z["H"][None]  # (1, n, d)
        res = {"embed": analyze_H(H)}
        n = H.shape[1]
    else:
        res = {}
        for pos in ("last", "mean", "anchor", "final"):
            if pos in z: res[pos] = analyze_H(z[pos]); n = z[pos].shape[1]
        res["template"] = str(z["template"])
    summary[name] = res
    json.dump(res, open(f"{outdir}/{name}.json", "w"))
    # figure: layerwise profile for each position
    labels = [f"P{m}" for m in range(1, n // 2)] + [f"E{n//2}"]
    poss = [p for p in ("last", "mean", "anchor", "final", "embed") if p in res]
    fig, axes = plt.subplots(1, len(poss), figsize=(4.2 * len(poss), 3.4), squeeze=False)
    for ax, pos in zip(axes[0], poss):
        P = np.array(res[pos]["profile"])
        for m in range(P.shape[1]):
            ax.plot(P[:, m], marker="o", ms=3, label=labels[m])
        ax.axhline(2 / (n - 1), color="k", ls=":", lw=0.8)
        ax.set_title(f"{name} [{pos}]", fontsize=8); ax.set_xlabel("layer"); ax.set_ylabel("share of centered energy")
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{figdir}/{name}.png", dpi=110); plt.close(fig)
    print(name, "done", flush=True)
json.dump(summary, open(f"{outdir}/_summary.json", "w"))
