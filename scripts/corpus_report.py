"""Corpus-side report: M*, circulant projection, kernel DFT, predicted profiles for months, weekdays,
and musical keys under explicit spelling aggregations. Usage: python scripts/corpus_report.py <merged.json> <outtag>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corpus.analyze import load_tally, build_M, spectrum_from_M, bootstrap_M
from pf.families import MONTHS, WEEKDAYS, PC_CANON_MAJOR, PC_CANON_MINOR, PC_ALL_SHARP, PC_ALL_FLAT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

path, tag = sys.argv[1], sys.argv[2]
groups = sys.argv[3].split(",") if len(sys.argv) > 3 else ["all"]
outdir = f"results/corpus/{tag}"; figdir = f"figures/corpus_{tag}"
os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)

ENH = {0: ["C", "B#"], 1: ["C#", "Db"], 2: ["D"], 3: ["D#", "Eb"], 4: ["E", "Fb"], 5: ["F", "E#"], 6: ["F#", "Gb"],
       7: ["G"], 8: ["G#", "Ab"], 9: ["A"], 10: ["A#", "Bb"], 11: ["B", "Cb"]}

def key_labels(spellings, mode):
    return [[f"KEY:{s}:{mode}" for s in (sp if isinstance(sp, list) else [sp])] for sp in spellings]

FAMS = {
    "months": [f"MONTH:{m}" for m in MONTHS],
    "weekdays": [f"WDAY:{w}" for w in WEEKDAYS],
    "major_canon": key_labels(PC_CANON_MAJOR, "major"),
    "minor_canon": key_labels(PC_CANON_MINOR, "minor"),
    "major_sharp": key_labels(PC_ALL_SHARP, "major"),
    "major_flat": key_labels(PC_ALL_FLAT, "major"),
    "major_merged": key_labels([ENH[i] for i in range(12)], "major"),
    "minor_merged": key_labels([ENH[i] for i in range(12)], "minor"),
    "keys_merged_modes": [[f"KEY:{s}:{m}" for s in ENH[i] for m in ("major", "minor")] for i in range(12)],
}

report = {}
for g in groups:
    T = load_tally(path, g)
    report[g] = {"ndocs": T["ndocs"], "nwords": T["nwords"], "Z": T["Z"], "cof_docs": T["cof_docs"]}
    for fam0, labels in FAMS.items():
      for stat in ("mstar", "pmi"):
        fam = fam0 if stat == "mstar" else fam0 + "@pmi"
        M, rho, uni, C = build_M(T, labels, return_counts=True, stat=stat)
        sp = spectrum_from_M(M)
        boots = bootstrap_M(T, labels, n_boot=200, stat=stat)
        bprof = np.array([spectrum_from_M(b)["profile_abs_lambda"] for b in boots])
        blam = np.array([spectrum_from_M(b)["lambda"] for b in boots])
        r = {"labels": [l if isinstance(l, str) else "/".join(l) for l in labels], "uni": uni.tolist(),
             "C": C.tolist(), "M": M.tolist(), "rho": rho.tolist(), "kappa": sp["kappa"].tolist(),
             "residual": sp["residual"].tolist(), "circ_frac_offdiag": sp["circ_frac_offdiag"],
             "lambda": sp["lambda"].tolist(), "profile_abs_lambda": sp["profile_abs_lambda"].tolist(),
             "profile_absM": sp["profile_absM"].tolist(),
             "boot_profile_mean": bprof.mean(0).tolist(), "boot_profile_sd": bprof.std(0).tolist(),
             "boot_lambda_sd": blam.std(0).tolist(),
             "eig_M": np.linalg.eigvalsh((M + M.T) / 2).tolist()}
        report[g][fam] = r
        n = M.shape[0]
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        im = axes[0].imshow(M, cmap="RdBu_r", **({"vmin": -2, "vmax": 2} if stat == "mstar" else {})); axes[0].set_title(f"M* {fam} [{g}]"); plt.colorbar(im, ax=axes[0])
        axes[0].set_xticks(range(n)); axes[0].set_xticklabels(r["labels"], rotation=90, fontsize=6); axes[0].set_yticks(range(n)); axes[0].set_yticklabels(r["labels"], fontsize=6)
        im = axes[1].imshow(sp["residual"], cmap="RdBu_r"); axes[1].set_title(f"residual (circ frac offdiag={sp['circ_frac_offdiag']:.2f})"); plt.colorbar(im, ax=axes[1])
        axes[2].plot(range(n), sp["kappa"], "o-"); axes[2].set_title("kappa(d)"); axes[2].set_xlabel("d")
        ks = np.arange(n); axes[3].bar(ks, sp["lambda"]); axes[3].set_title("lambda_k (DFT of kappa)"); axes[3].set_xlabel("k")
        fig.tight_layout(); fig.savefig(f"{figdir}/{fam}__{g}.png", dpi=110); plt.close(fig)
        print(f"[{g}] {fam}: n_uni={uni.sum():.0f} min_uni={uni.min():.0f} circfrac={sp['circ_frac_offdiag']:.3f} "
              f"profile(|lam|)={np.round(sp['profile_abs_lambda'],3)} sd={np.round(bprof.std(0),3)}", flush=True)
json.dump(report, open(f"{outdir}/report.json", "w"))
print("saved", outdir)
