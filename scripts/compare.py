"""Compare corpus-predicted paired profiles with model layerwise profiles.
Usage: python scripts/compare.py <corpus report.json> <corpus group> <spectra tag> <mapping json>
mapping json: {"model_family_prefix": "corpus_family", ...}  e.g. {"months": "months", "major_canon": "major_canon"}
Outputs results/compare/<tag>/*.json and figures/compare_<tag>/*.png
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import spectrum_cosine
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rep_path, group, tag, mapping = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
rep = json.load(open(rep_path))[group]
spec = json.load(open(f"results/spectra/{tag}/_summary.json"))
outdir = f"results/compare/{tag}"; figdir = f"figures/compare_{tag}"
os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)


def logratio(p):  # log(P1/P5) for 12-point families; None otherwise
    return float(np.log(p[0] / p[4])) if len(p) == 6 and p[0] > 0 and p[4] > 0 else None


out = {}
for mfam, cfam in mapping.items():
    for stat in ("", "@pmi"):
        cp = np.array(rep[cfam + stat]["profile_abs_lambda"])
        cpM = np.array(rep[cfam + stat]["profile_absM"])
        for name, res in spec.items():
            if not name.startswith(mfam + "__"): continue
            for pos in ("last", "mean", "anchor", "final", "embed"):
                if pos not in res: continue
                P = np.array(res[pos]["profile"])
                rows = []
                for l in range(P.shape[0]):
                    p = P[l]
                    if np.isnan(p).any(): rows.append(None); continue
                    rows.append({"cos_abslam": spectrum_cosine(p, cp), "cos_absM": spectrum_cosine(p, cpM),
                                 "spearman_abslam": float(spearmanr(p, cp).correlation),
                                 "model_logP1P5": logratio(p), "corpus_logP1P5": logratio(cp)})
                out[f"{name}[{pos}]{stat}"] = {"corpus_profile": cp.tolist(), "corpus_profile_absM": cpM.tolist(), "layers": rows}
json.dump(out, open(f"{outdir}/compare.json", "w"), indent=1)
# summary print
for k, v in out.items():
    rows = [r for r in v["layers"] if r]
    if not rows: continue
    cos = [r["cos_abslam"] for r in rows]; sp = [r["spearman_abslam"] for r in rows]
    best = int(np.argmax(cos))
    print(f"{k:60s} corpus={np.round(v['corpus_profile'],3)} best-layer cos={max(cos):.3f}@{best} spearman@best={sp[best]:.2f} "
          f"final-layer cos={cos[-1]:.3f}")
