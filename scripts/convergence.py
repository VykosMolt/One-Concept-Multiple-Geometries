"""Corpus-size convergence of the key PMI spectrum: cumulative shards 1,2,4,8,16,41."""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from corpus.merge import merge
from corpus.analyze import build_M, spectrum_from_M
from pf.fourier import circulant_projection
from pf.families import PC_CANON_MAJOR, MONTHS
from scipy.stats import spearmanr, rankdata
files = sorted(f"results/corpus_wiki/{f}" for f in os.listdir("results/corpus_wiki") if f.startswith("train-") and f.endswith(".json"))
x = np.arange(12); fd = np.minimum((7*(x[:,None]-x[None]))%12, (7*(x[None]-x[:,None]))%12); cd = np.minimum((x[:,None]-x[None])%12, (x[None]-x[:,None])%12)
BLACK = np.array([0,1,0,1,0,0,1,0,1,0,1,0]); blk = (BLACK[:,None]==BLACK[None]).astype(float); iu = np.triu_indices(12,1)
def partial(T, target, controls):
    t = rankdata(T[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
out = []
print("shards  words(B)  minKey  | months P1 P2 | keys(PMI) P1 P5 circfrac | RSA fifths chrom | partial fifths (ctrl block+commonness) | keys(M*) P1 P5")
for n in (1, 2, 4, 8, 16, 41):
    T = merge(files[:n])["all"]
    labs = [f"KEY:{k}:major" for k in PC_CANON_MAJOR]
    Mp, rho, uni, C = build_M(T, labs, return_counts=True, stat="pmi"); sp = spectrum_from_M(Mp)
    Mm = build_M(T, labs, stat="mstar"); sm = spectrum_from_M(Mm)
    Mo = build_M(T, [f"MONTH:{m}" for m in MONTHS], stat="mstar"); so = spectrum_from_M(Mo)
    lf = np.log(uni); common = lf[:,None]+lf[None]
    row = {"n": n, "words": T["nwords"], "min_key": float(uni.min()), "months": so["profile_abs_lambda"].tolist(), "keys_pmi": sp["profile_abs_lambda"].tolist(),
           "circfrac": sp["circ_frac_offdiag"], "rsa_f": float(spearmanr(Mp[iu], -fd[iu]).correlation), "rsa_c": float(spearmanr(Mp[iu], -cd[iu]).correlation),
           "partial_f": partial(Mp, -fd, [blk, common]), "keys_mstar": sm["profile_abs_lambda"].tolist()}
    out.append(row)
    print(f"{n:6d}  {T['nwords']/1e9:7.2f}  {uni.min():6.0f} | {so['profile_abs_lambda'][0]:.3f} {so['profile_abs_lambda'][1]:.3f} | {sp['profile_abs_lambda'][0]:.3f} {sp['profile_abs_lambda'][4]:.3f} {sp['circ_frac_offdiag']:.2f} | {row['rsa_f']:+.2f} {row['rsa_c']:+.2f} | {row['partial_f']:+.2f} | {sm['profile_abs_lambda'][0]:.3f} {sm['profile_abs_lambda'][4]:.3f}")
json.dump(out, open("results/corpus/wiki/convergence.json", "w"), indent=1)
