"""Fit geometries to 15x15 predictive matrices (three scorers) per family/template; ECI on behaviour (do enharmonic
twins receive similar probability profiles / are they predicted for each other?), circle|line, line|circle with
relabeling nulls (free, glyph-preserving); the 'merged' scorer is reported for B/C/D (neutral-pitch tasks) and flagged
as inappropriate for spelling tasks. Usage: python phase2/behavior_fit.py <tag>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.keys15 import KEYS15, GLYPH, ENH_PAIRS, candidate_geometries, n, S
from scipy.stats import spearmanr, rankdata
tag = sys.argv[1]; J = json.load(open(f"results/phase2/behavior/{tag}.json"))
corpus = json.load(open("results/corpus_merged/wiki_full.json"))["all"]["uni"]; logfreq = np.log(np.array([corpus.get(f"KEY:{k}:major", 0) + 1 for k in KEYS15], float))
G = candidate_geometries(logfreq=logfreq); iu = np.triu_indices(n, 1); NP = len(iu[0]); R = {k: rankdata(v[iu]) for k, v in G.items()}
CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "commonness"]
def partial(dvec, target, controls):
    t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + [R[c] for c in controls]); g = R[target]
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg))) if rt @ rt > 1e-12 and rg @ rg > 1e-12 else float("nan")
pk = np.array([[i, j] for i, j in zip(*iu)]); idx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
rng = np.random.default_rng(0); flats = np.where(GLYPH == -1)[0]; nats = np.where(GLYPH == 0)[0]; sharps = np.where(GLYPH == 1)[0]
def perm_glyph():
    p = np.arange(n); p[flats] = flats[rng.permutation(len(flats))]; p[nats] = nats[rng.permutation(len(nats))]; p[sharps] = sharps[rng.permutation(len(sharps))]; return p
PF = [rng.permutation(n) for _ in range(1000)]; PG = [perm_glyph() for _ in range(1000)]
def stats(L):
    Sm = (L + L.T) / 2; d = -Sm[iu]; ranks = rankdata(d) / NP
    return {"eci": float(ranks[idx].mean()), "cl": partial(d, "circle_fifths", CTRL + ["line_fifths"]), "lc": partial(d, "line_fifths", CTRL + ["circle_fifths"]),
            "rsa_circle": float(spearmanr(d, G["circle_fifths"][iu]).correlation), "rsa_line": float(spearmanr(d, G["line_fifths"][iu]).correlation)}
out = {}
print(f"{'family/template':24s} scorer | ECI (p_free/p_glyph) | circle|line (p) | line|circle (p) | RSA circle line | top-1 intervals on line (excl self)")
for key, rec in J.items():
    fam = key.split("__")[0]
    for scorer in ("total", "mean", "merged"):
        if scorer == "merged" and fam not in ("B_enharmonic", "C_harmonic", "D_chord"): continue
        L = np.array(rec[scorer]); s = stats(L)
        nf = np.array([[stats(L[np.ix_(p, p)])[k] for k in ("eci", "cl", "lc")] for p in PF]); ng = np.array([[stats(L[np.ix_(p, p)])[k] for k in ("eci", "cl", "lc")] for p in PG])
        pe = (np.mean(nf[:, 0] <= s["eci"]), np.mean(ng[:, 0] <= s["eci"])); pc = (np.mean(nf[:, 1] >= s["cl"]), np.mean(ng[:, 1] >= s["cl"])); pl = (np.mean(nf[:, 2] >= s["lc"]), np.mean(ng[:, 2] >= s["lc"]))
        top = np.argmax(np.where(np.eye(n, dtype=bool), -1e9, L), axis=1); ivs = [int(S[j] - S[i]) for i, j in enumerate(top)]
        from collections import Counter; ch = Counter(ivs).most_common(4)
        out[f"{key}|{scorer}"] = {**s, "p_eci": pe, "p_cl": pc, "p_lc": pl, "top_intervals": ivs}
        print(f"{key:24s} {scorer:6s} | {s['eci']:.2f} ({pe[0]:.3f}/{pe[1]:.3f}) | {s['cl']:+.2f} ({pc[0]:.3f}/{pc[1]:.3f}) | {s['lc']:+.2f} ({pl[0]:.3f}/{pl[1]:.3f}) | {s['rsa_circle']:+.2f} {s['rsa_line']:+.2f} | " + " ".join(f"{k:+d}:{v}" for k, v in ch), flush=True)
json.dump(out, open(f"results/phase2/behavior/{tag}_fit.json", "w"))
