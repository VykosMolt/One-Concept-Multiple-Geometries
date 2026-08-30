"""Seam-pair evidence for the corpus circle: weighted counts of the 15 seam pairs, Poisson bootstrap of
P(circle|line > line|circle), and white-key fifths-line ordering null (all 5040 orderings) for corpus PMI."""
import sys, json, itertools, numpy as np
sys.path.insert(0, '.')
from corpus.analyze import build_M
from pf.families import PC_CANON_MAJOR
from scipy.stats import spearmanr, rankdata
T = json.load(open("results/corpus_merged/wiki_full.json"))["all"]
x = np.arange(12); fp = (7 * x) % 12; signed = np.where(fp <= 6, fp, fp - 12)
circ = np.minimum((fp[:, None] - fp[None]) % 12, (fp[None] - fp[:, None]) % 12).astype(float); line = np.abs(signed[:, None] - signed[None]).astype(float)
iu = np.triu_indices(12, 1); seam = (circ != line)
BLACK = np.array([0,1,0,1,0,0,1,0,1,0,1,0]); blk = (BLACK[:,None]==BLACK[None]).astype(float)
def partial(S, target, controls):
    t = rankdata(S[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]; return float(np.corrcoef(rt, rg)[0, 1])
ENH = {0: ["C", "B#"], 1: ["C#", "Db"], 2: ["D"], 3: ["D#", "Eb"], 4: ["E", "Fb"], 5: ["F", "E#"], 6: ["F#", "Gb"], 7: ["G"], 8: ["G#", "Ab"], 9: ["A"], 10: ["A#", "Bb"], 11: ["B", "Cb"]}
FAMS = {"major_canon": [[f"KEY:{s}:major"] for s in PC_CANON_MAJOR], "major_merged": [[f"KEY:{s}:major" for s in ENH[i]] for i in range(12)]}
rng = np.random.default_rng(0)
for fam, labels in FAMS.items():
    M, rho, uni, C = build_M(T, labels, return_counts=True, stat="pmi"); N = T["nwords"]; Z = T["Z"]
    com = np.log(uni)[:, None] + np.log(uni)[None]
    print(f"## {fam}: min key count {uni.min():.0f}; weighted seam-pair counts (C_ij, f-weighted): median non-seam {np.median(C[iu][~seam[iu]]):.0f}, seam pairs:")
    names = PC_CANON_MAJOR
    print("   " + ", ".join(f"{names[i]}-{names[j]}={C[i,j]:.0f}" for i, j in zip(*iu) if seam[i, j]))
    cl = partial(M, -circ, [blk, com, line]); lc = partial(M, -line, [blk, com, circ])
    wins = 0; B = 500; cls = []; lcs = []
    for _ in range(B):
        u = rng.poisson(uni); c = rng.poisson(C); c = np.triu(c) + np.triu(c, 1).T
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log((c / Z) / np.outer(u / N, u / N)); m = np.where(np.isfinite(m), m, np.nan); m = np.where(np.isnan(m), np.nanmin(m) - 1, m)
        a = partial(m, -circ, [blk, com, line]); b = partial(m, -line, [blk, com, circ]); cls.append(a); lcs.append(b); wins += a > b
    print(f"   circle|line = {cl:+.2f} (boot {np.mean(cls):+.2f} ± {np.std(cls):.2f}), line|circle = {lc:+.2f} (boot {np.mean(lcs):+.2f} ± {np.std(lcs):.2f}); P(circle > line) = {wins/B:.3f}  [Poisson bootstrap on f-weighted counts: SDs understate true uncertainty]")
    # white-key ordering null
    W = [0, 2, 4, 5, 7, 9, 11]; S = M[np.ix_(W, W)]; u7 = np.triu_indices(7, 1); Lm = np.abs(np.arange(7)[:, None] - np.arange(7)[None])
    fifths_order = [5, 0, 7, 2, 9, 4, 11]; pos_in_W = [W.index(k) for k in fifths_order]
    obs = spearmanr(-S[np.ix_(pos_in_W, pos_in_W)][u7], Lm[u7]).correlation
    vals = np.array([spearmanr(-S[np.ix_(p, p)][u7], Lm[u7]).correlation for p in itertools.permutations(range(7))])
    print(f"   white-key fifths-line RSA = {obs:+.2f}; rank among 5040 orderings: {int((vals >= obs).sum())} (p = {(vals >= obs).mean():.4f}); best ordering {vals.max():+.2f}")
