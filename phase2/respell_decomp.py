"""Track 8 analysis: for the 21-label respelling design, per family/position/layer:
  - enharmonic collapse: mean distance between the 9 respelled pairs vs (a) letter-adjacent cross-glyph non-enharmonic
    control pairs and (b) all pairs (rank-based ECI over the 9 pairs, null = 9 random pairs / glyph-preserving);
  - cross-fitted decomposition: on training families, define the 'spelling' direction set as the span of pair
    differences (h_spellA − h_spellB) for the 9 tonics and the 'semantic' set as the span of pair means minus grand
    mean; evaluate on held-out families how much of the between-label variance lies in each subspace (variance
    explained by projecting held-out centered representations onto the fitted subspaces, with dimension-matched random
    subspaces as the null).
Usage: python phase2/respell_decomp.py <tag>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.contexts import FAMILIES
from pf.fourier import center
from scipy.stats import rankdata
tag = sys.argv[1]; Z = np.load(f"results/phase2/respell/{tag}.npz", allow_pickle=True); LABELS = list(Z["labels"]); PCS = Z["pcs"]; n = len(LABELS)
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
LET = np.array([ord(l[0]) - 65 for l in LABELS]); GL = np.array([-1 if l.endswith("b") else (1 if l.endswith("#") else 0) for l in LABELS])
iu = np.triu_indices(n, 1); pk = np.array([[i, j] for i, j in zip(*iu)]); NP = len(iu[0])
pidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in PAIRS]
ctrl = [(i, j) for i, j in zip(*iu) if abs(LET[i] - LET[j]) == 1 and GL[i] != GL[j] and PCS[i] != PCS[j]]
cidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ctrl]
rng = np.random.default_rng(0)
def dist(H): Hc = center(H); return np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)
fams = list(FAMILIES)
def load(fam, pos): return np.stack([Z[f"{fam}__t{ti}__{pos}"] for ti in range(4)], 0).mean(0)  # (L+1, 21, d)
res = {}
print(f"{'family':13s} pos   layer | ECI(9 enh pairs) ctrl-pairs  p(free) | var frac in fitted spelling / semantic subspace on held-out families (random-subspace null)")
for pos in ("last", "final"):
    Hf = {fam: load(fam, pos) for fam in fams}; Lp1 = Hf[fams[0]].shape[0]
    for fam in fams:
        rows = []
        for l in (Lp1 // 4, Lp1 // 2, 3 * Lp1 // 4, Lp1 - 2, Lp1 - 1):
            H = Hf[fam][l]; D = dist(H); d = D[iu]
            if np.allclose(d, 0): continue
            ranks = rankdata(d) / NP; eci = float(ranks[pidx].mean()); ec = float(ranks[cidx].mean())
            null = np.array([ranks[rng.choice(NP, 9, replace=False)].mean() for _ in range(2000)]); p = float(np.mean(null <= eci))
            # cross-fitted subspaces: fit on the other five families at the same layer
            train = [g for g in fams if g != fam]; Ht = np.concatenate([center(Hf[g][l]) for g in train], 0)
            diffs = np.concatenate([np.stack([center(Hf[g][l])[i] - center(Hf[g][l])[j] for i, j in PAIRS]) for g in train], 0)
            means = np.concatenate([np.stack([(center(Hf[g][l])[i] + center(Hf[g][l])[j]) / 2 for i, j in PAIRS]) for g in train], 0)
            k = 8
            Us = np.linalg.svd(diffs, full_matrices=False)[2][:k].T; Um = np.linalg.svd(means, full_matrices=False)[2][:k].T
            Hc = center(H); tot = float((Hc ** 2).sum())
            fs = float(((Hc @ Us) ** 2).sum()) / tot; fm = float(((Hc @ Um) ** 2).sum()) / tot
            rn = []
            for _ in range(50):
                Q = np.linalg.qr(rng.standard_normal((Hc.shape[1], k)))[0]; rn.append(float(((Hc @ Q) ** 2).sum()) / tot)
            # within-subspace enharmonic collapse: distances of the 9 pairs after projecting OUT the spelling subspace
            Hp = Hc - (Hc @ Us) @ Us.T; Dp = np.linalg.norm(Hp[:, None] - Hp[None], axis=-1); rp = rankdata(Dp[iu]) / NP; eci_p = float(rp[pidx].mean())
            rows.append({"layer": l, "eci": eci, "eci_ctrl": ec, "p": p, "frac_spelling": fs, "frac_semantic": fm, "frac_random": float(np.mean(rn)), "eci_after_removing_spelling": eci_p})
            print(f"{fam:13s} {pos:5s} L{l:<3d} | {eci:.2f} (ctrl {ec:.2f}) p={p:.3f} | spelling {fs:.2f} semantic {fm:.2f} random {np.mean(rn):.3f} | ECI after removing spelling subspace {eci_p:.2f}", flush=True)
        res[f"{fam}__{pos}"] = rows
json.dump(res, open(f"results/phase2/respell/{tag}_decomp.json", "w"))
