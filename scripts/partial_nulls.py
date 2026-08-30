"""Permutation nulls for the partial-fifths correlation and the white-key fifths-line RSA, with layer-selection correction.
Free null: relabel the 12 concept vectors uniformly. Block-preserving null: permute within black keys and within white keys.
Max-over-layers null: the null statistic is the max over layers of the permuted data.
Usage: python scripts/partial_nulls.py <tag> [source: multictx|predpos] [context] [position]"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import center
from scipy.stats import rankdata, spearmanr
tag = sys.argv[1]; source = sys.argv[2] if len(sys.argv) > 2 else "multictx"
ctx = sys.argv[3] if len(sys.argv) > 3 else None; pos = sys.argv[4] if len(sys.argv) > 4 else "last"
rep = json.load(open("results/corpus/wiki/report.json"))["all"]["major_canon@pmi"]; uni = np.array(rep["uni"])
x = np.arange(12); fp = (7 * x) % 12; fd = np.minimum((fp[:, None] - fp[None]) % 12, (fp[None] - fp[:, None]) % 12).astype(float)
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]); blk = (BLACK[:, None] == BLACK[None]).astype(float); com = np.log(uni)[:, None] + np.log(uni)[None]
from pf.families import PC_CANON_MAJOR
letters = np.array([ord(n[0]) - 65 for n in PC_CANON_MAJOR]); sl = (letters[:, None] == letters[None]).astype(float); al = np.abs(letters[:, None] - letters[None]).astype(float)
iu = np.triu_indices(12, 1); CTRL = [blk, com, sl, al]
Xc = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in CTRL]); Qc = np.linalg.qr(Xc)[0]
tgt = rankdata(-fd[iu]); rt = tgt - Qc @ (Qc.T @ tgt)
def partial_f(G):
    g = rankdata(G[iu]); rg = g - Qc @ (Qc.T @ g)
    return float(rg @ rt / np.sqrt((rg @ rg) * (rt @ rt))) if rg @ rg > 1e-12 else float("nan")
WHITE = [0, 2, 4, 5, 7, 9, 11]; FIFTHS_LINE = [5, 0, 7, 2, 9, 4, 11]
def white_line(H, order=FIFTHS_LINE):
    X = center(H)[order]; D = np.linalg.norm(X[:, None] - X[None], axis=-1); n = 7; L = np.abs(np.arange(n)[:, None] - np.arange(n)[None]); u = np.triu_indices(n, 1)
    return float(spearmanr(D[u], L[u]).correlation)
# exact null for the white-key line: all 5040 orderings of the 7 white keys
PERMS7 = list(itertools.permutations(range(7)))
def white_line_p(H):
    obs = white_line(H); X = center(H)[WHITE]; D = np.linalg.norm(X[:, None] - X[None], axis=-1); u = np.triu_indices(7, 1); Lm = np.abs(np.arange(7)[:, None] - np.arange(7)[None])
    Du = D[u]; vals = np.array([spearmanr(D[np.ix_(p, p)][u], Lm[u]).correlation for p in PERMS7])
    return obs, float((vals >= obs).mean())
rng = np.random.default_rng(0); NPERM = 1000
black_idx = np.where(BLACK == 1)[0]; white_idx = np.where(BLACK == 0)[0]
def perm_free(): return rng.permutation(12)
def perm_block():
    p = np.arange(12); p[black_idx] = black_idx[rng.permutation(5)]; p[white_idx] = white_idx[rng.permutation(7)]; return p
if source == "multictx":
    Hs = np.load(f"results/multictx/{tag}/major.npz", allow_pickle=True)[pos].mean(0)  # (L+1, 12, d) averaged over contexts
else:
    z = np.load(f"results/predict_position/{tag}_H.npz"); Hs = z[f"{ctx}__{pos}"]
Lp1 = Hs.shape[0]
Gs = [center(Hs[l]) @ center(Hs[l]).T for l in range(Lp1)]
obs = np.array([partial_f(G) for G in Gs]); obs = np.where(np.isnan(obs), -9, obs)
best = int(np.argmax(obs))
null_free = np.zeros((NPERM, Lp1)); null_blk = np.zeros((NPERM, Lp1))
for i in range(NPERM):
    p = perm_free(); q = perm_block()
    for l, G in enumerate(Gs):
        null_free[i, l] = partial_f(G[np.ix_(p, p)]); null_blk[i, l] = partial_f(G[np.ix_(q, q)])
null_free = np.nan_to_num(null_free, nan=-9); null_blk = np.nan_to_num(null_blk, nan=-9)
p_layer_free = float((null_free[:, best] >= obs[best]).mean()); p_layer_blk = float((null_blk[:, best] >= obs[best]).mean())
p_max_free = float((null_free.max(1) >= obs[best]).mean()); p_max_blk = float((null_blk.max(1) >= obs[best]).mean())
wl = [white_line(Hs[l]) for l in range(Lp1)]; wb = int(np.argmax(wl)); wobs, wp_layer = white_line_p(Hs[wb])
# layer-max null for white line: permute white-key labels (5040 exact is too slow across layers) -> 1000 random permutations of the 7 white keys
wnull = np.zeros((NPERM, Lp1))
for i in range(NPERM):
    p = np.arange(12); p[white_idx] = white_idx[rng.permutation(7)]
    for l in range(Lp1): wnull[i, l] = white_line(Hs[l][p])
wp_max = float((wnull.max(1) >= wobs).mean())
label = f"{tag} {source} {ctx or ''} [{pos}]"
print(f"{label:45s} partial fifths best {obs[best]:+.3f} @L{best}  p(layer, free)={p_layer_free:.3f} p(layer, block)={p_layer_blk:.3f}  p(max-over-layers, free)={p_max_free:.3f} p(max, block)={p_max_blk:.3f}  null-max mean={null_free.max(1).mean():+.2f} | final L{Lp1-1} {obs[-1]:+.3f} p(free)={float((null_free[:, -1] >= obs[-1]).mean()):.3f}"
      f" | white fifths-line best {wobs:+.2f} @L{wb} p(exact 5040)={wp_layer:.3f} p(max-over-layers)={wp_max:.3f}")
os.makedirs("results/nulls", exist_ok=True)
json.dump({"obs": obs.tolist(), "best": best, "p_layer_free": p_layer_free, "p_layer_block": p_layer_blk, "p_max_free": p_max_free, "p_max_block": p_max_blk,
           "white_line": wl, "white_best": wb, "white_p_exact": wp_layer, "white_p_max": wp_max, "final": float(obs[-1])},
          open(f"results/nulls/{tag}_{source}_{ctx or 'avg'}_{pos}.json", "w"))
