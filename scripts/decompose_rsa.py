"""Which component of the corpus PMI does the model Gram track? RSA of model Gram against:
corpus PMI (full), its circulant part, its non-circulant residual, pure fifths-distance, pure chromatic-distance,
black/white block, and unigram-commonness outer product; each with permutation null. Also partial correlations.
Usage: python scripts/decompose_rsa.py <tag> <corpus report.json> <fam> <pos> [layers]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import center, circulant_projection, project_out
from scipy.stats import spearmanr

tag, rep_path, fam, pos = sys.argv[1:5]
layers = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
rep = json.load(open(rep_path))["all"]
z = np.load(f"results/multictx/{tag}/{fam}.npz", allow_pickle=True)
Hs = z[pos]  # (n_ctx, L+1, 12, d)
r = rep[f"{fam}_canon@pmi"]; C = np.array(r["M"]); uni = np.array(r["uni"])
kappa, R, _ = circulant_projection(C); Ccirc = C - R
x = np.arange(12)
fd = np.minimum((7 * (x[:, None] - x[None])) % 12, (7 * (x[None] - x[:, None])) % 12)  # fifths distance
cd = np.minimum((x[:, None] - x[None]) % 12, (x[None] - x[:, None]) % 12)                 # chromatic distance
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float)
blk = (BLACK[:, None] == BLACK[None]).astype(float)            # same-color block
lf = np.log(uni); common = lf[:, None] + lf[None]              # commonness (additive in log-space)
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
names_c = PC_CANON_MAJOR if fam == "major" else PC_CANON_MINOR
letters = np.array([ord(n[0]) - ord("A") for n in names_c])
same_letter = (letters[:, None] == letters[None]).astype(float)
alpha_d = np.abs(letters[:, None] - letters[None]).astype(float)
iu = np.triu_indices(12, 1)
targets = {"PMI": C, "PMI_circ": Ccirc, "PMI_resid": R, "-fifths_dist": -fd, "-chrom_dist": -cd, "black_block": blk, "commonness": common, "same_letter": same_letter, "-alpha_dist": -alpha_d}
rng = np.random.default_rng(0)
perms = [rng.permutation(12) for _ in range(2000)]


def rsa(G, T): return float(spearmanr(G[iu], T[iu]).correlation)


def rsa_z(G, T):
    r0 = rsa(G, T); null = np.array([rsa(G[np.ix_(p, p)], T) for p in perms]); return r0, (r0 - null.mean()) / null.std()


print(f"## {tag} {fam} [{pos}]  corpus PMI: circulant fraction {r['circ_frac_offdiag']:.2f}; targets' mutual RSA:")
names = list(targets)
for a in names:
    print(f"   {a:13s}" + " ".join(f"{rsa(targets[a], targets[b]):+.2f}" for b in names))
print("layer | " + " | ".join(f"{n:>13s}" for n in names) + "  (Spearman, z vs relabel null)")
Lp1 = Hs.shape[1]
for l in (layers or range(Lp1)):
    Havg = Hs[:, l].mean(0); Hc = center(Havg); G = Hc @ Hc.T
    vals = [rsa_z(G, targets[n]) for n in names]
    print(f"{l:5d} | " + " | ".join(f"{v:+.2f} ({zz:+4.1f})" for v, zz in vals))
# partial: model Gram vs fifths distance controlling for black block + commonness (rank-based)
from scipy.stats import rankdata
def partial(G, T, controls):
    g = rankdata(G[iu]); t = rankdata(T[iu]); X = np.column_stack([np.ones(len(g))] + [rankdata(c[iu]) for c in controls])
    rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]; rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
    if np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)) or np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)): return float("nan")
    return float(np.corrcoef(rg, rt)[0, 1])
ctrl = [blk, common, same_letter, alpha_d]
print("partial Spearman of model Gram with -fifths_dist | -chrom_dist | PMI, controlling for black_block + commonness + same_letter + alpha_dist:")
for l in (layers or range(0, Lp1, 2)):
    Havg = Hs[:, l].mean(0); Hc = center(Havg); G = Hc @ Hc.T
    print(f"{l:5d}  fifths={partial(G, -fd, ctrl):+.3f}  chrom={partial(G, -cd, ctrl):+.3f}  PMI={partial(G, C, ctrl):+.3f}   [corpus PMI itself: fifths={partial(C, -fd, ctrl):+.3f} chrom={partial(C, -cd, ctrl):+.3f}]")
