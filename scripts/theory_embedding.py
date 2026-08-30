"""Theory-faithful prediction: W = Phi sqrt|Lambda| from the full V×V M* (or PMI) built over key-containing docs
with helper words; then the key rows' geometry. Usage: python scripts/theory_embedding.py <npz> <stat: mstar|pmi> [d]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import mode_energies, paired_vector, project_out, mode_isotropy, rsa_line, center, circulant_projection
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
from scipy.stats import spearmanr
z = np.load(sys.argv[1], allow_pickle=True); stat = sys.argv[2]; d = int(sys.argv[3]) if len(sys.argv) > 3 else None
C, uni, vocab, N, Z = z["C"], z["uni"], list(z["vocab"]), float(z["N"]), float(z["Z"])
V = len(vocab); vi = {w: i for i, w in enumerate(vocab)}
Pi = uni / N; Pij = C / Z
with np.errstate(divide="ignore", invalid="ignore"):
    rho = Pij / np.outer(Pi, Pi)
    if stat == "mstar": M = 2 * (rho - 1) / (rho + 1); M = np.where(np.isfinite(M), M, -2.0)
    else: M = np.log(rho); M = np.where(np.isfinite(M), M, np.nan); M = np.where(np.isnan(M), np.nanmin(M) - 1, M)
M = (M + M.T) / 2
w, Phi = np.linalg.eigh(M)
order = np.argsort(-np.abs(w)); w, Phi = w[order], Phi[:, order]
if d: w, Phi = w[:d], Phi[:, :d]
W = Phi * np.sqrt(np.abs(w))          # rows = words, W W^T = |M| (Karkada Eq. 30/32)
print(f"{stat}: V={V}, d={W.shape[1]}, |lambda| top5={np.round(np.abs(w[:5]),2)}, #neg among kept={int((w<0).sum())}")
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float)
x = np.arange(12); fd = np.minimum((7*(x[:,None]-x[None]))%12, (7*(x[None]-x[:,None]))%12); cd = np.minimum((x[:,None]-x[None])%12, (x[None]-x[:,None])%12)
iu = np.triu_indices(12, 1)
for fam, names in (("major", PC_CANON_MAJOR), ("minor", PC_CANON_MINOR)):
    idx = [vi.get(f"KEY_{n}_{fam}") for n in names]
    if any(i is None for i in idx): print(fam, "missing keys", [n for n, i in zip(names, idx) if i is None]); continue
    H = W[idx]
    v = paired_vector(mode_energies(H)); p = v / v.sum()
    Hb = project_out(H, BLACK); vb = paired_vector(mode_energies(Hb)); pb = vb / vb.sum()
    G = center(H) @ center(H).T
    print(f"## {fam} keys from full-vocab embedding ({stat}, d={W.shape[1]}):")
    print(f"   profile        {np.round(p,3)}  P1/P5={p[0]/p[4]:.2f}   iso7={mode_isotropy(H,7)[2]:.2f} iso1={mode_isotropy(H,1)[2]:.2f}")
    print(f"   black-projected {np.round(pb,3)}  P1/P5={pb[0]/pb[4]:.2f}")
    from scipy.stats import rankdata
    lf = np.log(uni[idx]); common = lf[:, None] + lf[None]; blk = (BLACK[:, None] == BLACK[None]).astype(float)
    letters = np.array([ord(n[0]) - 65 for n in names]); same_letter = (letters[:, None] == letters[None]).astype(float); alpha_d = np.abs(letters[:, None] - letters[None]).astype(float)
    def partial(T, target, controls):
        t = rankdata(T[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
        rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
    ctrl = [blk, common, same_letter, alpha_d]
    print(f"   Gram RSA: -fifths_dist {spearmanr(G[iu], -fd[iu]).correlation:+.2f}  -chrom_dist {spearmanr(G[iu], -cd[iu]).correlation:+.2f}  black_block {spearmanr(G[iu], blk[iu]).correlation:+.2f}  white fifths-line {rsa_line(H,[5,0,7,2,9,4,11]):+.2f} chrom-line {rsa_line(H,[0,2,4,5,7,9,11]):+.2f}")
    print(f"   partial (ctrl block+commonness+letter+alpha): fifths {partial(G, -fd, ctrl):+.2f}  chrom {partial(G, -cd, ctrl):+.2f}")
    # compare with block-only prediction
    Mb = M[np.ix_(idx, idx)]; kap, _, _ = circulant_projection(Mb)
    print(f"   12x12 block kappa (fifths order): {np.round(kap[[(7*k)%12 for k in range(12)]],2)}")
    # helper-word ablation: zero the key-key block, refactor (Karkada Fig 4)
    M2 = M.copy(); M2[np.ix_(idx, idx)] = 0.0
    w2, P2 = np.linalg.eigh(M2); o2 = np.argsort(-np.abs(w2)); w2, P2 = w2[o2], P2[:, o2]
    if d: w2, P2 = w2[:d], P2[:, :d]
    H2 = (P2 * np.sqrt(np.abs(w2)))[idx]; v2 = paired_vector(mode_energies(H2)); p2 = v2 / v2.sum(); G2 = center(H2) @ center(H2).T
    print(f"   key-key block ABLATED (helper words only): profile {np.round(p2,3)} P1/P5={p2[0]/p2[4]:.2f}; RSA -fifths {spearmanr(G2[iu], -fd[iu]).correlation:+.2f} -chrom {spearmanr(G2[iu], -cd[iu]).correlation:+.2f}")
