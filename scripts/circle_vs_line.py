"""Is the fifths structure a circle or a line of signed accidental count? Compare Spearman/partials of each
matrix with -circle_dist vs -line_dist (signed positions C=0,G=1,...,F#=6,Db=-5,...,F=-1); the two differ on the
15 pairs whose short arc crosses the F#/Db seam. Usage: python scripts/circle_vs_line.py <tags comma>"""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from pf.fourier import center
from scipy.stats import spearmanr, rankdata
x = np.arange(12); fifths_pos = (7 * x) % 12                       # 0..11 around the circle, C=0
signed = np.where(fifths_pos <= 6, fifths_pos, fifths_pos - 12)    # C0 Db-5 D2 Eb-3 E4 F-1 F#6 G1 Ab-4 A3 Bb-2 B5
circ = np.minimum((fifths_pos[:, None] - fifths_pos[None]) % 12, (fifths_pos[None] - fifths_pos[:, None]) % 12).astype(float)
line = np.abs(signed[:, None] - signed[None]).astype(float)
seam = (circ != line)
iu = np.triu_indices(12, 1); print("pairs where circle != line:", int(seam[iu].sum()))
BLACK = np.array([0,1,0,1,0,0,1,0,1,0,1,0]); blk = (BLACK[:,None]==BLACK[None]).astype(float)
rep = json.load(open("results/corpus/wiki/report.json"))["all"]
def partial(T, target, controls):
    t = rankdata(T[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
def report(name, S, uni=None):
    ctrl = [blk] + ([np.log(uni)[:,None] + np.log(uni)[None]] if uni is not None else [])
    print(f"{name:40s} rho(circle)={spearmanr(S[iu], -circ[iu]).correlation:+.2f} rho(line)={spearmanr(S[iu], -line[iu]).correlation:+.2f} | "
          f"circle|line={partial(S, -circ, ctrl + [line]):+.2f}  line|circle={partial(S, -line, ctrl + [circ]):+.2f} | "
          f"seam pairs only: rho(circle)={spearmanr(S[seam & np.triu(np.ones((12,12),bool),1)], -circ[seam & np.triu(np.ones((12,12),bool),1)]).correlation:+.2f}")
for fam in ("major_canon", "minor_canon"):
    r = rep[fam + "@pmi"]; report(f"corpus PMI {fam}", np.array(r["M"]), np.array(r["uni"]))
# theory embedding (d=300)
z = np.load("results/corpus_merged/keydocs_V3000.npz", allow_pickle=True)
C, uni, vocab, N, Z = z["C"], z["uni"], list(z["vocab"]), float(z["N"]), float(z["Z"]); vi = {w: i for i, w in enumerate(vocab)}
with np.errstate(divide="ignore", invalid="ignore"):
    M = np.log((C / Z) / np.outer(uni / N, uni / N)); M = np.where(np.isfinite(M), M, np.nan); M = np.where(np.isnan(M), np.nanmin(M) - 1, M)
M = (M + M.T) / 2; w, Phi = np.linalg.eigh(M); o = np.argsort(-np.abs(w))[:300]; W = Phi[:, o] * np.sqrt(np.abs(w[o]))
from pf.families import PC_CANON_MAJOR
idx = [vi[f"KEY_{n}_major"] for n in PC_CANON_MAJOR]; H = W[idx]; G = center(H) @ center(H).T
report("theory embedding major (d=300)", G, uni[idx])
for tag in sys.argv[1].split(","):
    f = f"results/multictx/{tag}/major.npz"
    if not os.path.exists(f): continue
    Hs = np.load(f, allow_pickle=True)["last"]; ucorp = np.array(rep["major_canon@pmi"]["uni"])
    for l in sorted(set([2, 6, 10, 14, Hs.shape[1] - 5, Hs.shape[1] - 3, Hs.shape[1] - 1])):
        Havg = Hs[:, l].mean(0); G = center(Havg) @ center(Havg).T; report(f"{tag} major last L{l}", G, ucorp)
    P = json.load(open(f"results/predictive/{tag}.json"))
    for c in ("modulates_to", "then_key"):
        L = np.array(P[c]["L"]); report(f"{tag} predictive {c}", (L + L.T) / 2, ucorp)

print("\n--- robustness: model-marginal commonness control (column-mean of L) and PC1 monotonicity in signed position ---")
def pc1(S):
    Sc = S.copy(); np.fill_diagonal(Sc, np.nan); Sc = np.where(np.isnan(Sc), np.nanmean(Sc, 1, keepdims=True), Sc); Sc = (Sc + Sc.T) / 2
    P = np.eye(12) - 1/12; Gc = P @ Sc @ P; w, V = np.linalg.eigh(Gc); v = V[:, np.argmax(w)]
    return float(abs(spearmanr(v, signed).correlation)), float(abs(spearmanr(v, fifths_pos).correlation)), float(abs(spearmanr(np.cos(2*np.pi*fifths_pos/12), v).correlation))
for tag in sys.argv[1].split(","):
    P = json.load(open(f"results/predictive/{tag}.json")); ucorp = np.array(rep["major_canon@pmi"]["uni"])
    for c in ("modulates_to", "then_key", "next_song"):
        L = np.array(P[c]["L"]); S = (L + L.T) / 2; marg = L.mean(0); mm = marg[:, None] + marg[None]
        ctrl = [blk, np.log(ucorp)[:,None] + np.log(ucorp)[None], mm]
        s, f_, cs = pc1(S)
        print(f"{tag} predictive {c:13s}: circle|line,marg={partial(S, -circ, ctrl + [line]):+.2f}  line|circle,marg={partial(S, -line, ctrl + [circ]):+.2f} | |rho(PC1, signed pos)|={s:.2f}  |rho(PC1, cos(circle))|={cs:.2f}")
r = rep["major_canon@pmi"]; Cp = np.array(r["M"]); s, f_, cs = pc1(Cp); print(f"corpus PMI major: |rho(PC1, signed pos)|={s:.2f}  |rho(PC1, cos(circle))|={cs:.2f}")
s, f_, cs = pc1(G_theory) if False else pc1(center(H) @ center(H).T); print(f"theory embedding: |rho(PC1, signed pos)|={s:.2f}  |rho(PC1, cos(circle))|={cs:.2f}")

print("\n--- is the model's 'line' just a spelling class? extra controls: flat-name block (Db,Eb,Ab,Bb), sign block (sharps|C|flats) ---")
hasb = np.array([1 if n.endswith("b") else 0 for n in PC_CANON_MAJOR]); hb = (hasb[:, None] == hasb[None]).astype(float)
sign = np.sign(signed); sb = (sign[:, None] == sign[None]).astype(float)
grad = -np.abs(signed[:, None] - signed[None]).astype(float)   # = -line
for name, S, u in [("corpus PMI major", Cp, np.array(rep["major_canon@pmi"]["uni"]))] + [
        (f"{t} predictive {c}", (lambda L: (L + L.T) / 2)(np.array(json.load(open(f"results/predictive/{t}.json"))[c]["L"])), np.array(rep["major_canon@pmi"]["uni"]))
        for t in sys.argv[1].split(",") for c in ("modulates_to", "then_key")]:
    base = [blk, np.log(u)[:, None] + np.log(u)[None]]
    print(f"{name:34s} line|circle,+flatblock={partial(S, -line, base + [circ, hb]):+.2f}  line|circle,+signblock={partial(S, -line, base + [circ, sb]):+.2f}  "
          f"line|circle,+both={partial(S, -line, base + [circ, hb, sb]):+.2f} | signblock|line,circle={partial(S, sb, base + [circ, line]):+.2f}  flatblock|line,circle={partial(S, hb, base + [circ, line]):+.2f} | circle|line,+both={partial(S, -circ, base + [line, hb, sb]):+.2f}")
