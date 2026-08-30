"""Checkpoint trajectory of the neutral-view residual correspondence (E_modulation and C_harmonic × A_win40 / D_doc; rich
nuisance model) plus line|circle, circle|line, twin ECI/asymmetry per checkpoint. Usage: python -m phase5.ckpt_fingerprint"""
import json, os, sys, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, S, GLYPH, ENH_PAIRS, candidate_geometries, n
from scipy.stats import spearmanr, rankdata
revs = ["stage1-step300-tokens1B", "stage1-step10000-tokens21B", "stage1-step23100-tokens49B", "stage1-step50000-tokens105B", "stage1-step140000-tokens294B", "stage1-step480000-tokens1007B", "stage1-step950000-tokens1993B", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B", "stage2-ingredient1-step23852-tokens51B", "stage2-ingredient2-step23852-tokens51B"]
Zc = np.load("results/phase5/cond_wikipedia.npz"); uni = Zc["uni"].astype(float); pc = (7 * S) % 12
off = np.ones((n, 12), bool)
for i in range(n): off[i, pc[i]] = False
I, J = np.where(off); CLASS_S = np.array([S[np.where(pc == z)[0]].mean() for z in range(12)]); clsfreq = np.array([np.log(uni[np.where(pc == z)[0]].sum() + 1) for z in range(12)])
def merge_cols(L):
    out = np.full((n, 12), -np.inf)
    for z in range(12): out[:, z] = np.logaddexp.reduce(L[:, np.where(pc == z)[0]], axis=1)
    return out
circ = np.minimum((pc[I] - J) % 12, (J - pc[I]) % 12).astype(float); line = np.abs(S[I] - CLASS_S[J]); signed = CLASS_S[J] - S[I]
base = np.column_stack([circ, line, signed, (GLYPH[I] != 0).astype(float), np.log(uni[I] + 1), clsfreq[J]])
rich = np.column_stack([base, circ ** 2, line ** 2, signed ** 2, circ * line, np.cos(2 * np.pi * circ / 12), np.cos(4 * np.pi * circ / 12), np.cos(6 * np.pi * circ / 12)] + [(I == i).astype(float) for i in range(n)])
def resid(y, X, lam=1.0):
    X1 = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)]); A = X1.T @ X1 + lam * np.diag([0] + [1] * X.shape[1]); b = np.linalg.solve(A, X1.T @ y); return y - X1 @ b
lf = np.log(uni + 1); G = candidate_geometries(logfreq=lf); iu = np.triu_indices(n, 1); NP = len(iu[0]); R = {k: rankdata(v[iu]) for k, v in G.items()}; CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "commonness"]
pk = np.array([[i, j] for i, j in zip(*iu)]); eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
def partial(d, t, c):
    tt = rankdata(d); X = np.column_stack([np.ones(NP)] + [R[x] for x in c]); g = R[t]; rt = tt - X @ np.linalg.lstsq(X, tt, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg))) if rt @ rt > 1e-12 else float("nan")
rng = np.random.default_rng(0); PERMS = [rng.permutation(n) for _ in range(2000)]
C = {}
for ex in ("A_win40", "D_doc"):
    M = Zc[ex].astype(float); Rr = (M + 0.5) / (M + 0.5).sum(1, keepdims=True); C[ex] = (M, merge_cols(np.log(Rr)))
print(f"{'checkpoint (tokens)':22s} fam | line|circle circle|line | twin ECI asym | neutral residual r (rich): A_win40 (p)  D_doc (p) | KL of rows to uniform")
rows = []
for r in revs + ["main"]:
    t = "olmo2_1b" if r == "main" else f"olmo2_1b_{r}"; f = f"results/phase2/behavior/{t}.json"
    if not os.path.exists(f): continue
    Jb = json.load(open(f))
    for fam in ("E_modulation", "C_harmonic"):
        Ls = [np.array(Jb[k]["total"]) for k in Jb if k.startswith(fam)]
        if not Ls: continue
        L = np.mean(Ls, 0); L = L - np.logaddexp.reduce(L, axis=1, keepdims=True); Sm = -(L + L.T) / 2; d = Sm[iu]
        lc = partial(d, "line_fifths", CTRL + ["circle_fifths"]); cl = partial(d, "circle_fifths", CTRL + ["line_fifths"]); e = (rankdata(d) / NP)[eidx].mean(); asym = np.mean([abs(L[i, a] - L[i, b]) for a, b in ENH_PAIRS for i in range(n)])
        logQ = merge_cols(L); qr = resid(logQ[off], rich); res = {}
        for ex, (M, logC) in C.items():
            cr = resid(logC[off], rich); rr = spearmanr(qr, cr).correlation
            nul = []
            for p in PERMS:
                Mp = M[np.ix_(p, p)]; Rp = (Mp + 0.5) / (Mp + 0.5).sum(1, keepdims=True); nul.append(spearmanr(qr, resid(merge_cols(np.log(Rp))[off], rich)).correlation)
            res[ex] = (rr, float((int((np.array(nul) >= rr).sum()) + 1) / (len(nul) + 1)))   # finite-sample estimator (b+1)/(B+1)
        ent = float(np.mean([-(np.exp(L[i]) * L[i]).sum() for i in range(n)]))
        lab = r.replace("stage1-step", "s1 ").replace("stage2-ingredient3-step", "s2i3 ").replace("stage2-ingredient1-step", "s2i1 ").replace("stage2-ingredient2-step", "s2i2 ").replace("-tokens", " ") if r != "main" else "released (other run)"
        print(f"{lab:22s} {fam[:1]}   | {lc:+.2f}       {cl:+.2f}      | {e:.2f}    {asym:.2f} | {res['A_win40'][0]:+.2f} ({res['A_win40'][1]:.3f})   {res['D_doc'][0]:+.2f} ({res['D_doc'][1]:.3f}) | row entropy {ent:.2f}")
        rows.append({"rev": r, "fam": fam, "lc": lc, "cl": cl, "eci": e, "asym": asym, "resid": res, "entropy": ent})
json.dump(rows, open("results/phase5/ckpt_fingerprint.json", "w"), indent=1)
