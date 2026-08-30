"""Analyze multi-context extraction: averaged spectra, black-projection, isotropy, white-key RSA,
and Gram RSA vs corpus PMI (with permutation null). Usage: python scripts/multictx_analyze.py <tag> <corpus report.json> [group]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import mode_energies, paired_vector, project_out, mode_isotropy, rsa_line, center
from scipy.stats import spearmanr

tag, rep_path = sys.argv[1], sys.argv[2]; group = sys.argv[3] if len(sys.argv) > 3 else "all"
rep = json.load(open(rep_path))[group]
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float)
W_FIFTHS = [5, 0, 7, 2, 9, 4, 11]; W_CHROM = [0, 2, 4, 5, 7, 9, 11]; WHITE = [0, 2, 4, 5, 7, 9, 11]
iu = np.triu_indices(12, 1); iuw = np.triu_indices(7, 1)
rng = np.random.default_rng(0)


def gram(H):
    Hc = center(H); return Hc @ Hc.T


def rsa(G, C, idx=None):
    if idx is not None: G = G[np.ix_(idx, idx)]; C = C[np.ix_(idx, idx)]; u = np.triu_indices(len(idx), 1)
    else: u = iu
    return float(spearmanr(G[u], C[u]).correlation)


def projM(C, v):
    n = C.shape[0]; P = np.eye(n) - np.ones((n, n)) / n; v = v - v.mean(); Q = P - np.outer(v, v) / (v @ v); return Q @ C @ Q


out = {}
for fam in ("major", "minor"):
    f = f"results/multictx/{tag}/{fam}.npz"
    if not os.path.exists(f): continue
    z = np.load(f, allow_pickle=True)
    Cpmi = np.array(rep[f"{fam}_canon@pmi"]["M"]); Cc = center(center(Cpmi).T).T  # double-centered
    Cb = projM(Cpmi, BLACK)
    res = {}
    for pos in ("anchor", "mean", "last"):
        Hs = z[pos]  # (n_ctx, L+1, 12, d)
        nctx, Lp1 = Hs.shape[:2]
        rows = []
        for l in range(Lp1):
            Havg = Hs[:, l].mean(0)
            if np.isnan(Havg).any(): rows.append(None); continue
            v = paired_vector(mode_energies(Havg)); p = v / v.sum()
            Hb = project_out(Havg, BLACK); vb = paired_vector(mode_energies(Hb)); pb = vb / vb.sum()
            G = gram(Havg); Gb = gram(Hb)
            r_all = rsa(G, Cc); r_blk = rsa(Gb, Cb); r_white = rsa(G, Cpmi, WHITE)
            # per-context RSA distribution
            rc = [rsa(gram(Hs[i, l]), Cc) for i in range(nctx)]
            # permutation null for r_all (relabel the 12 concepts)
            null = [rsa(G[np.ix_(pp, pp)], Cc) for pp in (rng.permutation(12) for _ in range(2000))]
            null = np.array(null); z_r = (r_all - null.mean()) / null.std(); p_r = float((null >= r_all).mean())
            nullb = np.array([rsa(gram(project_out(Havg[pp], BLACK)), Cb) for pp in (rng.permutation(12) for _ in range(500))])
            z_rb = (r_blk - nullb.mean()) / nullb.std()
            rows.append({"layer": l, "profile": p.tolist(), "profile_black": pb.tolist(), "iso7": mode_isotropy(Havg, 7)[2], "iso1": mode_isotropy(Havg, 1)[2],
                         "rsa_white_fifths": rsa_line(Havg, W_FIFTHS), "rsa_white_chrom": rsa_line(Havg, W_CHROM),
                         "gramRSA": r_all, "gramRSA_z": float(z_r), "gramRSA_p": p_r, "gramRSA_ctx_mean": float(np.mean(rc)), "gramRSA_ctx_sd": float(np.std(rc)),
                         "gramRSA_black": r_blk, "gramRSA_black_z": float(z_rb), "gramRSA_whiteonly": r_white})
        res[pos] = rows
        print(f"## {tag} {fam} [{pos}]  n_ctx={nctx}")
        print("layer  P1    P5   | P1b   P5b  | iso7 | whiteRSA f/c  | GramRSA(z,p)  ctx mean±sd | GramRSA blackproj (z) | white-only")
        for r in rows:
            if r is None: continue
            print(f"{r['layer']:5d} {r['profile'][0]:.3f} {r['profile'][4]:.3f} | {r['profile_black'][0]:.3f} {r['profile_black'][4]:.3f} | {r['iso7']:.2f} | {r['rsa_white_fifths']:+.2f} {r['rsa_white_chrom']:+.2f} | "
                  f"{r['gramRSA']:+.3f} ({r['gramRSA_z']:+.1f},{r['gramRSA_p']:.3f}) {r['gramRSA_ctx_mean']:+.3f}±{r['gramRSA_ctx_sd']:.3f} | {r['gramRSA_black']:+.3f} ({r['gramRSA_black_z']:+.1f}) | {r['gramRSA_whiteonly']:+.3f}")
    out[fam] = res
json.dump(out, open(f"results/multictx/{tag}/analysis.json", "w"))
