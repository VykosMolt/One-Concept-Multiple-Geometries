"""Corpus-side black-key diagnostics: project the black-key indicator out of M (both sides),
recompute predicted profile; white-key sub-block line RSA; kappa in semitone and fifths order.
Usage: python scripts/corpus_confounds.py <report.json> <group> <fam1,fam2,...>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import predicted_energy_from_M, paired_vector, circulant_projection, kernel_dft
from scipy.stats import spearmanr

rep = json.load(open(sys.argv[1]))[sys.argv[2]]
fams = sys.argv[3].split(",")
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float)
WHITE = [0, 2, 4, 5, 7, 9, 11]; W_FIFTHS = [5, 0, 7, 2, 9, 4, 11]; W_CHROM = [0, 2, 4, 5, 7, 9, 11]
FIFTHS_ORDER = [(7 * i) % 12 for i in range(12)]  # semitone index of fifths position i: C G D A E B F# Db Ab Eb Bb F


def proj_matrix(M, v):
    n = M.shape[0]; P = np.eye(n) - np.ones((n, n)) / n
    v = v - v.mean(); Q = P - np.outer(v, v) / (v @ v)
    return Q @ M @ Q


def profile(M, use_abs=True):
    E = predicted_energy_from_M(M, use_abs=use_abs); v = paired_vector(E); return v / v.sum()


def line_rsa(M, order):
    S = M[np.ix_(order, order)]; n = len(order)
    D = np.abs(np.arange(n)[:, None] - np.arange(n)[None]); iu = np.triu_indices(n, 1)
    return float(spearmanr(-S[iu], D[iu]).correlation)  # larger M = closer


for fam in fams:
    for stat in ("", "@pmi"):
        r = rep[fam + stat]; M = np.array(r["M"]); kappa = np.array(r["kappa"])
        p_raw = profile(M); p_proj = profile(proj_matrix(M, BLACK))
        # circulant-only version
        Mc = np.array([[kappa[(j - i) % 12] for j in range(12)] for i in range(12)])
        pc_raw = profile(Mc); pc_proj = profile(proj_matrix(Mc, BLACK))
        print(f"## {fam}{stat}  circfrac_offdiag={r['circ_frac_offdiag']:.2f}")
        print(f"   kappa semitone order d=0..11 : {np.round(kappa, 2)}")
        print(f"   kappa fifths order  d'=0..11 : {np.round(kappa[[(7*d) % 12 for d in range(12)]], 2)}")
        print(f"   |M|-profile raw      : {np.round(p_raw, 3)}   P1/P5={p_raw[0]/p_raw[4]:.2f}")
        print(f"   |M|-profile blackproj: {np.round(p_proj, 3)}   P1/P5={p_proj[0]/p_proj[4]:.2f}")
        print(f"   circ |lam| raw       : {np.round(pc_raw, 3)}   blackproj: {np.round(pc_proj, 3)}")
        print(f"   white-key line RSA: fifths={line_rsa(M, W_FIFTHS):+.2f} chromatic={line_rsa(M, W_CHROM):+.2f}")
        # black/white block means
        b = BLACK.astype(bool); off = ~np.eye(12, dtype=bool)
        print(f"   mean M: white-white={M[np.ix_(~b,~b)][off[np.ix_(~b,~b)]].mean():.3f} black-black={M[np.ix_(b,b)][off[np.ix_(b,b)]].mean():.3f} white-black={M[np.ix_(~b,b)].mean():.3f}")
