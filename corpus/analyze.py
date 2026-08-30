"""Build M* matrices from merged tallies; circulant projection; kernel DFT; predicted spectra.

Usage (library): from corpus.analyze import load_tally, build_M, spectrum_from_M
"""
from __future__ import annotations
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import circulant_projection, kernel_dft, predicted_energy_from_M, paired_vector

SUM_F = sum(17 - d for d in range(1, 17))  # 136


def load_tally(path, group="all"):
    d = json.load(open(path))
    return d[group]


def key_label(spelling, mode):
    return f"KEY:{spelling}:{mode}"


def build_M(tally, labels, min_count=0, return_counts=False, stat="mstar"):
    """labels: ordered list of concept label strings (each may be a list of alternative spellings to merge).
    Returns M* (n×n), rho, unigram counts, weighted co-occurrence counts."""
    N = tally["nwords"]; Z = tally["Z"]
    n = len(labels)
    uni = np.zeros(n); C = np.zeros((n, n))
    for i, lab in enumerate(labels):
        labs = lab if isinstance(lab, list) else [lab]
        for l in labs:
            uni[i] += tally["uni"].get(l, 0)
    for i, li in enumerate(labels):
        lis = li if isinstance(li, list) else [li]
        for j, lj in enumerate(labels):
            ljs = lj if isinstance(lj, list) else [lj]
            for a in lis:
                for b in ljs:
                    C[i, j] += tally["co"].get(f"{a}|{b}", 0.0)
    Pi = uni / N
    Pij = C / Z
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = Pij / np.outer(Pi, Pi)
        if stat == "mstar":
            M = 2 * (rho - 1) / (rho + 1)
            M = np.where(np.isfinite(M), M, -2.0)  # zero co-occurrence -> M* = -2
        elif stat == "pmi":
            M = np.log(rho)
            M = np.where(np.isfinite(M), M, np.nan)
            M = np.where(np.isnan(M), np.nanmin(M) - 1.0, M)  # zero cells: floor below the minimum observed PMI
        else:
            raise ValueError(stat)
    if return_counts:
        return M, rho, uni, C
    return M


def spectrum_from_M(M):
    """Returns dict with kappa, residual, circulant fraction (off-diagonal, centered), lambda_k,
    predicted |lambda_k| paired profile, and the |M|-based prediction (general, not assuming circulant)."""
    kappa, R, _ = circulant_projection(M)
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    Mo = M[off] - M[off].mean()
    Ro = R[off]
    frac_off = 1 - (Ro ** 2).sum() / (Mo ** 2).sum()
    lam = kernel_dft(kappa).real
    Eabs = predicted_energy_from_M(M, use_abs=True)   # general |M|-block prediction
    Ecirc_abs = np.abs(lam); Ecirc_abs[0] = 0
    return {"kappa": kappa, "residual": R, "circ_frac_offdiag": float(frac_off), "lambda": lam,
            "profile_abs_lambda": paired_vector(Ecirc_abs) / paired_vector(Ecirc_abs).sum(),
            "profile_absM": paired_vector(Eabs) / paired_vector(Eabs).sum(),
            "E_absM": Eabs}


def bootstrap_M(tally, labels, n_boot=200, seed=0, stat="mstar"):
    """Multinomial bootstrap of the unigram and co-occurrence counts (approximate: treats weighted
    counts as Poisson with the observed mean). Returns array (n_boot, n, n) of M*."""
    rng = np.random.default_rng(seed)
    M, rho, uni, C = build_M(tally, labels, return_counts=True)
    N = tally["nwords"]; Z = tally["Z"]
    out = []
    for _ in range(n_boot):
        u = rng.poisson(uni)
        c = rng.poisson(C)
        c = np.triu(c) + np.triu(c, 1).T
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (c / Z) / np.outer(u / N, u / N)
            if stat == "mstar":
                m = 2 * (r - 1) / (r + 1); m = np.where(np.isfinite(m), m, -2.0)
            else:
                m = np.log(r); m = np.where(np.isfinite(m), m, np.nan); m = np.where(np.isnan(m), np.nanmin(m) - 1.0, m)
        out.append(m)
    return np.asarray(out)
