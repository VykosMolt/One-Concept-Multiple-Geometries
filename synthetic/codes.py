"""Fixed codeword set with open-line geometry, and its assignments to the 15 states."""
import numpy as np
from scipy.stats import spearmanr
from synthetic.laws import STATES, N_STATES, TWIN_PAIRS
L = 8; NSYM = 3
CODEWORDS = np.array([[(i + p) // L for p in range(L)] for i in range(N_STATES)])   # symbols 0..2; Hamming = min(|i-j|, L)


def hamming(C):
    return (C[:, None, :] != C[None, :, :]).sum(-1)


def levenshtein(a, b):
    m, k = len(a), len(b); D = np.zeros((m + 1, k + 1), int); D[:, 0] = range(m + 1); D[0, :] = range(k + 1)
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return D[m, k]


def lev_matrix(C):
    return np.array([[levenshtein(list(a), list(b)) for b in C] for a in C])


def prefix_overlap(C):
    n = len(C); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 0
            while k < C.shape[1] and C[i, k] == C[j, k]: k += 1
            M[i, j] = k
    return M


SEM_LINE = np.abs(STATES[:, None] - STATES[None])
_z = STATES % 12; SEM_CIRC = np.minimum((_z[:, None] - _z[None]) % 12, (_z[None] - _z[:, None]) % 12)
iu = np.triu_indices(N_STATES, 1)


def assignment(kind: str, perm_seed: int | None = None, max_abs_rho: float = 0.15, max_tries: int = 10000):
    """Returns array idx[n] = codeword index assigned to state n (states in order -7..7)."""
    if kind == "aligned":
        return np.arange(N_STATES)
    rng = np.random.default_rng(perm_seed)
    H = hamming(CODEWORDS)
    for _ in range(max_tries):
        p = rng.permutation(N_STATES)
        rho = spearmanr(H[np.ix_(p, p)][iu], SEM_LINE[iu]).correlation
        if abs(rho) < max_abs_rho: return p
    raise RuntimeError("no permutation found")


def assignment_with_alignment(target_rho: float, seed: int, tol: float = 0.05, max_tries: int = 200000):
    """Dose-response helper: permutation whose Spearman(code distance, |n-m|) is within tol of target_rho."""
    rng = np.random.default_rng(seed); H = hamming(CODEWORDS); best = None
    for _ in range(max_tries):
        p = rng.permutation(N_STATES); rho = spearmanr(H[np.ix_(p, p)][iu], SEM_LINE[iu]).correlation
        if best is None or abs(rho - target_rho) < abs(best[1] - target_rho): best = (p, rho)
        if abs(rho - target_rho) < tol: return p, rho
    return best


def geometry_report(idx):
    C = CODEWORDS[idx]; H = hamming(C); Lv = lev_matrix(C); Po = prefix_overlap(C)
    twin = np.zeros((N_STATES, N_STATES)); 
    for a, b in TWIN_PAIRS: twin[a, b] = twin[b, a] = 1
    return {"rho_hamming_line": float(spearmanr(H[iu], SEM_LINE[iu]).correlation), "rho_hamming_circle": float(spearmanr(H[iu], SEM_CIRC[iu]).correlation),
            "rho_lev_line": float(spearmanr(Lv[iu], SEM_LINE[iu]).correlation), "rho_prefix_line": float(spearmanr(-Po[iu], SEM_LINE[iu]).correlation),
            "mean_hamming_twins": float(np.mean([H[a, b] for a, b in TWIN_PAIRS])), "mean_hamming_adjacent": float(np.mean([H[i, i + 1] for i in range(N_STATES - 1)])),
            "symbol_counts": np.bincount(C.ravel(), minlength=NSYM).tolist(), "lengths": sorted(set(len(c) for c in C))}


if __name__ == "__main__":
    print("codewords (rows = index 0..14):"); print(CODEWORDS)
    print("aligned:", geometry_report(assignment("aligned")))
    for k in range(5): print(f"permuted seed {k}:", geometry_report(assignment("permuted", k)))
