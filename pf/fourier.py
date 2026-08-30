"""Concept-axis Fourier instrumentation for N-point cyclic concept families.

Conventions (semitone coordinates for pitch classes):
    x in Z_N, N = 12, C=0, Db/C#=1, ..., B=11.
    hhat_k = (1/sqrt N) sum_x h_x exp(-2 pi i k x / N)      (numpy fft sign convention)
    E_k    = ||hhat_k||^2
Paired energies (real inputs => E_k = E_{N-k}):
    P_m = E_m + E_{N-m}, m = 1..N/2-1;  E_{N/2} separately;  E_0 = 0 after centering.
"""
from __future__ import annotations
import numpy as np

UNITS_12 = (1, 5, 7, 11)  # multiplicative units of Z_12: the cyclic orderings that are group automorphisms


def center(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    return H - H.mean(axis=0, keepdims=True)


def concept_dft(H: np.ndarray, do_center: bool = True) -> np.ndarray:
    """H: (N, d) rows ordered by concept coordinate x=0..N-1. Returns hhat (N, d) complex."""
    Hc = center(H) if do_center else np.asarray(H, dtype=np.float64)
    N = Hc.shape[0]
    return np.fft.fft(Hc, axis=0) / np.sqrt(N)


def mode_energies(H: np.ndarray, do_center: bool = True) -> np.ndarray:
    F = concept_dft(H, do_center)
    return (np.abs(F) ** 2).sum(axis=1)


def paired_energies(E: np.ndarray) -> dict:
    N = len(E)
    out = {}
    for m in range(1, N // 2):
        out[f"P{m}"] = float(E[m] + E[N - m])
    if N % 2 == 0:
        out[f"E{N//2}"] = float(E[N // 2])
    out["E0"] = float(E[0])
    out["total_nonconst"] = float(E[1:].sum())
    return out


def paired_vector(E: np.ndarray) -> np.ndarray:
    """[P1, ..., P_{N/2-1}, E_{N/2}] as a vector (N even)."""
    N = len(E)
    v = [E[m] + E[N - m] for m in range(1, N // 2)]
    v.append(E[N // 2])
    return np.asarray(v, dtype=np.float64)


def normalized_profile(E: np.ndarray) -> np.ndarray:
    v = paired_vector(E)
    s = v.sum()
    return v / s if s > 0 else v


def parseval_ok(H: np.ndarray, do_center: bool = True, rtol: float = 1e-9) -> tuple[bool, float, float]:
    Hc = center(H) if do_center else np.asarray(H, dtype=np.float64)
    lhs = float((Hc ** 2).sum())
    rhs = float(mode_energies(H, do_center).sum())
    return bool(np.isclose(lhs, rhs, rtol=rtol)), lhs, rhs


def conjugate_symmetry_ok(H: np.ndarray, rtol: float = 1e-9) -> bool:
    F = concept_dft(H)
    N = F.shape[0]
    return bool(np.allclose(F[1:], np.conj(F[N - 1:0:-1]), rtol=rtol, atol=1e-12))


# ---------------------------------------------------------------- automorphisms

def relabel_by_unit(H: np.ndarray, a: int) -> np.ndarray:
    """Return H' with H'[x] = H[(a*x) mod N]. For a=7, N=12 this maps a representation that is
    a smooth function of *fifths* position into semitone order (and vice versa, since 7*7=1)."""
    N = H.shape[0]
    idx = [(a * x) % N for x in range(N)]
    assert sorted(idx) == list(range(N)), f"{a} is not a unit mod {N}"
    return np.asarray(H)[idx]


def mode_permutation_by_unit(a: int, N: int = 12) -> list[int]:
    """If H'[x] = H[a x], then hhat'_k = hhat_{a^{-1} k}. Returns for each k the source mode index."""
    ainv = pow(a, -1, N)
    return [(ainv * k) % N for k in range(N)]


# ---------------------------------------------------------------- circulant analysis

def circulant_projection(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """kappa(d) = (1/N) sum_i M[i, (i+d) mod N]; residual = M - circ(kappa);
    returns kappa, residual, and fraction of (centered) Frobenius energy explained by circulant part."""
    M = np.asarray(M, dtype=np.float64)
    N = M.shape[0]
    kappa = np.array([np.mean([M[i, (i + d) % N] for i in range(N)]) for d in range(N)])
    C = np.array([[kappa[(j - i) % N] for j in range(N)] for i in range(N)])
    R = M - C
    denom = float((M ** 2).sum())
    frac = 1.0 - float((R ** 2).sum()) / denom if denom > 0 else float("nan")
    return kappa, R, frac


def kernel_dft(kappa: np.ndarray) -> np.ndarray:
    """lambda_k = sum_d kappa(d) exp(-2 pi i k d / N). Real when kappa(d)=kappa(-d)."""
    return np.fft.fft(np.asarray(kappa, dtype=np.float64))


def circulant_eigs(M: np.ndarray) -> np.ndarray:
    """Eigenvalues of the circulant projection of M, indexed by Fourier mode k (real parts)."""
    kappa, _, _ = circulant_projection(M)
    lam = kernel_dft(kappa)
    return lam.real


def matrix_abs(M: np.ndarray) -> np.ndarray:
    """|M| := M+ + M- (matrix absolute value) for symmetric M."""
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.abs(w)) @ V.T


def predicted_energy_from_M(M: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """Karkada-style prediction: centered Gram of representations = P |M|_S P (B.2/B.3, up to scale).
    For circulant M, Fourier modes are eigenvectors, so predicted E_k = |lambda_k| (k != 0).
    Here we compute it generally: E_k^pred = (1/N) f_k^H (P |M| P) f_k with f_k the DFT vector."""
    M = np.asarray(M, dtype=np.float64)
    N = M.shape[0]
    Pc = np.eye(N) - np.ones((N, N)) / N
    A = matrix_abs(M) if use_abs else M
    G = Pc @ A @ Pc
    E = np.zeros(N)
    for k in range(N):
        f = np.exp(-2j * np.pi * k * np.arange(N) / N) / np.sqrt(N)
        E[k] = float(np.real(np.conj(f) @ G @ f))
    return E


# ---------------------------------------------------------------- comparisons / nulls

def permutation_null(H: np.ndarray, n: int = 2000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Distribution of paired-profile vectors under random relabeling of the N concepts.
    Returns (n, N/2) array of paired energy vectors."""
    rng = rng or np.random.default_rng(0)
    N = H.shape[0]
    out = []
    for _ in range(n):
        p = rng.permutation(N)
        out.append(paired_vector(mode_energies(H[p])))
    return np.asarray(out)


def spectrum_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


# ---------------------------------------------------------------- confound diagnostics (added 2026-08-29 02:10)

def project_out(H: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove from H (N×d) the component along the centered indicator v (N,): H - v v^T H / (v^T v)."""
    Hc = center(H)
    v = np.asarray(v, float); v = v - v.mean()
    if np.allclose(v, 0): return Hc
    return Hc - np.outer(v, v @ Hc) / (v @ v)


def mode_isotropy(H: np.ndarray, k: int) -> tuple[float, float, float]:
    """Singular values (s1>=s2) of the 2×d matrix [Re hhat_k; Im hhat_k]; returns (s1, s2, s2/s1)."""
    F = concept_dft(H)
    A = np.stack([F[k].real, F[k].imag])
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0]), float(s[1]), float(s[1] / s[0]) if s[0] > 0 else float("nan")


def rsa_line(H: np.ndarray, order: list[int]) -> float:
    """Spearman correlation between representational distances of the rows of H and a 1-D line
    distance along `order` (list of row indices in line order)."""
    from scipy.stats import spearmanr
    Hc = center(H)
    idx = np.asarray(order)
    X = Hc[idx]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    n = len(idx)
    L = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    iu = np.triu_indices(n, 1)
    return float(spearmanr(D[iu], L[iu]).correlation)
