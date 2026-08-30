"""Synthetic sanity tests for pf.fourier. Run: python -m tests.test_synthetic"""
import numpy as np
from pf.fourier import (mode_energies, paired_energies, paired_vector, parseval_ok, conjugate_symmetry_ok,
                        relabel_by_unit, mode_permutation_by_unit, circulant_projection, kernel_dft,
                        predicted_energy_from_M, permutation_null, matrix_abs)

N, d = 12, 64
rng = np.random.default_rng(1)
x = np.arange(N)


def embed(signal_matrix, d=d, rng=rng):
    """signal_matrix: (N, m) -> embed in R^d via random orthonormal directions."""
    m = signal_matrix.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    return signal_matrix @ Q.T


def report(name, H):
    E = mode_energies(H)
    P = paired_energies(E)
    ok_p, l, r = parseval_ok(H)
    ok_c = conjugate_symmetry_ok(H)
    keys = ["P1", "P2", "P3", "P4", "P5", "E6"]
    tot = P["total_nonconst"]
    prof = " ".join(f"{k}={P[k]/tot:.3f}" for k in keys) if tot > 0 else "(zero)"
    print(f"{name:42s} parseval={ok_p} conj={ok_c}  {prof}")
    return P


print("=== 1. chromatic fundamental: h_x = [cos(2pi x/12), sin(2pi x/12)]")
S = np.stack([np.cos(2 * np.pi * x / N), np.sin(2 * np.pi * x / N)], 1)
P = report("chromatic k=1", embed(S)); assert P["P1"] / P["total_nonconst"] > 0.999

print("=== 2. fifth fundamental: h_x = [cos(2pi 7x/12), sin(2pi 7x/12)]  (semitone coords)")
S5 = np.stack([np.cos(2 * np.pi * 7 * x / N), np.sin(2 * np.pi * 7 * x / N)], 1)
P = report("fifths k=7 (P5)", embed(S5)); assert P["P5"] / P["total_nonconst"] > 0.999

print("=== 3. relabeling a chromatic circle by x -> 7x gives a fifths circle, and vice versa")
H1 = embed(S)
H1r = relabel_by_unit(H1, 7)
P = report("chromatic relabeled by 7", H1r); assert P["P5"] / P["total_nonconst"] > 0.999
assert np.allclose(relabel_by_unit(H1r, 7), H1), "7 must be an involution"
perm = mode_permutation_by_unit(7)
print("   mode permutation under x->7x (source mode for each k):", perm)
assert perm == [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
# energies permute accordingly for a random H
Hr = rng.standard_normal((N, d))
E, Er = mode_energies(Hr), mode_energies(relabel_by_unit(Hr, 7))
assert np.allclose(Er, E[perm]), "E'_k = E_{7k}"
print("   energy permutation verified on random H; invariant pairs: P2,P3,P4,E6; swapped: P1<->P5")
for a in (5, 11):
    pa = mode_permutation_by_unit(a)
    print(f"   unit a={a}: perm={pa}")

print("=== 4. mixtures")
for w in (0.25, 0.5, 0.75):
    S_mix = np.concatenate([np.sqrt(1 - w) * S, np.sqrt(w) * S5], 1)
    P = report(f"mix w5={w}", embed(S_mix))
    assert abs(P["P5"] / P["total_nonconst"] - w) < 1e-9

print("=== 5. noise: isotropic Gaussian -> flat expected spectrum (each P_m ~ 2/11, E6 ~ 1/11)")
acc = np.zeros(6)
for _ in range(500):
    acc += paired_vector(mode_energies(rng.standard_normal((N, d))))
acc /= acc.sum()
print("   mean profile:", np.round(acc, 3), " expected:", np.round([2/11]*5 + [1/11], 3))
assert np.allclose(acc, [2/11]*5 + [1/11], atol=0.01)

print("=== 6. signal + noise, SNR sweep (chromatic)")
for snr in (10, 1, 0.1):
    H = embed(S) * np.sqrt(snr) + rng.standard_normal((N, d)) * np.sqrt(2 / d)  # noise energy ~2 total-ish
    report(f"chromatic + noise, snr~{snr}", H)

print("=== 7. non-circulant perturbation: one concept displaced")
H = embed(S); H2 = H.copy(); H2[3] += 3 * rng.standard_normal(d) / np.sqrt(d) * np.linalg.norm(H[3])
report("chromatic, concept 3 displaced", H2)

print("=== 8. random label permutations destroy structure (null)")
nullv = permutation_null(embed(S), n=2000)
prof = nullv / nullv.sum(1, keepdims=True)
print("   null mean profile of a pure k=1 circle after random relabel:", np.round(prof.mean(0), 3))
print("   null 95th pct of P1 share:", np.round(np.quantile(prof[:, 0], 0.95), 3), " observed (true labels): 1.000")

print("=== 9. circulant projection + kernel DFT + |M| prediction")
kappa_true = np.array([np.exp(-min(dd, N - dd) / 2.0) for dd in range(N)])
Mc = np.array([[kappa_true[(j - i) % N] for j in range(N)] for i in range(N)])
kappa, R, frac = circulant_projection(Mc)
assert np.allclose(kappa, kappa_true) and np.allclose(R, 0) and abs(frac - 1) < 1e-12
lam = kernel_dft(kappa).real
Epred = predicted_energy_from_M(Mc, use_abs=False)
assert np.allclose(Epred[1:], lam[1:]), "for circulant PSD M, E_k^pred == lambda_k"
print("   exponential kernel sigma=2: lambda_k =", np.round(lam, 3))
# a fifths-smooth kernel: g(7d)
kappa5 = kappa_true[[(7 * dd) % N for dd in range(N)]]
lam5 = kernel_dft(kappa5).real
assert np.allclose(lam5, lam[[(7 * k) % N for k in range(N)]]), "lambda^(5)_k = lambda^(g)_{7k}"
print("   fifths-transformed kernel: lambda_k =", np.round(lam5, 3), " (== lambda_{7k})")
# non-PSD circulant: negative eigenvalue -> |M| flips sign
kappa_np = kappa_true - 0.5
Mnp = np.array([[kappa_np[(j - i) % N] for j in range(N)] for i in range(N)])
lam_np = kernel_dft(kappa_np).real
Eabs = predicted_energy_from_M(Mnp, use_abs=True)
assert np.allclose(Eabs[1:], np.abs(lam_np[1:])), "|M| prediction gives |lambda_k| for k != 0"
print("   non-PSD kernel: lambda_k =", np.round(lam_np, 3), "; |M| prediction =", np.round(Eabs, 3))
# non-circulant M: projection residual nonzero
Mnc = Mc + 0.3 * rng.standard_normal((N, N)); Mnc = (Mnc + Mnc.T) / 2
kappa, R, frac = circulant_projection(Mnc)
print(f"   noisy M: circulant fraction of energy = {frac:.3f}")
# permutation of labels changes circulant fraction
p = rng.permutation(N); kp, Rp, fracp = circulant_projection(Mc[np.ix_(p, p)])
print(f"   circulant M under random relabel: circulant fraction = {fracp:.3f} (was 1.0)")
print("\nALL SYNTHETIC TESTS PASSED")
