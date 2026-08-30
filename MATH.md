# MATH.md — definitions, verified identities, and what the theory actually predicts

All statements marked **[verified]** were checked numerically in `tests/test_synthetic.py`
(run log in `RESEARCH_LOG.md`). Statements marked **[theory, cited]** are quoted from the
literature, with the section reference.

## 1. Coordinates on Z_12

Pitch classes are indexed by semitone x ∈ Z_12, C=0, Db/C#=1, D=2, Eb/D#=3, E=4, F=5, F#/Gb=6,
G=7, Ab/G#=8, A=9, Bb/A#=10, B=11.

* Chromatic motion: x → x+1. Ascending fifths: x → x+7.
* 7² = 49 ≡ 1 (mod 12), so r(x) = 7x mod 12 is an involutive automorphism of Z_12, and
  r(x+7) = 7x + 49 = r(x) + 1 (mod 12). **[verified]** (r is checked to be an involution; the
  fifths-position of C,G,D,A,E,B,F#,Db,... is 0,1,2,...,7,... — e.g. r(7)=1, r(2)=2, r(9)=3.)
* The units of Z_12 are {1,5,7,11} ≅ Z_2×Z_2. These are exactly the cyclic orderings of the 12
  pitch classes that are group automorphisms: chromatic up (1), fourths up (5), fifths up (7),
  chromatic down (11). Note 5 ≡ −7: the "circle of fourths" is the circle of fifths reversed.

## 2. Circulant kernels do not distinguish the orderings

A fifths-smooth kernel in semitone coordinates is K_5(x,y) = g(7(y−x) mod 12), which is
circulant in x. **[verified]** So circulant structure of a 12×12 matrix tells you nothing about
*which* generator is smooth; only the Fourier index does.

## 3. Fourier modes and the automorphism

Characters χ_k(x) = exp(2πikx/12). For a representation H ∈ R^{12×d} (rows = concepts in
semitone order, mean-centered), define

    hhat_k = (1/√12) Σ_x h_x exp(−2πikx/12),      E_k = ||hhat_k||².

* Parseval: Σ_k E_k = ||H_c||²_F. **[verified]**
* Real H ⇒ hhat_{12−k} = conj(hhat_k) ⇒ E_k = E_{12−k}. **[verified]**
* Paired energies P_m = E_m + E_{12−m} (m=1..5), E_6 alone, E_0 = 0 after centering.
* If H'[x] = H[7x mod 12], then hhat'_k = hhat_{7k mod 12} (since 7⁻¹ = 7), so
  E'_k = E_{7k}. **[verified on random H]** The induced permutation on modes is
  k: 0 1 2 3 4 5 6 7 8 9 10 11 → 0 7 2 9 4 11 6 1 8 3 10 5.
  Consequences:
  - P1 ↔ P5 are swapped (chromatic fundamental ↔ fifths fundamental). **[verified]**
  - P2, P4, E6 are *fixed* (2·7=14≡2, 4·7=28≡4, 6·7=42≡6), and P3 maps to itself as a pair
    (3→9, 9→3). So **the only harmonic pair that discriminates chromatic from fifths order is
    P1 vs P5**; the remaining bins are invariant under the relabeling and must be compared
    *directly* between corpus and model regardless of which ordering is "right".
  - The unit 5 induces the same P-pair swap (1→5, 11→7), and 11 is complex conjugation
    (k → −k), which fixes every P_m. So all four automorphic orderings give the same
    paired spectrum up to the single swap P1↔P5.
* A pure fifths-fundamental h_x = [cos(2π·7x/12), sin(2π·7x/12)] has all its energy in P5.
  **[verified]**

## 4. "Not a torus"

Simultaneous P1 and P5 energy is a single 12-point orbit of one cyclic variable x embedded
via two harmonics, h_x = A·e(x) + B·e(7x) with e(θ) = (cos 2πθ/12, sin 2πθ/12): a closed
curve sampled at 12 points, living in a 4-dimensional subspace. It is not a 2-torus, because
there is no independent second phase; the two "angles" are 2πx/12 and 2π·7x/12, both
functions of x. Mixture tests **[verified]**: with h_x = √(1−w)e(x) ⊕ √w e(7x), P5/(P1+P5) = w
exactly.

## 5. Corpus statistic (Karkada et al. 2026, Eq. 1, App. A.2, B.1) **[theory, cited]**

    M*_ij = (P_ij − P_i P_j) / (½(P_ij + P_i P_j)) = 2 (ρ−1)/(ρ+1),   ρ = P_ij/(P_i P_j),
    −2 ≤ M*_ij ≤ 2,  M* ≈ log ρ = PMI for ρ near 1.

Karkada's P_ij (Eq. 19): symmetric skip-gram window, L = 16, distance weighting
f(d) = L+1−d, Wikipedia 20231101.en, V = 25,000 most frequent words. We reproduce this
construction restricted to the concept set S plus the normalizer Z, computing
ρ_ij = (C_ij / Z) / ((n_i/N)(n_j/N)) with C_ij the f-weighted symmetric pair count, Z the
f-weighted total number of (position, offset) pairs, N the number of word tokens.

Circulant projection: κ(d) = (1/12) Σ_i M*_{i,i+d}; residual R = M* − circ(κ);
λ_k = Σ_d κ(d) e^{−2πikd/12} (real, since κ(d) = κ(−d)). **[verified]** For a fifths-smooth
kernel κ_5(d) = g(7d): λ^{(5)}_k = λ^{(g)}_{7k}. **[verified]**

## 6. What the theory predicts for representation energies **[theory, cited]**

Karkada App. B.1–B.2: word2vec-like training ≈ argmin ||W W'^T − M*||_F. The *word* embedding
Gram matrix is W W^T = |M*| := M⁺ + M⁻ (matrix absolute value, Eq. 32–33), and the
PCA-projected centered embeddings of S satisfy W̄_S W̄_S^T = P |M*|_S P (Eq. 58), where
|M*|_S is the S-submatrix of the *full-vocabulary* matrix absolute value.

Under translation symmetry (Assumption 3.1 or B.1), the Fourier vectors diagonalize the
S-block, so the energy in mode k of the centered representations equals |m̃(k)| —
i.e. **E_k = |λ_k|, the absolute value of the kernel's k-th Fourier coefficient** (Prop. 1
/ App. E.2, Eq. 57). **[verified]** that for circulant M the general prediction
E_k^pred = f_k^H (P |M| P) f_k reduces to |λ_k| for k ≠ 0.

Caveats that Karkada state themselves and that matter here:
1. |M*|_S ≠ |M*_S| in general (the matrix absolute value does not commute with taking the
   S-block). They fit kernels to M*_S anyway and report that **relative amplitudes and
   wavenumbers are predicted, not absolute scale** (App. B.3). We therefore compare *normalized*
   spectra. Under Assumption B.1 (M⁺ and M⁻ separately circulant on S) the predicted energy is
   C̃⁺(k)+C̃⁻(k) ≥ |λ_k|, so |λ_k| is strictly a lower bound on what the block-level prediction
   says; the discrepancy is unknowable from S alone.
2. The theory is proved for embedding models. For LLM residual streams Karkada only show
   qualitative carry-over (PCA pictures for Gemma-2-2B, App. A.3: prompt "The month of the
   year is x", final-token residual, all layers). There is no theorem for transformers, and the
   paper's Limitations section explicitly says contextual sharpening across layers is
   unexplained. **So any corpus→LLM comparison here is a test of a qualitative conjecture,
   not of a theorem.**
3. Karkada's Fig. 4 shows the month circle can be reconstructed with the month–month block
   ablated (helper words carry it). So the 12×12 block spectrum is a *sufficient-statistic
   proxy* the theory itself says may be bypassed by the rest of M*. We record this as a known
   limitation of the block-only prediction and test helper-word variants where feasible.

## 7. Nulls

Under a uniformly random relabeling of the 12 concepts, E[E_k] = ||H_c||²/11 for every
k=1..11 (the permutation-averaged Gram is a I + b 11^T). So the null paired profile is
(2/11,2/11,2/11,2/11,2/11,1/11). **[verified: 500-sample mean matched to ±0.01]**
Concentration in any bin is tested against the empirical permutation distribution of that
bin (12! relabelings sampled).

## 8. Diagnostic warning (Convergent-Evolution paper, arXiv:2604.20817)

A spectral spike at a period does not by itself imply the mode is *linearly usable*; spectral
energy and geometric separability can dissociate. We treat E_k as a description of geometry,
not of computation (see brief; Feucht et al. 2026).

## 9. Circle vs line of fifths (added after the results; see RESEARCH_LOG 07:15)

Fifths position p(x) = 7x mod 12 (C=0, G=1, …, F#=6, Db=7, …, F=11). Signed accidental count
s(x) = p(x) if p(x) ≤ 6 else p(x) − 12 (F#=+6, Db=−5, F=−1). Circle distance
d_c = min(|Δp| mod 12, 12 − |Δp| mod 12); line distance d_l = |Δs|. They coincide except on the 15
unordered pairs whose short arc crosses the F#/Db seam, where d_l = 12 − d_c. In Karkada's terms
the circle is a periodic 1-D lattice (Cor. 2: integer-frequency sinusoidal PCs) and the line is an
open 1-D lattice (Prop. 3: quasi-sinusoids with non-integer wavenumbers; PC1 monotone in s).
Diagnostics used: partial Spearman of a similarity matrix with −d_c controlling for −d_l (and vice
versa), the seam-pair-only correlation, and |ρ(PC1, s)| vs |ρ(PC1, cos 2πp/12)|.
Note that the semitone-coordinate DFT (P5 bin) cannot distinguish the two: a line of 12 points
with an exponential kernel also puts most of its energy in the fifths fundamental.

## 10. Corrections after independent review (2026-08-29)

- Projecting out the centered black-key indicator removes **47.9% of the P5 *energy*** of a noise-free fifths circle
  (the indicator lies 79.6% inside the k=5/7 subspace, so a rank-1 removal takes about half of it); the earlier "13.5%"
  figure was the drop in P5 *share* (1.000 → 0.865). A circle still survives the projection as a share, but the
  projection is not mild.
- The relabeling null for the *projected* P5 share is ≈0.120, not 2/11 = 0.18: removing a direction that carries 79.6%
  of its energy in P5 lowers the null's P5 share. Observed projected P5 values of 0.11–0.14 are therefore *at* the
  projection-matched null (z ≈ +0.4), not below it. `scripts/confounds.py` and `multictx_analyze.py` always used the
  matched null (projection applied inside the permutation); the prose in RESULTS/RESEARCH_LOG had used 0.18.
- §4 ("not a torus"): the statement concerns a single 12-point orbit of one cyclic variable and is unrelated to the
  Krumhansl–Kessler key *torus* (circle of fifths × relative/parallel mode), which is a two-variable object over 24 keys.
- Partial Spearman correlations on 66 pairs need their own nulls; the free relabeling null and a block-preserving null
  (permute within black and within white keys) agree closely here (VIFs ≤ 1.4 for the control design), and best-over-
  layers values need a max-over-layers null (see `scripts/partial_nulls.py`).
