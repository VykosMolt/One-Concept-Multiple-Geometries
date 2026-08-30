# Which corpus operator does each theory factorize? (Phase V reading note)

**Karkada et al. (2602.15029)** — word2vec-like models (symmetric skip-gram window, L=16, f(d)=L+1−d) minimize
||W W'^T − M*||_F with M*_ij = (P_ij − P_iP_j)/(½(P_ij + P_iP_j)) ≈ PMI; word-embedding Gram = |M*|. The object is a
SYMMETRIC word–word association (co-occurrence within a window in either direction, normalized by unigram products).
Predicts: Fourier geometry when M* is translation-symmetric on a lattice; magnitudes matter (|λ_k|).

**Zhao, Behnia, Vakilian, Thrampoulidis (2408.15417)** — NTP with unconstrained features: min CE(WH) + λ/2(||W||²+||H||²)
over contexts j with sparse conditional vectors p̂_j (counts of next token z after context j / occurrences of j) and
support sets S_j. As λ→0 the logits split into L_in (sparse, frequency-dependent, PMI-like) + R(λ)·L_mm, where L_mm is the
max-margin / nuclear-norm component that depends ONLY on the support pattern S ∈ {0,1}^{V×m}. Word and context Grams
converge in direction to the SVD factors of L_mm (2505.08348: to those of the centered support matrix
S̃ = (I − 11^T/V) S). Subspace collapse: contexts with identical next-token supports converge to the same direction
regardless of the conditional magnitudes. The object is a DIRECTIONAL context → next-token matrix, and in the
dominant component only its zero/non-zero pattern.

**Consequences for keys (worked out, not assumed):**
1. Under a symmetric PMI operator, enharmonic twins (Cb/B) are closest pairs in Wikipedia because they are mentioned
   together (explicit equivalence prose): periodic identity at the seam (Phase II/IV).
2. Under a directional conditional operator, the row of "Cb major" (what follows it) is not the row of "B major"
   (Phase IV: latent JS 0.228): the contexts differ, so no subspace collapse is predicted for the two spellings, and the
   line-of-fifths/spelling-side organisation of the rows is what NTP geometry would inherit.
3. In the Zhao model the frequency-magnitude part (L_in) is the PMI-like piece; the max-margin part uses supports.
   For a 15-key vocabulary the supports are almost never exactly zero at Wikipedia scale, so the distinction between
   "support pattern" and "conditional probabilities" is weak here; we use smoothed conditionals (log C) as the
   directional operator and note that a strict support-pattern operator would be degenerate (all-ones) at this scale.
4. These are different operators on the same text, and they legitimately predict different geometries for the same
   15 labels: symmetric → neutral-pitch-class periodicity (Karkada-type), directional → tonal-spelling line
   (NTP-type). Whether the model's geometry follows the second is an empirical question (Phase V); it is not implied
   by either theory, since neither is proved for transformer residual streams.
