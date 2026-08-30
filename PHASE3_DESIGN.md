# PHASE3_DESIGN.md — synthetic causal test of output-code geometry (written before any training run)

## Question
When the true predictive statistic over 15 observed states is perfectly periodic (a circle over 12 latent classes with
three duplicated labels), can an *open-line* geometry of the output codewords induce line-like behaviour or line-like
predictive representations? And when the statistic is an open line, does an arbitrary (permuted) code destroy the
recovered line? Factorial: {CIRCLE, LINE} × {LINE_ALIGNED, PERMUTED_CODE}, paired seeds.

## Latent space
Observed source states n ∈ {−7,…,+7} (15). Latent class z(n) = n mod 12 ∈ {0,…,11}; duplicates −7≡5, −6≡6, −5≡7.
No music vocabulary anywhere: tokens are `<S_-7>` … `<S_+7>`, a shared query token `<Q>`, and code symbols `c0 c1 c2`.

## Oracle laws (synthetic/laws.py)
CIRCLE: over latent classes P_z(z'|z) ∝ exp[β cos(2π(z'−z)/12)], normalized over the 12 classes. Observed target label
m receives P(m|n) = P_z(z(m)|z(n)) / mult(z(m)), mult = 2 for the three duplicated classes, else 1. Properties (unit-
tested): rows of twin sources identical; twin targets receive equal mass; the class-marginal is exactly periodic; the
conditional depends on n only through z(n); Σ_m P(m|n) = 1.
LINE: P(m|n) ∝ exp(−|m−n|/τ) over the 15 observed labels (open; twin sources have different rows).
β and τ are chosen so the mean row entropy (over the 15 sources) matches to < 0.02 nats (β = 2.0 fixed; τ solved).
Both matrices are saved to results/phase3/oracles.npz.

## Output codewords (synthetic/codes.py)
One fixed set of 15 codewords of length L = 8 over the symbol alphabet {c0, c1, c2}: window code
codeword_i[p] = floor((i + p) / 8), i = 0…14, p = 0…7. Hamming distance between codewords i, j is exactly min(|i−j|, 8):
monotone in index distance up to saturation, identical for all |i−j| ≥ 8. Unambiguous boundaries (fixed length),
fixed vocabulary, fixed token inventory. Assignments:
- LINE_ALIGNED: state n ↦ codeword index n + 7 (code distance aligned with semantic |n − m|).
- PERMUTED_CODE(k): state n ↦ codeword π_k(n + 7) for a random permutation π_k with |Spearman(code distance,
  |n − m|)| < 0.15 (several k; permutation k is paired with training seed k).
The codeword inventory, lengths and symbol frequencies are identical across assignments by construction; only the
semantic↔lexical alignment changes. Code geometry is quantified before training (Hamming, Levenshtein, prefix overlap;
Spearman with |n − m|, with circle distance, and with the duplicate-pair indicator).

## Data
Each example: `<S_n> <Q> code(m)_1 … code(m)_8`, n uniform over 15, m ~ P(m|n) from the oracle. 300k training examples,
20k validation, per condition and seed (seed-paired across conditions: seed k draws the same n sequence and the same
uniform variates for m, so CIRCLE/LINE and ALIGNED/PERMUTED differ only in the law / code assignment).
Loss: cross-entropy on the 8 code positions only.

## Model (synthetic/model.py)
Decoder-only transformer from scratch: 4 layers, d_model 128, 4 heads, MLP 4×, learned positional embeddings,
context 10. AdamW lr 1e-3 (cosine), batch 256, 6000 steps. Seeds 0–4 per condition; checkpoints every 250 steps.

## Measurements (synthetic/measure.py), at every checkpoint
Behaviour: s_nm = log P(code(m) | `<S_n> <Q>`) by teacher forcing over the full 8-token codeword (equal lengths →
plain sequence log-prob is the primary scorer); q(m|n) = softmax_m s_nm; mapped to observed-state space (m is already a
state); compared with the oracle: mean row KL(q‖P), RSA (Spearman of −log q vs −log P over 210 ordered off-diagonal
pairs), partial Spearman circle|line and line|circle of the symmetrized −log q (controls: code Hamming distance),
duplicate-source divergence JS(q(·|n), q(·|n′)) for the 3 twin source pairs, duplicate-target asymmetry
mean |log q(m|n) − log q(m′|n)| over twin targets, and the code-geometry alignment RSA(−log q, Hamming).
Hidden geometry: residual at the shared `<Q>` position for the 15 sources, every layer: distance matrix → RSA with
circle, line, code geometry; partials circle|line, line|circle (control code distance); duplicate-collapse index ECI
(mean percentile rank of the 3 twin distances; null 0.5).
Nulls: 1000 free relabelings of the 15 states for each partial/ECI at the final checkpoint; paired-seed differences
(ALIGNED − PERMUTED) tested by sign across seeds.

## Pre-registered readings
- OUTPUT CODE SHAPES BEHAVIOR: CIRCLE×ALIGNED behaviour has higher line|circle / lower twin collapse than
  CIRCLE×PERMUTED at matched KL.
- OUTPUT CODE SHAPES REPRESENTATION ONLY: same for the `<Q>` hidden geometry but not behaviour.
- TRANSIENT LEXICAL BIAS: the difference exists early in training and vanishes at convergence.
- PREDICTIVE STATISTICS DOMINATE: no aligned/permuted difference in either measure; LINE×PERMUTED recovers the line
  (positive control).
- REPRESENTATION / BEHAVIOR DISSOCIATION: hidden geometry follows the code while behaviour follows the oracle.
- SYNTHETIC DESIGN FAILED: models do not reach low KL to the oracle.
Dose-response (ρ ≈ 0, .25, .5, .75, 1) only if the binary contrast is non-null.
