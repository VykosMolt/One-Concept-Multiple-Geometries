# PHASE4_DESIGN.md — sparse source-alias equivalence (written before any Phase-IV run)

## Question
When two surface labels (aliases) have exactly the same latent predictive state but one is sparsely observed as a
SOURCE, how much exposure to the rare alias does a transformer need before it treats the two as equivalent, and does a
line-aligned output code delay that collapse? Separate three things that a naive "make duplicates rare" design confounds:
rarity of the surface alias, rarity of the latent state, and failure to learn the latent quotient.

## Process (synthetic/phase4.py)
Latent classes z ∈ {0..11} sampled uniformly (class mass 1/12, constant across all conditions). Class → source label:
unique classes emit their single label; duplicated classes (z=5: {−7,+5}, z=6: {−6,+6}, z=7: {−5,+7}) emit the RARE alias
(−7, −6, −5) with probability r and the COMMON alias (+5, +6, +7) with probability 1−r. Targets: the Phase-III CIRCLE law
exactly (β = 2, class-level periodic, target aliases split 0.5/0.5) — the target distribution does not change with r.
Oracle rows of twin sources are identical for every r. Data are sampled online (fresh batch each step, seeded), so
cumulative rare-alias exposure = number of rare-alias source draws so far; validation sets are fixed per label.
Rarity schedule: r ∈ {0.5, 0.1, 0.03, 0.01, 0.003, 0.001}. Natural calibration (Wikipedia, Phase II counts):
Cb:B r = 0.024, C#:Db r = 0.22, Gb:F# r = 0.41 — the sweep spans 20× below and above the natural range; absolute
rare exposures at 12k steps × 256 × 3r/12: 384k (r = .5) … 768 (r = .001).
Equally-rare UNIQUE-state control: a separate condition in which the unique class z = 0 (label 0) has its source
frequency reduced to r/12 (freed mass spread uniformly over the other classes) while aliases stay balanced (r = 0.5);
its oracle-row KL at matched exposure is compared with the rare alias's equivalence error.
Codes: LINE_ALIGNED and PERMUTED (permutation k paired with seed k), as in Phase III. Model/optimizer as in Phase III.

## Runs
Primary: 6 rarity levels × 2 codes × 5 paired seeds, 12,000 steps (2× Phase III) = 60 runs.
Control: 6 rarity levels (unique-state rarity) × 2 codes × 5 seeds, 12,000 steps = 60 runs.
Extended: r ∈ {0.003, 0.001} × 2 codes × 5 seeds continued to 48,000 steps (to separate "frozen" from "slow").
Checkpoints at ≈ 30 geometrically spaced steps; each records steps, cumulative rare-source exposures (actual count), and:
 - behaviour q(m|n) for all 15 sources (teacher-forced full-codeword scoring);
 - KL to oracle: global (all 15 rows), common-alias rows, rare-alias rows, unique-control row;
 - PRIMARY: latent-class twin JS — JS( q_z(·|a), q_z(·|b) ) after merging target aliases into 12 classes; also 15-way JS;
 - hidden state at <Q>: normalized Euclidean and cosine distance between twin aliases (relative to the mean non-twin
   distance), ECI, circle|line, line|circle, code RSA.
Equivalence tolerance: chosen from the balanced runs (r = 0.5) — tol = 3 × the 95th percentile of their final latent
JS — fixed before looking at sparse runs. N_equiv(r) = cumulative rare exposures at the first checkpoint whose latent
JS < tol (and stays below for the remaining checkpoints).

## Pre-registered readings
SPARSE EQUIVALENCE FREEZES: latent JS stays ≫ tol for r ≤ 0.01 while global KL is at the Phase-III floor and the
common aliases are solved, over the full (extended) budget. SAMPLE-LIMITED ONLY: JS curves at different r collapse onto
one curve of cumulative rare exposure and converge with a roughly fixed number of exposures. CODE DELAYS EQUIVALENCE:
N_equiv(aligned) > N_equiv(permuted) consistently across r and seeds. RARITY, NOT SYMMETRY: the equally rare unique state
converges as slowly as the rare alias (same exposure requirement). REPRESENTATION LAGS BEHAVIOR: behavioural
equivalence precedes hidden-state collapse. FROZEN-TRANSIENT HYPOTHESIS FAILS: sparse aliases converge easily at
natural-range r, or asymmetry appears only when the alias is essentially unseen.
Target-rarity experiment (§12 of the brief) only after the source-rarity result is understood.
