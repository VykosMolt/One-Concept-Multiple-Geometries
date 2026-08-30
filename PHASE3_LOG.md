# PHASE3_LOG.md — chronological (machine clock)

Sat Aug 29 11:05:03 AM CEST 2026 — Phase III start. Design in PHASE3_DESIGN.md (written before any run). Oracle: β = 2.0, τ = 2.2455 (mean row
entropy 2.1368 nats both). Codewords: 15 × length-8 window code over 3 symbols, Hamming = min(|i−j|, 8);
aligned ρ(Hamming, |n−m|) = 0.99, ρ(Hamming, circle) = 0.30; permuted seeds 0–4: |ρ| ≤ 0.12, identical symbol counts.
Unit tests (tests/test_synthetic_phase3.py) pass: periodicity, twin equality, no open-line coordinate in CIRCLE,
line monotonicity, entropy match, code lengths/geometry, permutation preservation, scorer, remapping, oracle KL,
seed pairing. Oracle self-geometry (behaviour_stats on the oracle itself): see test output.

11:06 Smoke run (CIRCLE×ALIGNED, seed 0, 1500 steps, 14 s): KL to oracle 3.31 → 0.0012; behaviour circle|line
+0.97, line|circle transiently +0.08…+0.18 (steps 400–1000) then +0.10; twin-target asymmetry 0.9 → 0.11 (slow);
Q-hidden last layer circle|line +0.77, ECI 0.02 (twin sources collapse). Task is learnable; full factorial launched:
{circle,line} × {aligned,permuted} × seeds 0–4, 6000 steps, paired seeds (permutation k paired with seed k).

11:23 FACTORIAL RESULTS (20 runs; results/phase3/analysis_main.txt, analysis_uncontrolled.txt; figures/phase3/trajectories_main.png)
Convergence: all runs KL(q‖oracle) ≤ 0.001, RSA with oracle ≥ 0.989 → no design failure.
Behaviour at convergence: circle|line 0.98 (CIRCLE), line|circle 0.995 (LINE) in both code conditions; uncontrolled
line|circle equals the oracle's own value (CIRCLE: aligned 0.436, permuted 0.422, oracle 0.439; LINE: 0.995 all);
twin-target asymmetry 0.10 vs 0.09 (Δ +0.013, 5/5, t p .044). No lasting behavioural effect of the output code.
Trajectory (CIRCLE law, aligned − permuted, paired): at steps 50/100/150 the aligned models are closer to the oracle
(KL 0.078 vs 0.276 at step 100) and already at the oracle's line|circle (0.44/0.38 vs 0.21/0.11; Δ +0.23…+0.27, 5/5);
twin-target asymmetry is higher under aligned code through ≈ step 1500 (+0.1…+0.27); all differences ≈ 0 by step 500–2000.
=> aligned code accelerates learning of the smooth neighbourhood structure (shared tokens generalize) and delays the
equalization of maximally-different twin codewords; it never pushes behaviour beyond the oracle's own line component.
Hidden geometry at <Q> (CIRCLE law): circle|line ≈ +0.6 in both; line|circle −0.14 (aligned) vs −0.23 (permuted) at mid
and last layers, i.e. a +0.08…+0.10 shift toward the line under aligned code, 5/5 seeds (t p .001 mid, .014 last),
persistent from step ≈ 500 to the end; RSA with the code +0.18 vs +0.05. Neither condition's line|circle exceeds its
relabeling null (q95 ≈ +0.17…+0.21): the code shifts the predictive-state geometry but does not create a line.
Twin-source collapse (ECI 0.02) identical in both.
Positive control (LINE×PERMUTED): behaviour recovers the line exactly; <Q> hidden line|circle +0.46 (last) / +0.41
(mid) with an arbitrary code. Oddity: LINE×PERMUTED hidden circle|line +0.32 (mid), +0.23 (last) vs −0.07 / +0.09 under
the aligned code, and twin ECI 0.66 vs 0.93 — a periodic-looking component in the representation of an open law when
the code is arbitrary; hypothesis: a horseshoe (open-boundary) embedding of the line, which the circle|line partial
reads as periodic. To be checked on the saved models.
Dose-response launched (CIRCLE law; ρ ≈ 0, .25, .5, .75, 1; 3 seeds).

11:24 PCA of <Q> hidden states (final models, mean over 5 seeds, layers 2/4):
LINE×ALIGNED: |ρ(PC1,n)| 0.99/0.97, |ρ(PC2,(n−n̄)²)| 0.96/0.96, variance 0.88/0.09 → clean horseshoe (Karkada Prop. 3
open-boundary embedding). LINE×PERMUTED: 0.47/0.43 and 0.61/0.52 → the same law is embedded far less linearly when the
code is arbitrary; its positive circle|line partial is the signature of a bent/irregular open embedding, not periodicity.
CIRCLE×ALIGNED: PC1 not monotone (0.33/0.36) but PC2 quadratic in n (0.85/0.77); CIRCLE×PERMUTED: 0.39/0.33 → the
aligned output code imprints a bow along the line onto the representation of a perfectly periodic law, while behaviour
stays exactly periodic. This is the cleanest statement of the representational effect.

11:27 Follow-up queued (bounded, 6 runs): CIRCLE law with rare alternative spellings — class mass split 0.9/0.1
between primary (+5,+6,+7) and alternative (−7,−6,−5) spellings (oracle twin-target log-ratio 2.197; still periodic at
the class level; twin sources still identical rows). Question: does the aligned code leave a larger *excess* twin
asymmetry (model minus oracle) than the permuted code when twin evidence is sparse, and does it persist? 3 paired seeds.

11:28 Bug: while adding the rare law I appended a comment to the argv line of train.py, which commented out the
'steps'/'tag' assignments; dose runs from ρ≈.25 seed 1 onward and all rare runs crashed at start (no results affected —
the four finished dose runs predate the edit). Fixed; missing runs relaunched.

11:41 Dose-response (15 runs, ρ ≈ −0.03/.23/.46/.71/1.0): early behaviour line|circle vs ρ Spearman +0.85, KL@100
−0.68, PC2 bow +0.57, hidden l|c shift +0.29 mid / −0.09 last, final KL flat. Rare-twin follow-up: excess twin asymmetry
converges to ≈0 under both codes by step 6000 (KL 0.0006); hidden l|c shift +0.06…+0.09 persists (3/3). Phase III closed:
the stop-condition question is answered — output-code geometry alters predictive representation slightly and behaviour
transiently; it cannot make converged behaviour line-like when the statistics are periodic.
