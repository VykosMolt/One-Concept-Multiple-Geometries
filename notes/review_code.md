# Independent code & instrument verification (Opus reviewer 1, 2026-08-29) — verbatim summary of findings

Environment note: GPU busy at review time; item-5 re-extraction on CPU (same model/dtype). No repo file modified.

Item 1 (fourier.py / synthetic tests): PASS. DFT convention, Parseval, mode permutation verified for all four units
{1,5,7,11}; circulant_projection is the orthogonal projection; matrix_abs, predicted_energy_from_M (= |lambda_k| for
circulant M), project_out (a genuine projection that also centers), mode_isotropy, rsa_line, permutation_null (projection
applied inside the null in confounds.py / multictx_analyze.py) all correct. Docstring nits: "(centered) Frobenius energy",
spurious 1/N.

Item 2 (corpus counting): PASS on arithmetic — independent brute-force P_ij implementation matches process_doc exactly
(self-pairs, both directions, Z = 2 N sum f). Sentence-initial exclusion consistent (dropped from unigrams and windows).
Merge reads only train-*.json; sums match wiki_full.json exactly; all == nocof + cof. Ordering-relevant biases:
(1) MATERIAL: canonical spelling set mixes flat spellings with sharp F#; Wikipedia is spelling-consistent per document,
so seam pairs are starved (Db|F# = 17 vs Db|Gb = 160; merged 214). (2) small, conservative: multiword mention position.
(3) not a bias: nwords whitespace approximation (+0.7%) is a uniform PMI shift (lambda_0 only). cof_docs counter
under-reports in the prefilter branch (counter only).

Item 3 (report.json recomputation): numbers reproduce (kappa fifths order, P5/P1 0.5050/0.1718, circ frac 0.5024,
F# = 232, months profile). FAIL: "monotone" (7.07 -> 7.13); FAIL: "2.00 +/- 0.01, rho ~ 1800" (2 of 132 entries outside
+/-0.01, max 0.0256; rho median 1558, mean 1683, range 155–6698). Fragility: P5 = 0.505 profile hinges on the PMI diagonal
kappa(0) = 8.64 (sensitivity table; removing the diagonal inverts to P5 = 0.116). mstar key profiles are noise (should not
be presented). Bootstrap SDs understate uncertainty >= 3x (Poisson on f-weighted counts, no document clustering).

Item 4 (partials, circle/line): PASS exact — independent QR implementation; corpus partial fifths +0.6230; circle|line
+0.4004, line|circle +0.2689; 15 seam pairs listed; signed positions verified. Overstatement: spelling-control range
"+0.45…+0.58" omits Qwen then_key +0.34.

Item 5 (extraction/analysis + re-extraction): PASS; layer-9 anchor raw P5 0.332, projected 0.133, iso 0.57 reproduced.
Wording error: projected P5 is AT the projection-matched null (0.133 vs null mean 0.120, z +0.38, p 0.29), not below it.
Cosmetic: multictx_analyze uses double-centered PMI, decompose_rsa uses raw PMI.

Item 6: (a) predictive_matrix normalization/symmetrization sound; adding the model's column marginal as a control RAISES
partial fifths (1B then_key +0.24 -> +0.36; Qwen chord_after +0.42 -> +0.58; 7B +0.50 -> +0.61). (b) behavior_fewshot:
no leakage; scoring UNFAIR (unnormalized log-prob over unequal token lengths; multi-token names under-selected 1.6–7x;
accuracies are lower bounds; "errors at the seam" confounded). (c) theory_embedding: correct reading of Eq. 30/32; block
ablation "close to tautological" (1% of Frobenius norm). (d) circle_vs_line: correct. Latent bug: partial() returns 0.0421
instead of NaN on constant input.

Item 7 (tables vs files): predicting-position table exact; predictive matrices, few-shot means, per-model partial fifths
reproduce. Mismatches: Gemma/Qwen black-block max +0.84 (table +0.82/+0.83); Qwen argmax L14 not L16; flat-null cosine
0.748 (paired) / 0.704 (uniform), not 0.73; "0.97–0.99" is best-layer.

Top issues: (1) canonical spelling contaminates the corpus circle/line row — merged family: circle|line +0.58,
line|circle −0.07, partial fifths +0.66 (strengthens the thesis); (2) corpus P5 = 0.50 hinges on the diagonal;
(3) few-shot scoring length bias; (4) "below the null" wording; (5) cosmetic number errors; (6) partial() NaN guard;
(7) bootstrap SDs, unconverged circ frac; (8) predictive partials are conservative. Nothing overturns the headline claims.
