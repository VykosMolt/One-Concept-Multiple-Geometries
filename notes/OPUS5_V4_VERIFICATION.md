# Opus 5 v4 verification ledger

Independent verification ledger for the 120 numbered findings in
notes/review_opus5_v4.md. The review source is unchanged. Claims were checked
against paper/main.tex and paper/main.pdf, the cited source code, and the
cited raw result artifacts; markdown summaries are marked as
provenance-limited when the underlying raw artifact is absent.

PENDING_FINAL_V4 means that the v3 observation is recorded below, but the
corrected value must be taken from the final accepted v4 artifact after the
B1/B2 feature correction, fold-local preprocessing, and deterministic-seed
rerun. It is intentionally not an approval of the v3 number.

## Status counts

| status | count |
|---|---:|
| YES | 63 |
| NO | 12 |
| PARTIAL | 4 |
| PROVENANCE-LIMITED | 1 |
| PENDING_FINAL_V4 | 40 |
| total | 120 |

## Blocking findings

### B1 — the flagship circ feature was chromatic, not fifths-circle

The v3 code in phase5/fingerprint.py:47,51 computed
pc = (7*S) mod 12 and then cyclic distance in pc. That is chromatic
semitone distance. The intended circle-of-fifths distance is cyclic distance
in the signed line coordinate S. The independently checked examples are
C -> G: circle_fifths = 1, chromatic = 5 and
C -> D-flat: circle_fifths = 5, chromatic = 1. The v3 feature therefore
controlled the wrong geometry in both views; the rich harmonics were also
harmonics of chromatic distance. The same defect propagated to
phase5/ckpt_fingerprint.py and to cross-corpus/checkpoint results that consume
those fingerprints. scripts/corpus_cluster_boot.py and phase5/operators.py
use the correct candidate geometry and are not implicated by B1. The claim
that the flagship gain was beyond an explicit circle-of-fifths baseline is
therefore false as written in the abstract, central-claim box, contributions,
Discussion, and Appendix D.6.

### B2 — mean line coordinates for merged targets were not a defensible sole representation

For target-aggregated 12-class responses, the v3 mean coordinates for the
merged classes were {-7,+5} -> -1, {-6,+6} -> 0, and {-5,+7} -> +1.
These are not key spellings on the open line. A log-sum-exp target class is
dominated by the spelling with the larger model probability, so nearest
spelling distance and signed nearest-spelling displacement are principled
features to retain alongside class-level terms. The independent sensitivity
check changed the eight flagship v3 gains from 0.0368/0.0292/0.0164/0.0356
(window) and 0.0451/0.0359/0.0233/0.0393 (document) to
0.0128/0.0077/0.0021/0.0105 and 0.0173/0.0108/0.0057/0.0122,
respectively; Qwen-window became non-significant in that sensitivity. The
v4 baseline must therefore be reported from its complete artifacts, not
from the old 14--34 percent headline.

### B3 — the template-robustness clause misstated all three displayed quantities

The v3 leave-one-template-out artifacts independently give window
+0.0145...+0.0465, document +0.0203...+0.0519, and maximum p 0.0040
(lo1 x Qwen x window), not the paper's +0.026...+0.051,
+0.030...+0.052, and p <= .0005. The qualitative conclusion that every
leave-one-template-out aggregate is significant is supported by the v3
files, and Table C.7 itself matches v3. The corrected v4 clause remains
pending.

## M1--M6, statistical, methods, and presentation findings

### Major findings

* M1: The abstract's Spearman about 0.80 versus PMI 0.1--0.5 is an
  untested, hand-picked-layer statistic with no null or interval. The matched
  held-out comparison is the tested quantity and gives a smaller roughly 2:1
  contrast. Lead with the tested statistic or qualify the descriptive
  Spearman.
* M2: The favorable Phase-I 12-key merged-target line result must not be
  generalized to the Phase-II 15-key merged scorer. The latter's line
  controlled for circle is approximately -0.23...-0.01 across models and
  is mostly non-significant. The pooled-twin circle result is also close to
  definitional; see the reconstruction below.
* M3: results/corpus/wiki/convergence.json independently shows the
  canonical fifths partial increasing from about +0.27 at 0.11B words to
  +0.63 at 3.12B, without a visible plateau. Call the full-corpus
  estimate robust/significant under its null, not converged in magnitude.
* M4: The control sets differ: Result 1 uses block, commonness, and the
  other geometry; Result 4 uses glyph, edit, same-letter, alphabet,
  commonness, and the other geometry; behaviour uses another Phase-II set.
  State these per section.
* M5: phase5/thin_wikipedia.py uses one seed-0 binomial thinning draw for
  the size-matched baseline. The v4 extra seeds exist, but the final
  cross-corpus comparison must aggregate their baseline uncertainty.
* M6: The 7B +12 top-1 count is 8/60 source-by-template cells under the
  total scorer and 9/60 under length normalization, not 8--9/15 keys.
  Counting either sign gives 12/60 and 15/60; under the total scorer +1 is
  modal (11/60).

### Statistical audit

* The independently rerun 400-draw flagship null is calibrated (mean
  -0.0007, SD 0.0031; v3 observed +0.0368, about 12 null SDs).
  Joint key relabeling is appropriate for arbitrary key labels but does not
  test every smooth matrix.
* Within-row log-softmax, Delta-R2, and Delta-CV are invariant to a
  source-level additive shift in log C; the row normalizer cannot by itself
  create the signal. This is a useful Methods disclosure.
* The 36-cell multiplicity caveat is correct, but the v3 Bonferroni argument
  does not survive the nearest-spelling sensitivity (p approximately
  .003--.10).
* The document-cluster bootstrap (9,168 clusters, B=300) is appropriate;
  its v3 SD (0.0052 for 1B-window) exceeds the permutation spread. The
  41-shard percentile interval is visibly skewed and needs a limitation or
  better interval.
* v3 Appendix D.6 incorrectly says all standardized coefficients. The
  corpus predictor was unstandardized in v3, making lambda=1 effectively
  OLS at this sample size. v4 must standardize the corpus column with
  training-fold statistics.
* Leave-one-source-row-out separation itself is correct; the v3 z-scoring
  leakage is not acceptable when it can be removed. The response is not
  allowed into preprocessing.
* The twin line-controlled residualization has five columns, not merely a
  target line position and glyph class. The 17/108 count is inclusive at
  p <= .05; at p < .05 it is 15.
* Synthetic Result 7 uses five paired seeds. The t-test p=.044 and p=.001
  should be accompanied by the sign-test floor/results (p=.0625 or larger).

### Methods/code consistency

The independently checked inconsistencies are: wrong circ definition
(B1); unstandardized v3 corpus predictor despite D.6; incomplete Phase-II
null inventory and b/B versus (b+1)/(B+1) wording; checkpoint residuals
adding source-row dummies not described by the section 7 rich model; +0.5
numerator smoothing in matched PMI; a conjunction among the four family-C
regexes; undisclosed sentence-initial exclusion; and an unsupported Phase-I
statement about adding the model's column marginal. The stale
results/phase5/fingerprint_wikipedia_v3.txt and
results/phase5/fingerprint/wikipedia_v3.json also disagree because they are
different v3 runs; the paper's twin numbers come only from the older text
file and must be labelled or regenerated.

### Presentation and reference checks

Figure 5's near-universal stars, Figure C.1's window masses on the document
panel and positional hatched-bar legend, Figure 7's misleading stage-1-token
axis for stage-2/released points, Figure 4's omitted selected layers, and the
second legend in Figure 5(c) need correction or qualification. c/l and l/c
currently hide three control sets, and Table C.1 prints a different
finite-sample p conversion from the prose. Pre-2026 bibliography metadata
was independently checked and is correct. The 2026 entries were then checked
against their arXiv records and official proceedings pages on 2026-08-30;
titles, authors, and claimed ICML/COLM/workshop venues resolve. The omitted
Sadek workshop venue was added to paper/refs.bib from PMLR volume 303.

## M2 reconstruction: 12-key merged target versus 15-key merged scorer

| design | merged object | family/statistic | independently checked result | supported interpretation |
|---|---|---|---|---|
| Older Phase-I 12-key analysis | response target columns collapsed to 12 neutral pitch classes; corpus/model marginal design | modulates_to line controlled for circle, across four models | line partial canonical -> merged: 1B +0.69 -> +0.46; Gemma +0.74 -> +0.52; Qwen +0.79 -> +0.66; 7B +0.74 -> +0.56 | This specific 12-key target-aggregated analysis supports a qualified line-remains statement. |
| Phase-II 15-key enharmonic-merged scorer | model behaviour scorer pools twin spellings into neutral target classes | line controlled for circle, 4 templates per model and family | results/phase2/behavior/summary.txt merged rows: line partial -0.23...-0.01 across models/families listed in the review; mostly non-significant | The 15-key merged scorer does not support line survival. Its positive circle partial is largely induced by pooling twins and low ECI. |
| Corpus Phase-I merged family | corpus PMI target class merging | circle-vs-line partials with the Phase-I control set | merged circle-vs-line +0.58, line-vs-circle -0.07, P=1.00; canonical +0.40/+0.27, P=.632 | This is a corpus association result, not evidence for the 15-key behavioural merged scorer. |

These are different key sets, response constructions, families, and
statistics. No single sentence should merge them into “the line survives
enharmonic merging at two-thirds strength.”

## Corpus convergence disclosure

| corpus statistic | words | value | source/action |
|---|---:|---:|---|
| canonical fifths partial | approximately 0.11B | +0.27 | results/corpus/wiki/convergence.json; early point |
| canonical fifths partial | approximately 1.16B | +0.45 | results/corpus/wiki/convergence.json; intermediate point |
| canonical fifths partial | 3.1189B | +0.63 | results/corpus/wiki/convergence.json; full corpus |

The estimate is robust/significant under the appropriate null, but its
magnitude was still increasing over the observed corpus sizes. The paper
must not call it asymptotically converged. Any helper-vocabulary convergence
result is a separate analysis and must retain its own label.

## Complete 120-finding ledger

| ID | Reviewer claim / location | Independently verified status | Source artifact/key or line | Corrected value / action | Manuscript location affected |
|---:|---|---|---|---|---|
| 1 | Wikipedia documents, section 3 L128: 6.41M | YES | results/corpus/wiki/report.json all/ndocs = 6,407,814 | Retain rounded 6.41M. | section 3 L128 |
| 2 | Wikipedia words, section 3 L128: approximately 3.1B | YES | report.json all/nwords = 3,118,919,923 | Retain rounded approximately 3.1B. | section 3 L128 |
| 3 | Months circulant fraction, section 3 L136: 97% | YES | report.json all/months/circ_frac_offdiag = 0.9678 | Retain 97%. | section 3 L136 |
| 4 | Months paired spectrum, section 3 L136 | YES | report.json all/months/profile_abs_lambda = .499,.256,.118,.081,.033,.012 | Retain rounded spectrum. | section 3 L136 |
| 5 | Keys M* off-diagonal, section 3 L138: 1.97--2.00 | YES | report.json all/major_canon/M = 1.9744--1.9994 | Retain range. | section 3 L138 |
| 6 | PMI kernel in fifths order, section 3 L138 | YES | report.json major_canon@pmi/kappa, reindexed by 7d' mod 12 | Retain 8.64, 7.89, 7.59, 7.16, 7.07, 7.13, 6.48. | section 3 L138 |
| 7 | Naive Fourier share, section 3 L138: P5 .50 versus P1 .17 | YES | report.json major_canon@pmi/profile_abs_lambda = .505/.172 | Retain rounded shares and state naive profile. | section 3 L138 |
| 8 | Canonical fifths partial, section 3 L150: +.62 [.44,.68] | YES | results/corpus/wiki/cluster_boot.txt | Retain as v3 shard-cluster estimate; add convergence caveat. | section 3 L150 |
| 9 | Merged fifths partial, section 3 L150: +.66 [.49,.71] | YES | cluster_boot.txt merged row | Retain as v3 estimate with family label. | section 3 L150 |
| 10 | Merged circle-controlled and line-controlled partials, section 3 L150 | YES | cluster_boot.txt merged rows | Retain +.580 [.385,.643], -.070 [-.195,.073], P=1.000, with control-set label. | section 3 L150 |
| 11 | Canonical circle/line partials, section 3 L150 | YES | cluster_boot.txt canonical rows | Retain +.400 [.052,.551], +.269 [-.077,.582], P=.632. | section 3 L150 |
| 12 | ECI15, section 3 L153: .04 [.03,.44] | YES | results/phase2/corpus/pmi15.json eci15 statistics | Retain rounded value and interval. | section 3 L153 |
| 13 | White-key ordering rank, section 3 L150: 78/5040, p=.016 | YES | results/corpus/wiki/seam.txt | Retain rank 78, p=.0155 rounded. | section 3 L150 |
| 14 | Seam starvation, section 3 L150: Db-F# 17 versus Db-Gb 160 | YES | seam.txt; pmi15.json C[Db,Gb] = 160 | Retain, naming the two constructions. | section 3 L150 |
| 15 | Twin PMI, section 3 L153: 9.9--11.0 versus median 7.3 | YES | results/phase2/corpus/pmi15.txt | Retain range and median. | section 3 L153 |
| 16 | 15-key partials, section 3 L153: c/l +.42 and l/c +.38 | YES | pmi15.json stats = .4195/.3794 | Retain with explicit 15-key corpus label. | section 3 L153 |
| 17 | Helper factorization inputs, section 3 L150: V=3000 and 14,235 docs | YES | results/corpus_merged/keydocs_V3000.npz, C shape (3000,3000), ndocs=14235 | Retain. | section 3 L150 |
| 18 | Seam co-mentions, section 3 L153: 41 | YES | results/phase2/corpus/enharmonic_pairs_context.json | Retain 41 entries. | section 3 L153 |
| 19 | Half contain “enharmonic”, section 3 L153 | YES | enharmonic_pairs_context.json: 20 of 41 flagged true | Retain “20 of 41” or rounded half. | section 3 L153 |
| 20 | “Most of the rest slash notation”, section 3 L153 | NO | enharmonic_pairs_context.json: 3 of 21 non-enharmonic segments contain slash; PHASE2_RESULTS.md L125 | Say 3/21 slash; describe the remainder as harp-notation prose and key-change narration. | section 3 L153 |
| 21 | Key-signature cue class, section 3 L153: +.68/.00 | YES | results/phase2/corpus/conditional15.txt line 6 | Retain +.68 +/- .03 and +.00 +/- .06. | section 3 L153 |
| 22 | Withdrawn Poisson bootstrap, section 3 L150: canonical +/- .02/.03 and P=.87 | NO | results/corpus/wiki/seam.txt | Canonical SDs are +/- .05/.05, P=.874; merged SDs are +/- .02/.03, P=1.000. Keep families separate. | section 3 L150; Appendix A |
| 23 | Accidental-indicator energy, section 4 L165: 79.6% | YES | results/phase2/aliasing/summary.txt | Retain .796. | section 4 L165 |
| 24 | Boxcar widths, section 4 L172: 18/37/55/70/80/83% | YES | aliasing/summary.txt | Retain rounded .182/.373/.553/.700/.796/.829. | section 4 L172 |
| 25 | Partition statistics, section 4 L165/L172 | YES | aliasing/summary.txt | Retain .247, .314, .066, .067 with threshold definitions. | section 4 L165, L172 |
| 26 | Root-letter identity, section 4 L172: 34% | YES | aliasing/summary.txt = .339 | Retain rounded 34%. | section 4 L172 |
| 27 | Standard-spelling black-key Gram RSA, section 4 L165: +.70...+.86 | NO | results/decouple/*.txt; minimum is +.56 at 7B L2 and Gemma L26 is +.68 | Correct floor/range to +.56...+.86. | section 4 L165 |
| 28 | Respelled black-key RSA, section 4 L165: -.14...+.12 | YES | results/decouple/*.txt | Retain. | section 4 L165 |
| 29 | Respelled glyph RSA, section 4 L165: .5/.4/.7/.3 | YES | results/decouple/*.txt = .51/.43/.67/.28 | Retain with rounded values. | section 4 L165 |
| 30 | Best-layer fifths partials, Table C.1 | YES | results/nulls_multictx.txt = .206@L13, .290@L23, .233@L14, .342@L30 | Retain table values. | Table C.1 |
| 31 | 7B max-over-layer p, Table C.1: .045/.017 | YES | nulls_multictx.txt | Retain, but align text's p conversion with table. | Table C.1 |
| 32 | Same p in text, section 4 L172: .046/.018 with (b+1)/(B+1) | NO | nulls_multictx.txt; Appendix D.6 L541 says b/B | State whether table and prose use b/B or (b+1)/(B+1), and use one convention consistently. | section 4 L172; Table C.1; D.6 L541 |
| 33 | Other three p range, section 4 L172: .09--.25 | NO | nulls_multictx.txt; maximum nonsignificant p=.214 (Qwen block .193) | Correct range to .09--.21. | section 4 L172 |
| 34 | White-key exact-null p, Table C.1 note | YES | Table C.1 source nulls = .037/.003/.025/.000 | Retain rounded values; explain exact zero if desired. | Table C.1 |
| 35 | 7B prompt-final c/l p range, section 4 L172: .006--.03 | NO | results/phase2/geometry/olmo2_7b_symbol.json; p_max_free=.004... .030 | Correct lower bound to .004 and identify null. | section 4 L172 |
| 36 | Qwen/7B last-token line significant in all six families, section 4 L172 | NO | phase2 geometry JSONs: free-null p=.004--.034; glyph-preserving Qwen all six n.s., 7B 3/6 n.s. | State the null; do not claim all-six significance under glyph-preserving null. | section 4 L172 |
| 37 | Spelling/semantic subspace shares, section 4 L172: .53--.73/.46--.66 | NO | results/phase2/respell/*_decomp.txt, 120 last-position rows | Correct ranges to spelling .53--.76 and semantic .46--.73; explain nonexclusive shares. | section 4 L172 |
| 38 | Context effects, section 4 L172: +.04...+.07 band | NO | results/phase2/contrast/summary_symbol.txt; Gemma relational-other = -.026 | Say three models show the positive band; Gemma does not. | section 4 L172 |
| 39 | Behaviour l/c ranges, section 5 L177 | YES | results/phase5/scorer_robustness.txt | Retain ranges by model. | section 5 L177; Table C.6 |
| 40 | Behaviour c/l ranges, section 5 L177 | YES | scorer_robustness.txt | Retain small-model and 7B ranges. | section 5 L177; Table C.6 |
| 41 | “4/4 in 14 of 16 cells”, Results glance and section 5 | NO | scorer_robustness.txt; results/phase2/behavior/summary.txt | Correct to 15/16; only 7B x B_enharmonic is 3/4. | Glance L87; section 5 L177; PHASE2_RESULTS.md L135 |
| 42 | 63/64 to 58/64 templates, section 5 L177 | YES | scorer_robustness.txt: 15x4+3=63; family totals 13+16+15+14=58 | Retain after correcting 14/16. | section 5 L177 |
| 43 | c/l moves by at most .10, section 5 L177 | NO | scorer_robustness.txt; Gemma D_chord -.14 -> -.03 = .11 | Correct maximum movement to .11. | section 5 L177 |
| 44 | ECI falls .05--.25, section 5 L177 | YES | scorer_robustness.txt / Table C.6 | Retain range. | section 5 L177 |
| 45 | Length-normalized l/c values, section 5 L177 | YES | scorer_robustness.txt | Retain .37/.43/.50/.45. | section 5 L177 |
| 46 | Phase-I merged-target l/c, section 5 L177 | YES | results/predictive/{olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b}.txt modulates_to rows | Retain only with the 12-key merged-target label. | section 5 L177 |
| 47 | Merged-scorer c/l positive in every model, section 5 L177 | YES | results/phase2/behavior/summary.txt merged rows | Retain with the negative line/circle result and definitional caveat. | section 5 L177 |
| 48 | 7B enharmonic ECI and other models, section 5 L180 | YES | behavior/summary.txt B_enharmonic rows | Retain .11/.05; Qwen .51, 1B .70, Gemma .83. | section 5 L180 |
| 49 | “+12 top-1 in 8--9 of 15 keys”, section 5 L180 | NO | results/phase2/behavior/olmo2_7b.json recomputation | Correct to 8/60 total and 9/60 length-normalized cells; optionally report +/-12 counts 12/60 and 15/60. | section 5 L180 |
| 50 | Outside-family ECI .67--.90, section 5 L180 | YES | behavior/summary.txt totals excluding B | Retain. | section 5 L180 |
| 51 | Few-shot dominant accuracy, section 5 L180 | YES | results/behavior/fewshot_three_scorers.txt | Retain .08/.33/.75/.92. | section 5 L180 |
| 52 | PC1 tracks signed accidental count, section 5 L177 | YES | Reconstructed from retained matrices by scripts/circle_vs_line.py; results/phase5/circle_vs_line_pc1_v4.txt | Behavioural PC1 absolute Spearman is .70--.80 across the reported contexts; retained as a saved-matrix reconstruction, not a new model run. | section 5 L177 |
| 53 | Operator ECIs, section 6 L196 | YES | results/phase5/operators.txt | Retain exact six values, with operator definitions. | section 6 L196 |
| 54 | Operator controlled partials, section 6 L196 | YES | operators.txt | Retain values and section-specific controls. | section 6 L196 |
| 55 | Directional behaviour Spearman, section 6 L196 | YES | operators.txt | Retain descriptive values but add no-null/no-CI qualification. | section 6 L196 |
| 56 | Symmetrized/reverse/helper/PMI/40-word Spearman, section 6 L196 | YES | operators.txt | Retain as descriptive comparisons with selected-layer disclosure. | section 6 L196 |
| 57 | 7B prompt-final operator predictions, section 6 L196 | YES | operators.txt | Retain ranges and selected layers. | section 6 L196 |
| 58 | Span-mean operator predictions, section 6 L196 | YES | operators.txt | Retain max .15 and Qwen helper .37. | section 6 L196 |
| 59 | Document-conditional similarity, section 6 L196 | YES | operators.txt | Retain .14 maximum and behaviour -.07...+.03. | section 6 L196 |
| 60 | Matched-operator Delta-KL values, section 6 L196 | PENDING_FINAL_V4 | results/phase5/fingerprint/matched_{sym,rev,pmi}_neutral.json; v3 values reproduce paper | Record v3 sym .039/.030/.016/.040, dir .037/.029/.016/.036, rev .029/.027/.013/.030, PMI .021/.014/.012/.029; replace with final v4. | section 6 L196; Tables C.2/C.3 as applicable |
| 61 | “all p <= .001” for matched operators, section 6 L196 | PENDING_FINAL_V4 | v3 matched JSONs: max p=.0016, including reverse-Qwen .0016, sym-Qwen .0014, PMI-Gemma .0012 | v3 reviewer mismatch is recorded; state final v4 maximum and do not retain “all <= .001” unless supported. | section 6 L196 |
| 62 | Matched operators retain half to three quarters, section 6 L196 | PENDING_FINAL_V4 | v3 matched JSONs: PMI/sym .54,.46,.76,.73; PMI/directional .57,.48,.74,.81 | Replace loose range with exact v4 retention values or qualified range. | section 6 L196 |
| 63 | Flagship Delta-KL window, section 7/Table C.2 | PENDING_FINAL_V4 | v3 fingerprint/wikipedia_v3_neutral*.json = .03677/.02921/.01639/.03556 | v3 exactness is verified; publish only final v4 values. | Abstract; glance; section 7; Table C.2 |
| 64 | Flagship Delta-KL document, section 7/Table C.2 | PENDING_FINAL_V4 | v3 fingerprint/wikipedia_v3_neutral*.json = .04509/.03591/.02330/.03931 | v3 exactness is verified; publish only final v4 values. | Abstract; glance; section 7; Table C.2 |
| 65 | Theory-only KL and “14--34%”, section 7 | PENDING_FINAL_V4 | v3 fingerprint/wikipedia_v3_neutral*.json theory KL=.1167--.1340 and reduction=14.05--33.7% | Record v3 mismatch context; derive corrected headline from final v4. | Abstract; central claim; section 7 |
| 66 | p <= 8e-4 and b=0 in six of eight, section 7 | PENDING_FINAL_V4 | v3 fingerprint JSONs: max p=.0008; b=0 in 6/8 | Recompute with v4 and final B; retain finite-sample estimator. | Abstract; section 7; Appendix D.6 |
| 67 | Delta-R2 window/document, section 7 | PENDING_FINAL_V4 | v3 JSONs: window .286/.298/.185/.329; document .357/.382/.236/.375 | Preserve as v3 audit fact only; update from v4. | section 7; Table C.2 |
| 68 | Seven of eight document-cluster CIs, section 7/Table C.2 | PENDING_FINAL_V4 | v3 *_docboot.json q025/q975 match seven printed rows | Rebuild/recheck v4 document bootstrap and report all eight. | section 7; Table C.2 |
| 69 | 7B-document CI lower endpoint +.008, section 7/Table C.2 | PENDING_FINAL_V4 | v3 docboot q025=.00746, which rounds to +.007 | Record v3 rounding mismatch; report final v4 endpoint at stated precision. | Table C.2 |
| 70 | Rich Delta-KL and p, section 7/Table C.2 | PENDING_FINAL_V4 | v3 wikipedia_v3_neutral_rich*.json exact match | Replace all rich values/p with final v4 artifacts. | section 7; Table C.2 |
| 71 | Rich KL0 and “6--28%”, section 7 | PENDING_FINAL_V4 | v3 rich JSONs KL=.1052--.1231 and reduction=6.4--27.6% | Recalculate corrected rich reduction and range. | section 7; central claim |
| 72 | Target-prior base/rich Delta-KL, section 7/Table C.2 | PENDING_FINAL_V4 | v3 wikipedia_v3_neutral_tp.json and _rich_tp.json exact values | Keep as v3 trace only; update target-prior result from final v4. | section 7; Table C.2 |
| 73 | Document residual correspondence, section 7 | PENDING_FINAL_V4 | v3 wikipedia_v3 JSONs: .5989/.5881/.5000/.5982; rich .5294/.5183/.3908/.5050 | Recompute with true fifths features and report v4 residuals. | section 7; Table C.2 |
| 74 | Spelled-view theory LOO Spearman .78--.95, section 7 | PENDING_FINAL_V4 | v3 fingerprint/wikipedia_v3.json loo.theory=.7829--.9543 | Update after corrected feature fit. | section 7; Table C.7 |
| 75 | Spelled Delta-R2/Delta-KL cells, section 7 | PENDING_FINAL_V4 | v3 wikipedia_v3.json: Delta-R2 .102/.220/.022/.023 and .136/.124/.052/.296; Delta-KL .0056/.0117/.0074/-.0015 | Retain v3 mismatch record only; update every affected cell from v4. | section 7; Table C.7 |
| 76 | Spelled theory KL .05--.07, section 7 | PENDING_FINAL_V4 | v3 wikipedia_v3.json KL=.0523--.0708 | Recompute and report v4 range. | section 7; Table C.7 |
| 77 | Leave-one-template-out clause, section 7 L209 | PENDING_FINAL_V4 | v3 wikipedia_v3_neutral_lo{0..3}.json: window .0145--.0465; doc .0203--.0519; max p=.0040 | v3 reviewer claim is independently a mismatch; replace with final v4 ranges and p. | section 7 L209; Table C.7 |
| 78 | Table C.7 all rows/cells | PENDING_FINAL_V4 | v3 wikipedia_v3_neutral_{t0..t3,lo0..3}.json: all printed entries match v3 | Rebuild table from final v4 files; do not use v3 match as acceptance. | Table C.7 |
| 79 | Abstract says b=0 in all flagship cells | PENDING_FINAL_V4 | v3 fingerprint JSONs: b=0 in 6/8, not all 8 | v3 action is “six of eight”; update exact v4 count. | Abstract L68 |
| 80 | Audit-note Delta-CV range -.026...+.011, section 7 L225 | PENDING_FINAL_V4 | v3 target-aggregated flagship Delta-CV range -.026...+.047; narrower range is spelled view | Correct parenthetical and then recompute v4. | section 7 L225 |
| 81 | Significant Delta-CV cells, section 7 L225 | PENDING_FINAL_V4 | v3 JSONs: 7B five .0285/.0424/.0467/.0745/.0867; Gemma .0200/.0224 | Preserve v3 audit fact; classify v4 cells from final artifacts. | section 7 L225 |
| 82 | Twin difference cosines, section 7 L228 | PENDING_FINAL_V4 | stale fingerprint_wikipedia_v3.txt, 36 cells: Cb/B .65--.92; Gb/F# .41--.86; Db/C# .02--.54 | Label stale source or regenerate; update v4 twin outputs and deterministic-null p. | section 7 L228 |
| 83 | “17 of 108 cells”, section 7 L228 | PENDING_FINAL_V4 | fingerprint_wikipedia_v3.txt: 17 at p <= .05; 15 at p < .05 because p printed to 2 dp | State inclusive threshold and recompute under final v4. | section 7 L228 |
| 84 | Cross-corpus sizes, section 8/Table C.3 | YES | results/phase5/scan_*.log | Retain exact sizes; these counts are not feature-model dependent. | section 8; Table C.3 |
| 85 | Wikipedia extraction counts, section 8 | YES | results/phase5/scan_wikipedia_perdoc.log | Retain 47,247 A; 10,775 B; 19,162 D; 317 C; 9,168 docs. | section 8 |
| 86 | 633-pair Delta-KL, section 8/Table C.3 | PENDING_FINAL_V4 | v3 olmomix_dclm_neutral.json = .0365/.0290/.0301/.0471, max p=.01849 | Recompute all four cells with v4 features. | section 8; Table C.3 |
| 87 | 3,843-pair Delta-R2/Delta-KL, section 8/Table C.3 | PENDING_FINAL_V4 | v3 olmomix_dclm_big_neutral.json: Delta-R2=.1822/.1275/.1525/.1904; Delta-KL=.0211/.0097/.0171/.0235 | Replace with v4 values and final p. | section 8; Table C.3 |
| 88 | DCLM theory KL .17--.21, section 8 | PENDING_FINAL_V4 | v3 big-DCLM neutral JSON KL=.1728--.2098 | Recompute from v4. | section 8 |
| 89 | 633-pair beats thinned Wikipedia, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_neutral.txt: +.0244/.0246/.0252/.0301 | Use multi-seed thinning and corrected v4 fingerprints. | section 8; Figure C.1 |
| 90 | 3,843-pair mixed result, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_big_neutral.txt: doc +.0079...+.0114; window -.0069/-.0077/-.0163; Qwen +.0023 | Recompute with v4 and thinning-seed variance; preserve negative cells if they remain. | section 8; Figure C.1 |
| 91 | Table C.3 all 36 DiD entries, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_neutral.txt and big_neutral.txt: all printed v3 entries match | Regenerate all entries; no cherry-picking stable cells. | Table C.3 |
| 92 | Spelled-view significant cross-corpus cells, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_spelled.txt: .0115/.0116/.0037/.0046/.0166; 54-sh min p=.33 | Recompute all spelled cells with v4 feature model. | section 8; Table C.3 |
| 93 | DiD tally, section 8: 10/72, 36/34/2, .004--.05 | PENDING_FINAL_V4 | v3 parsed crosscorpus_compare files | Record v3 tally; publish only corrected v4 tally. | section 8 |
| 94 | Modulation DiD on DCLM, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_big_neutral.txt: -.0138...+.0095, min p=.09 | Recompute and preserve exact signs/status. | section 8 |
| 95 | 9-sh spelled modulation-window failure, section 8 | PENDING_FINAL_V4 | v3 crosscorpus_compare_big_spelled.txt: p=.76 | Verify under v4 before retaining. | section 8 |
| 96 | Table C.4 all 20 synthetic cells, section 9 | YES | results/phase3/analysis_main.txt and summary_main.json | Retain; not inherited from B1/B2. | section 9; Table C.4 |
| 97 | Code-controlled l/c footnote, section 9 | YES | analysis_main.txt line_given_c = .1135/.4207/.7865/.995 | Retain rounded values. | section 9; Table C.4 |
| 98 | Uncontrolled l/c versus oracle, section 9 | YES | results/phase3/analysis_uncontrolled.txt = .436/.422, oracle .439 | Retain. | section 9 |
| 99 | Twin-target asymmetry, section 9 | YES | analysis_main.txt: mean +.013, 5/5, t p=.044; sign p=.062 | Retain both inferential views. | section 9 |
| 100 | Early KL, section 9: .078 versus .276 at step 100 | YES | phase3 trajectory table, step 100 | Retain. | section 9 |
| 101 | Dose-response Spearman, section 9 | YES | phase3/dose_analysis.txt = +.85 line, -.68 KL | Retain. | section 9 |
| 102 | Representation shift, section 9 | YES | phase3 analysis: hidden_last +.081 (p=.014), hidden_mid +.097 (p=.001) | Retain as paired shift with both absolute partials negative. | section 9 |
| 103 | Absolute partials and null q95, section 9 | YES | phase3 analysis: -.15 versus -.23; q95 .17--.21 | Retain. | section 9 |
| 104 | Positive control, section 9 | YES | phase3 analysis: l/c .995; hidden_mid c/l +.32 | Retain. | section 9 |
| 105 | 91 runs, section 9 | YES | results/phase4/runs: 60 primary + 18 control + 12 extended + 1 smoke | Retain 91 and count the smoke run explicitly. | section 9 |
| 106 | Global KL floor and rare-row ratio, section 9 | YES | PHASE4_RESULTS.md: .0010--.0071 and 3.3x--24.3x | Retain rounded ranges. | section 9 |
| 107 | Power law, section 9 | YES | phase4/exposure_collapse.txt: 1.18 N^(-.74); partials -.843/-.186 | Retain with fit scope. | section 9 |
| 108 | Aligned-code effect, section 9 | PARTIAL | phase4 analysis: p=.020/.035/.157; prefactors .94/1.5; exponents -.72/-.76 | Retain p/prefactor facts but remove common-exponent implication. | section 9 |
| 109 | Unique-state control “about as well”, section 9 | PARTIAL | phase4 unique control .0064@2585; rare alias .0105@2270 | Clarify these are different arms and that unique state is learned better in the quoted values. | section 9 |
| 110 | Timing thresholds, section 9 | PARTIAL | phase4 file: representation 222--2425; behaviour 759--7698 | Keep approximate rounded ranges but cite exact endpoints. | section 9 |
| 111 | Natural ratios written Cb:B etc., section 9 | PARTIAL | PHASE4_RESULTS.md L26; underlying shares 121/438=.28 and 162/232=.70 | Call them rare shares, not ratios; correct the two misleading values. | section 9 |
| 112 | Cb/B latent JS=.228, 81st percentile, section 9 | PROVENANCE-LIMITED | PHASE4_RESULTS.md L9 only; no raw result artifact located | Qualify as summary-only or recover raw evidence before relying on it. | section 9 |
| 113 | Checkpoint l/c trajectory, section 10/Table C.5 | PENDING_FINAL_V4 | v3 results/phase5/ckpt_trajectory.txt; v4 output exists but is not accepted here | Record v3 trajectory -.24 -> .19 -> .27 -> .34 -> .33 -> .42/.42/.44; replace with final v4. | section 10; Table C.5; Figure 7 |
| 114 | Checkpoint document residual, section 10/Table C.5 | PENDING_FINAL_V4 | v3 ckpt_trajectory.txt values/p match printed rows | Recompute with v4 rich nuisance features and final deterministic seeds. | section 10; Table C.5 |
| 115 | Table C.5 all 11 rows x 4 columns, section 10 | PENDING_FINAL_V4 | v3 ckpt_trajectory.txt all values match to printed precision | Regenerate all rows from ckpt_fingerprint_v4.json. | Table C.5 |
| 116 | Twin asymmetry 4.3 -> 2.6, section 10 | PENDING_FINAL_V4 | v3 ckpt_trajectory.txt = 4.34 -> 2.63 | Replace from final v4 checkpoint artifact. | section 10; Figure 7 |
| 117 | Stage-2 deltas, section 10 | PENDING_FINAL_V4 | v3 ckpt_trajectory.txt: line +.09/.09/.11; residual +.04/.01/.02 | Recompute under v4; retain seed labels. | section 10; Table C.5 |
| 118 | c/l approximately 0 throughout, Result 9 | PENDING_FINAL_V4 | v3 trajectory/Figure 7 includes -.25 at 49B | Remove “throughout”; explicitly mention the 49B excursion and update v4 trajectory. | Result 9; section 10; Figure 7 |
| 119 | Released-model provenance numbers, section 10 L272 | YES | phase5/soup_check.py rerun on retained safetensors; results/phase5/soup_check_v4.txt | Ingredient endpoints are mutually close (median relative distance about .096) but each and their mean are about 1.12 from the released model; retain with artifact citation. | section 10 L272 |
| 120 | Figure 1 caption rho=.25 (15-key), .68 (12-key) | YES | phase2/keys15.py geometry recomputation = .2494/.6806 | Retain; this confirms the paper distinguishes the geometries here. | Figure 1 caption |

## Update rule after the v4 rerun

Rows marked PENDING_FINAL_V4 must be updated from the accepted v4 JSON/text
artifacts, not by copying the preview or the v3 value. The v3 mismatch facts
in those rows are retained as audit history. B1, B2, and B3 remain blocking
until the corrected values, manuscript, figures, and cross-references are
rechecked together.
