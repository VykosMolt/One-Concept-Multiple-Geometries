# Adversarial review — "One Concept, Multiple Geometries", tag `paper-v2` (24754d7)

Reviewer: independent Opus 5 pass, 2026-09-02. Read: all 559 lines of `paper/main.tex`, the 24-page
`paper/main.pdf` (rendered), `paper/refs.bib`, `paper/make_figs.py`, `phase5/{theory_features,fingerprint,
crosscorpus_compare_v4,operators}.py`, `phase2/{keys15,behavior_fit,geometry_fit,corpus15}.py`,
`scripts/theory_embedding.py`, and the results files listed in the brief. No analysis chain was re-run; all
computations below are seconds-long reads of stored artifacts, plus closed-form Fourier arithmetic.

Grades: **BLOCKING** = claim false or unsupported as stated. **MAJOR** = materially misleading or a real
statistical problem. **MINOR** = presentation.

Counts: **4 BLOCKING, 14 MAJOR, 19 MINOR.**

---

## BLOCKING

### B1. The flagship held-out residual is one held-out source row: C♭ major, 38 corpus mentions

`paper/main.tex:243` (Result 7), `:68` (abstract), `:90` (glance), `:110` (central claim), `:117`
(contribution 4). Every `loo.kl_rows` array is stored in
`results/phase5/fingerprint/wikipedia_v4_neutral_rich.json`, so the per-row decomposition of ΔKL is free.
Rich model, target-aggregated, modulation × window, per-row ΔKL (theory − both), in the 15-key order
C♭ G♭ D♭ A♭ E♭ B♭ F C G D A E B F♯ C♯:

```
OLMo-1B  C♭:+0.0341  ... C♯:-0.0077   mean +0.00346
Gemma    C♭:+0.0398  ... C♯:+0.0020   mean +0.00607
Qwen     C♭:+0.0018  ... C♯:-0.0122   mean -0.00018
OLMo-7B  C♭:+0.0472  ... C♯:-0.0373   mean +0.00334
```

Leave-one-source-row-out on the reported statistic (the mean over 15 held-out rows):

| model | rich ΔKL | drop C♭ | % of full retained | base ΔKL | drop C♭ | TP-rich ΔKL | drop C♭ |
|---|---|---|---|---|---|---|---|
| OLMo-2-1B | +0.00346 | +0.00126 | 36 % | +0.00406 | +0.00213 | +0.00391 | +0.00140 |
| Gemma-2-2B | +0.00607 | +0.00366 | 60 % | +0.00533 | +0.00376 | +0.00446 | +0.00224 |
| Qwen2.5-3B | −0.00018 | −0.00032 | — | −0.00013 | −0.00023 | −0.00033 | −0.00049 |
| OLMo-2-7B | +0.00334 | **+0.00020** | **6 %** | +0.00381 | +0.00076 | +0.00297 | +0.00017 |

Qwen's null flips sign on dropping one row too: dropping C♯ gives +0.00068 (rich) / +0.00087 (base).
Cf. `results/phase5/cond_wikipedia.npz['uni']`: C♭ has 38 mentions in 3.1 B words and its `A_win40` row
contains 48 counts in total; C♯ has 121 mentions, 66 row counts. `results/phase5/fingerprint/
wikipedia_v4_neutral_rich.json` also shows the C♭ row is where the theory model is worst
(`kl_rows.theory` C♭ = 0.131–0.197 against a 15-row median near 0.03), i.e. the row with the most headroom
is also the row with the least corpus evidence.

So the claim "a small, model-dependent residual … significant by relabeling in three models" is, for two of
the three models, a statement about C♭ major. For OLMo-2-7B — a model the paper carries through the
abstract, glance box and contributions — 94 % of the reported gain is one row estimated from ~50 co-mention
events. `PHASE5_RESULTS.md:198` already records "C♭/G♭ rows rest on 38/162 source mentions" as a limitation;
the manuscript does not, and no leave-one-source-row-out sensitivity appears anywhere, although Appendix C.7
reports leave-one-*template*-out robustness from the same artifacts. Report the jackknife; restate Result 7,
the abstract and the glance bullet in terms of what survives it.

### B2. "the RSA with the glyph block rises to +0.5/+0.4/+0.7/+0.3" is false; the paper's own Table C.1 has the contradicting number

`paper/main.tex:158` (Result 2), against `paper/main.tex:364–367` (Table C.1, glyph-RSA column
+0.86/+0.84/+0.84/+0.86) and `results/decouple/{olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b}.txt`. Under the
canonical spelling the black-key block and the glyph block are *the same partition*, RSA +0.84…+0.86. After
respelling, the black-key RSA collapses to −0.14…+0.12 (correct as stated) and the glyph RSA goes to
max-over-six-reported-layers 0.51 / 0.43 / 0.67 / 0.28. That is a **fall** from ~0.85, not a rise. The
sentence, and the abstract's "a respelling intervention moves the block with the glyph"
(`paper/main.tex:66`), assert a directional result the artifacts contradict. What the data support is
weaker: the residual block that survives respelling tracks the glyph rather than black-key membership, but
the block loses half to two-thirds of its coherence in the process. Two further problems compound this:

* The four quoted numbers are **unnulled maxima over six layers**. Every other max-over-layers quantity in
  the paper (`:168`, Table C.1) carries a max-over-layers selection null; this one does not.
* The intervention confounds glyph with a ~1000× frequency drop. From
  `results/corpus_merged/wiki_full.json`, the four replacement strings have unigram counts
  **B♯ 1, E♯ 2, F♭ 21, C♭ 38**, replacing C 6348, E 3101, F 4290, B 1544. A collapse of *both* block RSAs
  toward zero is exactly what an out-of-vocabulary explanation predicts, and that is what happens. Figure 3's
  caption concedes "glyph and rarity remain entangled" (`:165`) in one clause; the Result box and the
  abstract state the causal reading without it.

### B3. "circle or line adding ≤0.04 in every cell" is false, and the quoted R²_cv range matches no computable summary

`paper/main.tex:170`. Recomputed from `results/phase2/geometry/*_symbol.json` (`per_layer[*].cv_ortho`,
`cv_circle`, `cv_line`), restricted to the prompt-final position the sentence is about:

* `max(cv_circle, cv_line) − cv_ortho > 0.04` in **30 of 660** model × family × layer cells; the maximum is
  **+0.104** (OLMo-2-7B, D_chord, layer 28), with twelve consecutive layers of that cell between +0.04 and
  +0.10 and eight consecutive layers of OLMo-2-7B B_enharmonic between +0.04 and +0.08.
* `cv_ortho` at the prompt-final position ranges **−0.043 to +0.755** per layer, not "0.1–0.55".
* Under the only reading that rescues the ≤0.04 bound — "cell" = model × family, averaged over layers — the
  bound holds (max mean gain +0.030) but the R² range is then **0.062 to 0.449**, still not 0.1–0.55.

The conclusion the sentence supports ("Hidden-state geometry is therefore not where this paper's positive
claim lives") survives; the stated bound does not. Define "cell", quote the correct numbers, and say whether
the summary is per layer or per layer-mean. The same sentence is the source of `PHASE2_RESULTS.md:50–51`, so
the error is upstream.

### B4. The Reproducibility section names a file that does not exist, and claims a repository contents that is not true

`paper/main.tex:289`: "Model weights, corpora and extracted hidden-state tensors are excluded from the mirror
and listed with sizes and SHA-256 digests in `MIRROR_MANIFEST.md`." That file is **not in the repository**
(`ls MIRROR_MANIFEST.md` → No such file). The same sentence says "All code, results files, figures and logs
are in the public repository". `.gitignore:6–7` excludes `results/multictx/*/*.npz` and
`results/corpus_merged/*.npz`, and both are load-bearing: `results/phase5/circle_vs_line_pc1_v4.txt` records
that Result 3's PC1 statement (`paper/main.tex:175`, |ρ| = 0.70–0.80) is reconstructed from
`results/multictx/*/major.npz`, and Result 1's helper-word factorization (`:148`,
`scripts/theory_embedding.py`) consumes `results/corpus_merged/keydocs_V3000.npz`. Neither is redistributable
from the tag as described.

---

## MAJOR

### M1. The document-cluster interval is an undeclared, uncorrected percentile bootstrap that is shifted toward zero; the pivotal interval from the same draws flips the headline caveat

`paper/main.tex:243`, `:284`, Table C.2 (`:375–391`), `phase5/fingerprint.py:687`. The code takes
`np.percentile(arr, 2.5)` / `97.5` over 300 draws; the paper never says which interval it reports. The
replicate distribution is systematically shifted toward zero — exactly the attenuation you expect when
resampling with replacement leaves ~63 % distinct documents and a *gain from adding a noisier predictor*
therefore shrinks. Interval midpoint versus point estimate, rich model, aggregated modulation:

| cell | observed ΔKL | percentile CI | midpoint | shift | pivotal CI (2θ̂ − q) |
|---|---|---|---|---|---|
| OLMo-1B × window | +0.00346 | [−0.00092, +0.00543] | +0.00226 | −35 % | **[+0.00148, +0.00783]** |
| Gemma × window | +0.00607 | [+0.00165, +0.00710] | +0.00438 | −28 % | [+0.00504, +0.01050] |
| Qwen × window | −0.00018 | [−0.00104, +0.00033] | −0.00036 | — | [−0.00069, +0.00068] |
| OLMo-7B × window | +0.00334 | [−0.00060, +0.00460] | +0.00200 | −40 % | **[+0.00207, +0.00728]** |

The shift is toward zero in every positive cell and away from zero in every negative cell (see the harmonic
rows), i.e. it is shrinkage, not noise. "Under the rich model only Gemma's window interval excludes zero" —
the sentence that carries the negative caveat in the abstract (`:68`), the glance box (`:90`), the central
claim (`:110`), contribution 4 (`:117`) and Result 7 — is therefore an artifact of choosing the percentile
interval. Three of four models exclude zero under the pivotal interval from the identical 300 draws. This
cuts *against* the paper's own conclusion; the honest fix is to name the interval type, report both, and
state that the bootstrap estimates ΔKL for an effectively smaller corpus. Separately, `B = 300` puts the
2.5 % endpoint on the 7.5th order statistic — far too few draws for a decision rule of the form "excludes
zero".

### M2. The matched-operator control is close to vacuous: the corpus matrix is 98 % symmetric, and the paper never says so

`paper/main.tex:184`, `:88`, `:114`, Result 4 title (`:193`), Appendix A row "Symmetric vs directional
framing" (`:335`), abstract (`:68`). Computed from `results/phase5/cond_wikipedia.npz['A_win40']`:

* off-diagonal Pearson(N, Nᵀ) = **0.977**, Spearman 0.956;
* the three conditional-row JS distance matrices over 105 pairs correlate ρ = **0.985** (directional vs
  symmetrized), 0.967 (directional vs reverse), 0.991 (symmetrized vs reverse);
* as actually used in the regression, the log-conditional columns correlate Pearson **0.980** (directional
  vs symmetrized) and 0.934 (directional vs reverse).

"a symmetrized conditional from the same counts does equally well, so the contrast is one of construction,
not directionality" is therefore a report that two near-identical objects behave near-identically. The
finding is real but it is a fact about *this corpus at this window* — key co-mentions within 40 words are
almost order-symmetric — not a general separation of construction from directionality, and the symmetry
statistic appears nowhere in the paper. It carries the title change, the abstract, contribution 1 and an
Appendix A "Corrected" row. Report Pearson(N, Nᵀ) and rescope the claim.

### M3. The relabeling p-values are unstable at the scale of the effect

Same artifacts. Three predictors that correlate ρ ≥ 0.91 with each other give, for OLMo-1B (Result 4,
`paper/main.tex:196`): symmetrized +0.0052 (p = .0038), directional +0.0041 (p = .015), reverse +0.0019
(p = .107). One is "significant at Bonferroni-36", one at nominal .05, one is null — for the same corpus
statistic. Result 4 concedes "should not be used to rank the operators universally" but attributes it to the
corrected regression being "less orderly" rather than to the estimator's variance. Since the paper's entire
positive Phase-V statement is a *count of models passing p < .05*, this instability is the more important
fact and belongs in the audit box. It also bears directly on the Qwen-vs-rest split, which B1 shows is one
row wide.

### M4. The cluster bootstrap is quoted only where it is negative for the headline, and suppressed where it is decisive against the corpus

`paper/main.tex:243`: "The harmonic-relation and chord families show no significant cell under either theory
model." From `results/phase5/fingerprint/wikipedia_v4_neutral_docboot.json`, base model:

* Gemma × C_harmonic × window: ΔKL = −0.00230, CI [−0.00381, **−0.00055**] — excludes zero, negatively.
* OLMo-7B × C_harmonic × document: ΔKL = −0.00685, CI [−0.01004, **−0.00106**] — excludes zero, negatively.

Under the same criterion the paper applies to Gemma's positive modulation cell, adding the corpus row
*significantly hurts* held-out prediction in two harmonic cells. "No significant cell" is true only for the
one-sided relabeling test; the bootstrap the paper elsewhere treats as the more conservative arbiter says
something stronger and is not reported. Also undeclared: the document bootstrap was run for only 16 of 36
aggregated cells (C_harmonic and E_modulation × window/document); D_chord and B_any got none.

### M5. The "conditional-row vs association" taxonomy is contradicted by the helper-word factorization — the operator Karkada's theory actually predicts

`paper/main.tex:194` (Result 4), `:88` (glance), `:114` (contribution 1), `:276` (discussion), Figure 4
(`:190`, greys/gold vs blues). From `results/phase5/operators.txt`, Spearman with next-key behaviour:

```
conditional-row   directional 0.79–0.80   symmetrized 0.78–0.82   reverse 0.78–0.83
association       helper-word 0.64–0.68   Karkada PMI 0.39–0.54   same-count PMI 0.10–0.24
```

The within-"association" spread (0.10 → 0.68) is four times the between-class gap the paper contrasts
(0.68 → 0.80). Twin ECI splits the same way: association operators give 0.04, 0.05 and **0.74** — the helper
factorization keeps twins *farther apart than any conditional-row operator*. So the taxonomy explains
neither the alignment ordering nor the enharmonic behaviour, and the one association operator that Karkada's
theory is actually about behaves like the conditionals. Result 4 states the HELP numbers, but the discussion
(`:276`, "PMI-type association strongly identifies enharmonic twins") and the abstract (`:68`, "against
0.1–0.5 for PMI-type operators", which silently omits HELP at 0.64–0.68) do not. Figure 4's colour scheme
asserts a dichotomy its own bar heights refute.

### M6. The distinction being reported is first-order vs second-order (syntagmatic vs paradigmatic) similarity, and the paper does not name or cite it

`paper/main.tex:103–107`, `:184`, `:276`. PMI is a direct-association (syntagmatic) statistic on the pair
(i, j); conditional-row JS is a distributional-context (paradigmatic) statistic. Enharmonic twins are
syntagmatically bound (they co-occur in enharmonic-equivalence prose — the paper's own document scan at
`:151` finds 41 co-mentions, 20 containing the word "enharmonic") and paradigmatically distinct (their next-key
rows differ). That is the whole of the "operator-dependent geometry" result on this domain, and it is a
textbook distinction in distributional semantics that appears nowhere in `main.tex`, `refs.bib` or
`LITERATURE_AUDIT.md`. Contribution 1 and the title are stated as if the operator dependence itself were
novel; with the standard framing the novelty narrows to (a) the specific musical instance and (b) which
operator matches behaviour. That is still a contribution, but it needs to be positioned against the existing
literature rather than presented as a new phenomenon.

### M7. The multiplicity family is chosen at its smallest defensible size

`paper/main.tex:247` (audit box): "only Gemma survives a Bonferroni factor of 36." The Wikipedia held-out
grid actually run is 36 cells × 2 views × {base, rich} × {±target prior} = **288** tests
(`results/phase5/fingerprint/wikipedia_v4{,_neutral}{,_rich}{,_tp}{,_rich_tp}.json`), plus four matched
operators, four cross-corpora × 36 × 2 views, and 12 checkpoints × 2 families × 2 extractions. Gemma's best
p is .0004 rich-window; ×36 = .014 (survives), ×288 = .115 (does not). Nothing in the corrected Phase V
survives a correction over the grid the authors actually computed. Say so, or justify 36 as the pre-declared
family.

### M8. The paper's stated orthographic control family is used by no analysis, and four different control sets are used without disclosure

`paper/main.tex:126` names the orthographic family as "(glyph class, has-accidental, edit distance,
root-letter identity, alphabet distance, token count)"; `:132` says partials control "for the orthographic
and commonness features". Actual code:

| analysis | file | controls |
|---|---|---|
| Result 1, 15-key PMI (`:151`) | `phase2/corpus15.py:12` | glyph, edit, same_letter, commonness |
| Result 3, behaviour (`:175`) | `phase2/behavior_fit.py:12` | + alphabet |
| Result 4, operators (`:194`) | `phase5/operators.py:16` | + alphabet |
| §4, key-name geometry (`:168–170`) | `phase2/geometry_fit.py:26` | + alphabet, tokcount |
| §4, context contrast | `phase2/context_contrast.py:16` | + alphabet, tokcount |

`has_accidental` and `n_accidentals` (unsigned key-signature-size difference) are **defined in
`phase2/keys15.py:candidate_geometries` and used as controls nowhere**, even though `n_accidentals` is the
closest orthographic proxy for line distance and *is* in the Phase-V regression (`paper/main.tex:551`,
"unsigned key-signature-size difference"). Appendix D never states any control set. To be fair to the
authors, the omission is conservative: adding both to the Phase-II behavioural partial (recomputed from
`results/phase2/behavior/*.json`) *raises* line|circle in every model × family — modulation goes
+0.44→+0.59 (1B), +0.54→+0.58 (Gemma), +0.52→+0.53 (Qwen), +0.56→+0.64 (7B). Result 3 is not at risk; the
methods description is simply not what was run, and cannot be reproduced from the paper.

### M9. "neither privileged" is contradicted by every headline

`paper/main.tex:129`: "we report the *spelled* view (15 targets) and this *target-aggregated* view side by
side, neither privileged." The abstract, the glance box, the central claim, contribution 4 and Result 7's
lead paragraph all quote target-aggregated numbers; the spelled view is a trailing sentence with "no claim is
built on it" (`:243`). The spelled view is in fact the *stronger* result (window ΔKL +0.0038/+0.0092/+0.0010/
+0.0096, p = .005/.013/.159/.0008 — verified against `results/phase5/fingerprint/wikipedia_v4.json`), so the
choice is conservative, but the audit box declares only that "Modulation was emphasized after the v3 grid was
seen" and not that the view was chosen after seeing both. Either declare the view choice as post hoc or drop
"neither privileged".

### M10. The target-aggregated view is structurally blind to the enharmonic seam the paper is about

`phase5/theory_features.py:104–118` (`pair_indices`): in the aggregated view the source's own pitch class is
excluded. For the six twin sources (C♭, G♭, D♭, B, F♯, C♯) the excluded class *contains the twin*, so the
enharmonic partner is never a scored target. Appendix D.6 states the exclusion (`:549`) but not the
consequence. The paper's flagship §8 analysis therefore cannot see the one relation — "these two spellings
are the same neutral pitch class" — that motivates the 15-key design and the entire circle/line contrast.
Note the interaction with B1: the C♭ row that carries the effect is also a row whose most informative target
has been removed.

### M11. Ridge with λ = 1 on standardized columns is OLS, and the paper does not say so

`paper/main.tex:555`, `phase5/fingerprint.py:78–93`, `phase5/theory_features.py:522–540`. Columns are scaled
to unit variance on the training pairs, so `diag(XᵀX) ≈ n_train = 154` and λ = 1 gives a shrinkage factor of
154/155 = **0.9935**: 0.65 % regularization. The "rich" design is 25 theory columns + corpus + target prior +
unpenalized intercept ≈ 28 near-unpenalized parameters fitted on 154 pair observations drawn from **14
independent source rows**. λ was not tuned and there is no evidence it was ever varied. This is the estimator
behind every number in §8; the base-vs-rich comparison and the claim that the rich model is the more
demanding baseline both rest on it. State that the estimator is effectively OLS, or report a λ sweep.

### M12. §7's headline overstates what a stipulated-oracle experiment can show

`paper/main.tex:68` (abstract), `:89` (glance), `:213`, `:220`. The synthetic factorial trains to KL ≤ 0.001
against an oracle whose law is *defined* to be periodic; "converged behaviour is periodic in both code
conditions" is close to a statement that the optimizer converged. What the experiment bounds is a mechanism
in a toy world with 15 states, 12 latent classes, 8-symbol codewords and a 4-layer d = 128 model; the
abstract's "Synthetic controls show that output-code geometry and sparse-alias learning are insufficient for
the converged line" reads as a statement about *the natural line*, which no synthetic run contains. §7's own
closing sentence ("Other lexical or learning explanations are not thereby excluded", `:220`) is the right
scope and should be the one that propagates to the abstract and the glance box. Note also that §6 already
makes the output-code hypothesis unnecessary: if Wikipedia's conditional rows are line-like, a converged
model is line-like without any code effect.

### M13. The "31 %" base rate is not the base rate for the failure mode demonstrated

`paper/main.tex:66` (abstract), `:158` (Result 2), `:168`, Appendix B (`:349–351`). I reproduced every
Fourier number exactly by exhaustive enumeration of the 4 094 non-trivial binary partitions of ℤ₁₂:
boxcar shares 18.2/37.3/55.3/70.0/79.6/82.9 %, the accidental indicator at 79.6 % in the k = 5/7 pair,
24.7 % > ½, 31.4 % ≥ ½, 6.6 % > 0.7 — all correct. But the quantity that matters for "a categorical
attribute mimicking a cyclic fundamental is the common case, not the exception" is the rate at which a random
binary attribute lands in the **fifths** pair specifically, and that is **5.0 %** (204/4 094 at ≥ ½) and
**0.88 %** (36/4 094 at > 0.7), not 31 %. The 31 % figure counts concentration into *any* of six mode groups,
including the chromatic fundamental, which is not the failure mode under discussion. The qualitative lesson
("Fourier power … does not identify the feature that caused it", `:168`) is correct and well made; the
quantitative framing is a factor of six too generous and appears in the abstract.

### M14. Figure 6 — the figure that carries Result 7 — shows relabeling stars and no cluster intervals

`paper/main.tex:239`, `paper/make_figs.py`. Every panel is annotated with `*: relabeling p < .05`; none shows
the document-cluster bootstrap interval, which is the uncertainty the surrounding text and the abstract treat
as decisive. Given M1 and B1 the figure presents the most favourable of three available uncertainty
statements. Add the intervals (or the per-row spread), and mark the C♭ row's contribution.

---

## MINOR

1. `paper_style/tables.tex:26`: `\statusBounded` expands to `\textsc{Bounded screen}` — a leftover from
   another template. Appendix A therefore prints "Bounded screen" for four rows (few-shot accuracies, corpus
   P₅ share, spelled-view gain, target-popularity robustness). Visible in the PDF, p. 15–16.
2. Estimator inconsistency: Appendix A (`:334`) records replacing `b/B` with `(b+1)/(B+1)` as a *correction*,
   but Phase-II behaviour and geometry still use `b/B` (`phase2/behavior_fit.py:33`,
   `phase2/geometry_fit.py`), including the "15 of 16 cells" count in Result 3 and Table C.6's caption
   (`:473`). Scope the Appendix A row honestly or fix Phase II.
3. `paper/refs.bib` `brenner2026grid` gives the title "Predictive statistics shape emergent world
   representations of grid walkers" for arXiv:2603.16689; `LITERATURE_AUDIT.md:97` records the same arXiv id
   with the title "Grid-World Representations in Transformers Reflect Predictive Geometry". One is wrong.
4. `:148`: "the block is ≈1 % of the matrix". For a 12×12 block in a V = 3000 PMI matrix that is 0.0016 % of
   cells or 0.4 % of rows. (1.4 % if the zeroed block is the 42 key spellings of
   `RESEARCH_LOG.md:318`, in which case the text should not say "12×12".)
5. `:217`: "rare rows carry 3–24× the error" holds only for r ≤ 0.01. From `results/phase4/analysis.txt` the
   rare/common KL ratio is 0.94–1.1 at r = 0.5, 1.4–1.5 at r = 0.1 and 1.4–2.2 at r = 0.03.
6. `:213`: "dose-response over five alignment levels gives Spearman +0.85" — `results/phase3/dose_analysis.txt`
   computes it over n = 15 runs at 5 nominal levels. With 5 distinct levels the effective n is 5; say so or
   quote a level-mean correlation.
7. `:272`: "an unexplained dip at 1T" contradicts Table C.5 (`:460–461`), where 1 T is the residual **peak**
   (+0.22, p = .083) and the drop is at 2 T (+0.10). Fix the token count or the word.
8. `:217`: "91 runs". `results/phase4/` holds 90 `log_*.txt` files; the extra appears to be the "smo" smoke
   run visible in `exposure_collapse.txt`. State what the 91st is.
9. `:217`: "the representational twin distance collapses after ≈250–2,400 rare exposures". The artifact's
   range is 222–2 425 (`results/phase4/exposure_collapse.txt`, timing block).
10. `:217`: "In the exposure-matched cross-arm comparison at r = .01, the unique state is learned better than
    the rare alias (row KL 0.0064 at 2.6k exposures vs 0.0105 at 2.3k)". The 0.0105 figure is the r = 0.003
    arm; labelling the comparison "at r = .01" is misleading. Neither number carries uncertainty, on 3–5 seeds.
11. `:66` abstract: "against 0.1–0.5 for PMI-type operators". The Karkada-window PMI reaches +0.54 for
    OLMo-7B (`results/phase5/operators.txt`).
12. `:151` quotes line|circle = +0.38 for the 15-key PMI (from `results/phase2/corpus/pmi15.txt`, 4-control
    set) and `:194` quotes +0.39 for the same statistic (from `results/phase5/operators.txt`, 5-control set).
    Symptomatic of M8.
13. `:85` glance box lists ECI = 0.04 among "what is established" without its interval; `:151` and
    `results/corpus/wiki/cluster_boot.txt` give [0.03, 0.44], a fifteen-fold range driven by C♭'s 38 mentions.
14. Phase-II nulls draw one fixed permutation bank (`np.random.default_rng(0)`, `phase2/behavior_fit.py:19`)
    and reuse it across every model, family and template, so the "15 of 16 cells" and "63/64 templates"
    counts are positively correlated across cells, not independent replications.
15. Figure 4 (`:190`) omits the reverse conditional and the document conditional from the plot although both
    are discussed in Result 4's text.
16. `:170` calls the OLMo-7B prompt-final p-values "raw free-null p = .004–.03"; the artifact key is
    `p_max_free`, i.e. already max-over-layers corrected (`results/phase2/geometry/olmo2_7b_symbol.json`).
    Also no correction across the six families (24 tests).
17. `:148`: "the canonical fifths partial rises from +0.27 at 0.11 B words to +0.63 at 3.12 B". It is
    non-monotone: 0.273 → 0.241 (0.21 B) → 0.353 → 0.429 → 0.446 → 0.632
    (`results/corpus/wiki/convergence.json`).
18. Two overfull hboxes in Table C.2 (`paper/main.log:2605, 2610`, tex lines 387–401): the bootstrap-CI
    column overruns the text block by 10.3 pt.
19. `paper/main.log` (Sep 1 10:57) is older than `paper/main.tex` and `paper/main.pdf` (Sep 2 01:57/01:58).
    The PDF text does match the tex — I checked the Result 7 numbers — but the build log in the tag does not
    correspond to the shipped PDF.

---

## What I tried to break and could not

For calibration, since several of the above are severe. I spot-checked ~150 numbers, far more than the 20 the
brief asked for, and found no transcription error anywhere:

* **Table C.2**, all 8 rows × 6 columns (ΔKL base/rich/TP, p-values, KL₀, both bootstrap CIs): exact against
  `wikipedia_v4_neutral{,_rich,_tp,_rich_tp,_docboot,_rich_docboot}.json`.
* **Table C.3**, all 36 DiD cells and both Wilcoxon p columns: exact. The "8 significant, four favouring
  OLMo and four opposing, 33 positive / 39 negative" tally in Result 8 is exactly right, as are 8/0/6/0 and
  7/3/8/2 and all four modulation×window corpus rows and all four corpus document/pair counts.
* **Table C.5**, all 12 checkpoints × 4 columns: exact against `ckpt_trajectory_v4.txt`. Soup-check numbers
  (0.096, 1.118, 1.117, 22 tensors) exact.
* **Table C.6**, all 16 rows × 9 columns: exact against `scorer_robustness.txt`; the derived counts 63/64 →
  58/64, "c|l moves by at most 0.11", "ECI falls by 0.05–0.25" all check.
* **Result 1**: every shard-cluster bootstrap number, the white-key rank 78/5040 (p = .0155), the withdrawn
  Poisson SDs, the PMI kernel in fifths order (8.64, 7.89, 7.59, 7.16, 7.07, 7.13, 6.48 — I re-derived the
  fifths reindexing from the stored semitone κ), M* saturation 1.974–1.999, median ρ = 1558.5, months
  spectrum 0.499/0.256/0.118/0.081/0.033/0.012, keys P₅ = 0.505 vs P₁ = 0.172.
* **Appendix B**: the mode permutation k ↦ 7k mod 12 = (0,7,2,9,4,11,6,1,8,3,10,5), the boxcar shares, the
  79.6 %, and all three partition counts reproduce exactly from first principles.
* **Result 6**: rare-spelling shares 0.0241 / 0.2165 / 0.4112 recomputed from the corpus unigrams.
* **Result 7 spelled view**: all 8 ΔKL/p, the KL₀ range, the leave-one-out Spearman range, the "5 of 8 ΔR²",
  the "one of eight / two of eight scorer disagreements", the "16 of 108 twin cells", and all three twin
  cosine ranges: exact.
* **`phase5/fingerprint.py` vs Appendix D.6**: the description is accurate. Fold-local tokenizer fitting,
  fold-local scaling on training pairs only, the held-out response never entering preprocessing, the corpus
  scaler refit inside every fold *and* every permutation, SHA-256 seed derivation with no salted `hash()`,
  the (b+1)/(B+1) estimator with the stated B — all match the code.
* The corpus column is not a disguised target-popularity feature: I regressed logC on the base aggregated
  theory design (R² = 0.748) and the residual is orthogonal to the class co-mention column marginal
  (r = −0.005), because `target_logfreq_sum` absorbs it. That alternative explanation of the residual gain
  is genuinely excluded.
* The Phase-III code-controlled vs uncontrolled partial choice (Appendix C.4 footnote, `:436`) is the correct
  one and is disclosed, even though the controlled measure would give the opposite sign for the
  aligned−permuted contrast.
* Appendix A is unusually complete and several rows are self-incriminating in a way that most papers avoid.

Also noted but not a finding: four base features (`source_is_flat`, `source_is_sharp`, `source_logfreq`, and
the source tokenizer residual) are constant within a source row and therefore cannot affect the log-softmax
normalized held-out prediction at all. The abstract's "frequency and tokenizer controls" is carried entirely
by the target-side columns. Worth a sentence in D.6.

---

## Structure and length

The restructure works: leading with §3–§6 and demoting Phase V to §8 is the right call, and §8's opening
paragraph (`:225`) sets expectations honestly. But 24 pages is not defensible for what survives.

* **§7 (synthetic) → appendix.** It is two negative controls for hypotheses that §6 already makes
  unnecessary, and it costs ~1.5 pages of dense text plus a five-panel figure. One paragraph in §6 plus
  Appendix C.4 would carry it.
* **§8.2 and §8.3 → half a page.** Both are null. Result 8 and Result 9 currently get ~2 pages of prose and
  two figures to say "nothing here". Table C.3, Figure C.1, Table C.5 and Figure C.2 already carry the
  detail.
* **§4's second and third paragraphs (`:168`, `:170`) → appendix.** They are a list of eight negative or
  bounded findings with no through-line; the reader needs only "hidden-state geometry is not where the
  positive claim lives" plus a pointer.
* That gets you to ~18 pages with the same content. If you want a conference-length paper, the alias result
  (§4) plus the operator comparison (§6) plus one page of §8 is a clean 9-page submission and the strongest
  version of this work.
* Glance box vs body: consistent everywhere I checked except M9 ("neither privileged") and M5 (the abstract's
  "0.1–0.5 for PMI-type operators" silently drops HELP at 0.64–0.68).
* The "What is not claimed" list (`:94`) is the best paragraph in the paper and should survive any cut.

---

## Verdict

**MAJOR REVISION.**

### Five things to fix, ranked

1. **Report the leave-one-source-row-out jackknife of Result 7 and rewrite the abstract, glance box, central
   claim and contribution 4 around what survives it.** As it stands, OLMo-2-7B's headline gain is 94 % one
   row (C♭ major, 38 corpus mentions) and OLMo-1B's is 63 %; only Gemma's effect is distributed. The
   artifacts already contain everything needed (B1).
2. **Name the bootstrap interval type, report the pivotal interval alongside the percentile one, and raise
   B above 300.** The single most-repeated negative caveat in the paper — "cluster-bootstrap support only for
   Gemma" — reverses to three of four models under the pivotal interval from the same draws, because the
   percentile interval inherits a 28–40 % shrinkage toward zero (M1).
3. **Fix the two false statements in Result 2 and §4**: the glyph RSA *falls* from +0.85 to +0.28–0.67 under
   respelling (Table C.1 contains the contradicting number), the four values are unnulled six-layer maxima,
   and the respelled strings have 1/2/21/38 corpus mentions against 6348/3101/4290/1544 — so glyph is
   confounded with a ~1000× frequency drop, not just "rare" (B2). Separately, "circle or line adding ≤ 0.04
   in every cell" is violated in 30 of 660 prompt-final cells with a maximum of +0.104, and the quoted
   R²_cv range matches no computable summary (B3).
4. **Rescope the matched-operator control and the operator taxonomy.** Report Pearson(N, Nᵀ) = 0.977 and
   state that "directionality contributes nothing" is a property of this corpus's near-symmetric co-mention
   matrix; report that the helper-word factorization — an association operator, and the one Karkada's theory
   predicts — sits at 0.64–0.68 and keeps twins farthest, so the association/conditional-row dichotomy does
   not survive its own within-class spread; and position the finding against the standard first-order vs
   second-order similarity distinction (M2, M5, M6).
5. **Fix reproducibility and multiplicity.** `MIRROR_MANIFEST.md` does not exist; `results/corpus_merged/*.npz`
   and `results/multictx/*/*.npz` are `.gitignore`d yet Results 1 and 3 depend on them; the paper's stated
   orthographic control family is used by no analysis and four different sets are used across sections
   without disclosure; Bonferroni-36 is the smallest defensible family when 288 Wikipedia held-out tests were
   run (B4, M7, M8).

### Honest assessment

This is a careful, unusually self-critical piece of work whose numerical hygiene is better than almost
anything I review: I checked roughly 150 numbers across nine Result boxes, the abstract and four appendix
tables against the stored artifacts and re-derived the entire Fourier appendix from scratch, and found not
one transcription error. Appendix A is a genuine audit trail rather than a limitations paragraph. The
orthographic-alias result (§4) is the real contribution — a clean, transportable failure mode for Fourier
representation analysis, correctly derived, with a causal intervention and an exhaustive base-rate
calculation — and it deserves to be published. The operator comparison (§6) is a good empirical observation
undersold by an over-strong taxonomy and undercut by a control (symmetrized conditional) that is
mathematically near-identical to the thing it controls. §8 is the problem: after two rounds of correction the
positive Phase-V result has shrunk to a 5.9–13 % KL reduction whose largest component, in two of the three
"positive" models, is a single held-out row for a key with 38 mentions in 3.1 billion words — and the paper's
own conservative framing does not catch this because it reports template robustness but never row robustness.
The paper is honest about being weak; it is not yet honest about *where* the weakness is. Cut §7 and §8.2–8.3
to appendices, rebuild §8.1 around the jackknife, fix the four false statements, and this is a solid paper at
a strong interpretability venue — an ICLR/COLM workshop or the ICML Mechanistic Interpretability workshop as
it stands after revision, or a credible main-track ICLR/COLM submission if §4 is promoted to the lead
contribution and §8 is reduced to the honest one-paragraph null it has become. In its current 24-page form,
with §8 in the abstract and the title, it would be rejected at a main track for overclaiming a result that
one held-out row controls.

---

## Addendum — post-tag artifacts that appeared during this review

`results/phase5/ridge_lambda_sensitivity_v4.txt` and `results/phase5/operator_eci_boot_v4.txt` (timestamps
02:26–02:28, untracked, produced by a concurrent process after `paper-v2`) are not part of the manuscript I
reviewed, but they bear on two findings and I read them rather than pretend otherwise.

* They **support** M11's premise and sharpen it. ΔKL is flat between λ = 0.01 and λ = 1 (base OLMo-1B
  +0.0038 → +0.0041), confirming the estimator is effectively OLS, but it then **grows monotonically with
  λ**: at λ = 100 every cell is positive, including Qwen (+0.0060 base window, +9.4 % of KL₀, versus
  −0.0001 at λ = 1). So the paper's "three versus one" split — the sentence that appears in the abstract,
  the glance box, the central claim and contribution 4 — is not stable under the one estimator hyperparameter
  that was never tuned or reported. That strengthens M11 from "undeclared" to "load-bearing and undeclared",
  and it compounds B1 and M3: the model-dependence the paper reports is contingent on one held-out row, on
  which of three ρ ≥ 0.93 predictors is used, and on λ.
* The ECI bootstrap partly **answers** M2 in the paper's favour on one point: subsampling B's row to C♭'s 48
  events still gives directional ECI 0.545, so the conditional-row/PMI ECI gap is not a pure sample-size
  artifact. It does not address M2's actual claim, which is that Pearson(N, Nᵀ) = 0.977 makes the
  directional/symmetrized contrast near-vacuous; that number still belongs in the paper.
* Both files also illustrate B4 from the other side: analyses the manuscript will presumably cite are being
  generated after the tag the Reproducibility section points at.
