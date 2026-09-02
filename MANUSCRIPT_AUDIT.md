# MANUSCRIPT_AUDIT.md — hardening pass on *One Concept, Multiple Geometries* (2026-08-29 22:00 → 2026-08-30)

## Submission verdict: **CENTRAL CLAIM REVISED** (and, after revision, READY FOR ADVERSARIAL REVIEW)

The primary empirical claim survives every new check and is strengthened (target-prior control, document-cluster
bootstrap, 5,000-relabeling nulls). The *conceptual* framing had to change: the "symmetric vs directional" contrast
does not isolate directionality — a symmetrized conditional built from the same 40-word counts predicts behaviour as
well as the directional one, and a PMI built from the same counts predicts poorly. The paper now claims
*conditional-row vs association construction*, not *directional vs symmetric*. Title kept; subtitle, abstract,
central-claim box, §6, Figure 4 and the discussion rewritten accordingly.

### The seven explicit questions
1. **Was the 300-null / p<.001 problem real?** YES. Every Phase-V permutation p was computed as b/B with no +1: residual
   correspondence and twin nulls B=2,000, held-out (ΔCV/ΔR²/ΔKL) nulls **B=300**, checkpoint nulls B=500. "p<.001" for the
   held-out ΔKL was printed from 0.000 at B=300 — floor 1/301 = 0.0033 — and was not justified. Fixed: `phase5/fingerprint.py`
   now uses p=(b+1)/(B+1) with B=5,000 (Wikipedia) / 2,000 (other corpora); `phase5/ckpt_fingerprint.py` B=2,000 with the same
   estimator; all Phase-V JSONs/txt regenerated. Flagship cells (target-aggregated modulation × window/document, four
   models): b=0 of 5,000 in six of eight cells (p ≤ 2×10⁻⁴), p ≤ 8×10⁻⁴ in all eight. Phase I/II nulls (B=2,000 / 500–1,000,
   b/B) are now described as such in Appendix D; the +1 correction moves them by ≤ 0.002 and the text quotes the corrected
   values (7B prompt-final p .006–.03; key-name 7B p .046/.018).
2. **Does matched same-count directionality survive?** NO — directionality per se contributes nothing. From the same
   ordered 40-word count matrix N: symmetrized conditional (rows of N+Nᵀ) RSA with behaviour +0.82/+0.78/+0.80/+0.81
   (1B/Gemma/Qwen/7B) vs directional +0.80/+0.78/+0.79/+0.80 vs reverse +0.83/+0.78/+0.80/+0.81; PMI from the same counts
   +0.12/+0.16/+0.10/+0.24 (Karkada L=16 PMI +0.39/+0.46/+0.40/+0.54). Twin ECI: conditionals 0.56–0.60, matched PMI 0.05.
   Held-out ΔKL (target-aggregated E): symmetrized +0.039/+0.030/+0.016/+0.040 ≈ directional +0.037/+0.029/+0.016/+0.036;
   reverse +0.029/+0.027/+0.013/+0.030; matched PMI +0.021/+0.014/+0.012/+0.029 (all p ≤ .001). The contrast is
   conditional-row construction (row-normalized local co-mention rows, row-divergence geometry) vs association (PMI,
   marginal-normalized); the story was changed accordingly. Files: `results/phase5/operators.txt`,
   `results/phase5/fingerprint/matched_{sym,rev,pmi}*.json`, `results/phase5/cond_wikipedia_matched_*.npz`.
3. **Does held-out ΔKL survive a training-row target-prior control?** YES, and it grows: with each target's mean
   log-probability over the 14 training rows added to the theory model (recomputed inside every fold), target-aggregated
   E ΔKL = +0.041/+0.035/+0.025/+0.047 (window), +0.047/+0.042/+0.031/+0.049 (document), all p ≤ 2×10⁻⁴; with the rich
   model too +0.027/+0.022/+0.011/+0.025 and +0.035/+0.029/+0.018/+0.029 (p ≤ 8×10⁻⁴). The harmonic family, which lost its
   gain under the rich model, regains it with the prior. Files: `results/phase5/fingerprint/wikipedia_v3_neutral_tp.json`,
   `..._neutral_rich_tp.json`.
4. **Does the cluster bootstrap change Result 1 materially?** Merged family: no (fifths +0.66 [0.49, 0.71]; circle|line
   +0.58 [0.39, 0.64]; line|circle −0.07 [−0.20, 0.07]; P(circle>line)=1.00). Canonical family: the Poisson bootstrap's
   P(circle>line)=0.87 falls to 0.63 and the intervals are wide (circle|line [0.05, 0.55], line|circle [−0.08, 0.58]); the
   paper already led with the merged family and now says so with the cluster intervals; Poisson SDs withdrawn everywhere.
   The 15-spelling ECI 0.04 has a heavy upper tail ([0.03, 0.44]) driven by the 38 mentions of C♭ major — stated.
   File: `results/corpus/wiki/cluster_boot.txt` (41 shards, B=1,000). The flagship ΔKL has a document-cluster bootstrap
   (9,168 documents, B=300; `results/phase5/fingerprint_wikipedia_v3_neutral_docboot.txt`): 95% intervals exclude 0 in all
   eight E cells (Qwen×document narrowly: [+0.0003, +0.024]).
5. **Does length-normalized scoring change Result 3?** NO. line|circle survives in all 16 model×family cells (E:
   +0.44→+0.37, +0.54→+0.43, +0.52→+0.50, +0.56→+0.45); significant templates 63/64 → 58/64; circle|line moves ≤ 0.10;
   ECI falls 0.05–0.25 but twins remain far from identified. Table C.6; `results/phase5/scorer_robustness.txt`.
6. **Bibliography corrections** (all entries re-verified against arXiv/publisher metadata): Karkada et al. → Daniel J.
   Korchinski, Andres Nava, ICML 2026; Sadek & Bakarji → Najla Sadek, PMLR 303:1–13; Prieto et al. → Lucas Prieto, Edward
   Stevinson, Melih Barsbey, Tolga Birdal, Pedro A. M. Mediano; Singh & Chopra → Simardeep Singh, Paras Chopra; Hu et al.
   → Zhimin Hu, Lanhao Niu, Sashank Varma (was "Jiayi Hu, Xiaoxuan Niu"); Zhao & Thrampoulidis 2025 title → "Geometry of
   semantics in next-token prediction: How optimization implicitly organizes linguistic representations"; Feucht et al.
   full author list (12); Fu et al. → COLM 2026; Engels et al. ICLR 2025; Nanda et al. ICLR 2023; Zhong et al. NeurIPS
   2023; Kandpal et al. ICML 2023; Gurnee & Tegmark ICLR 2024; Chuan et al. NCA 32:1023–1036 (2020) with DOI; Huang et al.
   IUI 2016 pp. 241–250 with DOI; Moss et al. DOI added; Gemma-2/Qwen2.5 author lists corrected to "Team + first authors";
   Levy & Goldberg pages added; unused/placeholder entries (du2026symmetry, karkada2025closedform, park2026information)
   removed. `paper/refs.bib`.
7. **Final exact central claim** (paper §1, Central claim box): *Different corpus operators over the same concept family
   preserve different structures and equivalence relations. For keys, PMI-type association statistics identify the
   enharmonic spellings and carry robust periodic fifths structure; conditional-row operators built from the same local
   counts separate the spellings that association identifies, whether directional or symmetrized, while remaining
   fifths-smooth. Model behaviour — restricted next-key distributions over a fixed candidate set — lies on the
   spelling-sensitive open line, and it is the local conditional operators, not the association operators tested, that
   align with it and add held-out predictive value beyond explicit circle, line, orthography and frequency baselines
   and beyond a training-row target prior — generically across models and corpora; whether any part of it is specific
   to a model's own training data is left open.*

### Item-by-item status
| # | Item | Status | Evidence / change |
|---|---|---|---|
| 1 | P-value audit | **changed** | See Q1. Appendix D.6 states every estimator and B; Appendix A row added; audit note in §7 discloses the 300-permutation origin of the earlier "p<.001". |
| 2 | Matched symmetric/directional operators | **changed (central story)** | See Q2. §6 defines the three matched operators before comparison; Result 4 and Figure 4 report them; abstract/subtitle/central claim/discussion reframed to conditional-row vs association. |
| 3 | Training-row target-prior control | **checked, changed** | See Q3; reported in Result 5, glance box, abstract, Table C.2 column. `--targetprior` in `phase5/fingerprint.py`. |
| 4 | Central regression fully specified | **changed** | Appendix D.6: response, predictor, features, z-scoring (with the leakage caveat), ridge λ=1 unpenalized intercept fixed a priori, LOO procedure, softmax conversion, KL direction and 1e-12 floor, aggregation of templates, target-aggregated construction, permutation and bootstrap procedures. Templates listed in D.4; model repositories/revisions and the unsloth Gemma mirror in D.5. |
| 5 | Poisson bootstrap replaced | **changed** | Shard-cluster bootstrap (`scripts/corpus_cluster_boot.py`) for Result 1; document-cluster bootstrap for the flagship ΔKL; Poisson SDs and P=0.87 withdrawn (Appendix A row corrected; RESULTS.md annotated). `--nboot` default 0. |
| 6 | Scorer vs view naming | **changed** | "enharmonic-merged behavioural scorer" (families B–D) vs "12-class target-aggregated view" (diagnostic projection of the scored distribution); defined in §2.2 and §7; stated not to be a scoring rule for modulation. |
| 7 | Length-normalized robustness | **checked, changed** | Q5; Result 3 robustness sentence + Table C.6. |
| 8 | Template robustness | **checked, changed** | Flagship gain significant in every leave-one-template-out aggregate (all models) and in 3 of 4 single templates (template 3 n.s. for Qwen, 7B); §7 statement + Table C.7. Hold-out explicitly stated as source-row, not template. |
| 9 | Overclaims / language | **changed** | All eight listed phrases replaced ("far better than symmetric PMI"; "follows accidental-glyph/spelling structure rather than black-key membership"; "these two mechanisms are insufficient"; seam-concentrated distinction instead of "not a different metric"; "identifies enharmonic twins and carries robust periodic structure"; "most closely aligned with the next-key behavioural task among those tested"; "in this domain … can depend"; "separates the enharmonic spellings that the symmetric operator identifies"). |
| 10 | "Behaviour" defined as restricted-candidate distribution | **changed** | Abstract, §2.2, Appendix D.4; COND written as ordered key-mention conditional p(j|i), never as the next-token distribution. |
| 11 | Reframe symmetric/conditional story | **changed** | Q2; formulation adopted almost verbatim from the brief. |
| 12 | Figure 5 selectivity | **changed** | 2×2 figure: (a,b) target-aggregated ΔKL, (c) spelled ΔKL, (d) spelled ΔR²; disagreement visible. Rank ΔCV stays in the audit note. |
| 13 | Cross-corpus figure to appendix | **changed** | Figure 6 → Figure C.1 in Appendix C.3; main text keeps a compact statement; "seven combinations … more than a null comfortably produces" removed. |
| 14 | Gemma in operator comparison | **changed** | Added to `phase5/operators.py` and Figure 4 (+0.46 PMI, +0.67 helper, +0.78 conditional). |
| 15 | Multiple-testing language | **changed** | Audit note in §7: no cell pre-specified as primary; all 36 cells reported; modulation emphasized post hoc; flagship p ≤ 2×10⁻⁴ survives Bonferroni×36, rich-model p's do not; no new correction scheme. |
| 16 | Bibliography | **changed** | Q6. |
| 17 | Karkada/PMI wording | **changed** | "Karkada-style helper-word factorization applied to the PMI matrix"; SGNS "related through the factorization of shifted PMI". |
| 18 | Abstract | **changed** | ~330 words (from 569), six short paragraphs; specificity-cell details, DCLM counts and checkpoint numbers removed; subtitle changed to "Corpus operators recover different structures, and task-aligned conditionals predict held-out model behaviour — with musical keys as the instrument". |
| 19 | Model audits ≠ human review | **changed** | Intro no longer cites the reviews as evidence; Appendix A calls them model-assisted audit passes; "audit trail is part of why we believe it" replaced. |
| 20 | Self-contained reproducibility | **changed** | Appendix D (spellings/aliases, matching regex and sentence-initial rule, window weighting, conditional counting, prompts, candidate formatting, tokenizer handling, sequence-probability convention, template averaging, model revisions, hidden positions, ridge/CV, nulls, bootstrap units). |
| 21 | Structure | **no change required** | Order kept; cross-corpus detail moved to appendix. |
| 22 | Claim hierarchy | **changed** | Contributions and glance box follow primary / conceptual / methods / falsification / secondary / not-claimed. |
| 23 | Stop condition | **respected** | No new models, corpora, keys, patching or synthetic sweeps; the only new computations are the seven listed checks (plus a rescan of Wikipedia for per-document counts, which reproduced the aggregate matrices exactly). |

### Unresolved / caveats
- The cross-corpus fingerprints were rerun with B=2,000 for consistency; the DiD tables (per-row KL differences) do not
  depend on nulls and are unchanged. Reported cross-corpus p-values were refreshed from the reruns.
- The Phase-II geometry and behaviour nulls (B=500 / 1,000, b/B) were not rerun; the paper states their B and estimator
  and quotes the (b+1)/(B+1) values, which differ from the originals by ≤ 0.002.
- Feature z-scoring in the ridge uses all pairs' feature values (not responses); disclosed in D.6.

---

# Round 4 (2026-08-30 → 09-02): Opus 5 review and the v4 correction → paper-v2

## Verdict: **CENTRAL NUMBER CORRECTED — PAPER SURVIVES WITH THE CENTRAL CLAIM NARROWED**

The round-4 Opus 5 review (`notes/review_opus5_v4.md`; brief `notes/review_brief_v4.md`) found that the flagship
regression's circle predictor was chromatic distance (B1) and that the target-aggregated line feature was a class
centroid (B2). Both confirmed; a further audit found global predictor standardization and process-dependent RNG
seeds. All four were corrected (`V4_CORRECTION_REPORT.md`), the full Phase-V chain was recomputed, and every
inherited number was compared (`results/phase5/v3_v4_comparison.{csv,md}`). Item-level disposition of all 120
review findings: `notes/OPUS5_V4_VERIFICATION.md` (63 confirmed, 12 reviewer errors, 4 partial, 1
provenance-limited, 40 resolved from the final v4 artifacts).

## What changed in the manuscript

- Headline: "14–34 % held-out KL reduction beyond circle, line, orthography and frequency, all four models" →
  "5.9/13.0/−0.4/7.3 % under the corrected rich baseline; positive by relabeling in OLMo-1B, Gemma and OLMo-7B,
  null in Qwen; cluster-bootstrap support only for Gemma; document conditional weaker (OLMo-1B, Gemma only)".
  Subtitle, abstract, glance box, central claim, contribution (2), Result 5, audit note, discussion, limitations.
- Result 4: matched-operator held-out regressions restated (PMI null; reverse n.s. in two models); the document
  conditional's "paradox" conjecture dropped.
- Result 6 / §8: "generic across corpora; specificity inconclusive (10/72, all OLMo-favouring)" → heterogeneous
  direct gains; DiD 8/72 split 4/4; fingerprint neither established nor suggested; Table C.3 and Figure C.1 rebuilt
  from v4 with five thinning seeds (mean ± SD).
- Result 9 / §10: the 294B-first-significant / stable-from-4T reading withdrawn; Table C.5 and Figure 7 rebuilt.
- Table C.2 (base/rich/target-prior/bootstrap columns), Table C.7 (all cells), Figure 5 (v4 inputs, caption
  corrected: scorers disagree in 1/8 aggregated and 2/8 spelled cells), D.6 rewritten (feature definitions,
  fold-local preprocessing, deterministic seeds).
- Appendix A: rows added for B1, B2, global standardization, process-dependent seeds; fingerprint row → retracted;
  target-prior row → bounded; spelled-view row updated.
- Independent of v4 (review items verified against raw artifacts): two merging experiments separated (12-key
  merged target vs 15-key merged scorer, §5); corpus-size non-convergence disclosed (§3); "14 of 16" → 15 of 16;
  Poisson SDs attributed to the correct family; black-key RSA floor +0.56; slash-notation count 3/21; 7B top-1
  counts as cells of 60; last-token line nulls stated; subspace shares 0.53–0.76 / 0.46–0.73; Gemma context
  contrast −0.026; max-over-layers p range 0.09–0.21; c|l movement 0.11; §9 wording (exponents, unique-state
  arm, thresholds, rare shares, Cb/B JS provenance); Sadek et al. venue; Fig 4 caption layers stated.
- Reproducibility: mirror tag `paper-v2`; `paper-v1` immutable; `V4_CORRECTION_REPORT.md` referenced.

## Process record

Contract frozen before implementation; independent mechanical verifier and independent reviewer (two NOT_READY
rounds, then READY) gated the launch on commit a56ad3e; v3 artifacts byte-identical; invalid preview outputs
quarantined outside the repository; `phase5/v4_provenance.py --verify` records VALID at chain completion
(`results/phase5/rerun_v4.log`); re-running it on the dirty paper-editing tree fails closed by design.

## Unresolved / caveats

- The corrected held-out effect is small relative to the residual KL and is not robust to the document-cluster
  bootstrap except in Gemma; the paper says so and builds no further claim on it.
- The checkpoint residual trajectory is now weak; retained as suggestive, one run.
- The adversarial Sol review of paper-v2 has not yet been run (to be launched separately on the frozen tag).

## Restructure (2026-09-02, after the v4 integration)

The paper-v1 architecture routed the reader to Phase V as "the quantitative core" and "the central number"; after v4
that number is small and model-dependent, so the manuscript was re-ordered to lead with what is robust. New order:
§1 Introduction · §2 Setting · §3 corpus PMI · §4 orthographic alias · §5 behaviour on the line · §6 operators
(the main result) · §7 synthetic controls (Results 5–6, Figure 5) · §8 "Beyond correspondence" with §8.1 held-out
test (Result 7, Figure 6, audit note), §8.2 other corpora (Result 8), §8.3 one pretraining run (Result 9) · §9
Discussion. The checkpoint figure moved to Appendix C.5 as Figure C.2 with the provenance note; Results 5/6/7/8/9 of
the v4 draft are now 7/8/5/6/9. Subtitle, abstract, glance box, central-claim box, contributions and roadmap were
rewritten to the same order; no number changed.

---

# Round 5 (2026-09-02): independent Opus 5 and Sonnet 5 reviews of paper-v2, plain-language rewrite

Reports: `notes/review_opus5_v5.md` (4 BLOCKING, 14 MAJOR, 19 MINOR; MAJOR REVISION) and `notes/review_sonnet5_v5.md`
(0/5/9; MAJOR REVISION); brief `notes/review_brief_v5.md`. Both re-verified the Phase-V numbers (≈150 and ≈30 checks,
no transcription errors). Verdict after integration: **PAPER SURVIVES BUT THE HELD-OUT RESIDUAL IS NOW STATED AS
FRAGILE.**

## Findings acted on (all verified against artifacts before editing)

- **Row jackknife (Opus B1).** `phase5/row_jackknife.py` → `results/phase5/row_jackknife_v4.txt`: C♭ major (38 mentions,
  48 co-mention events) carries 66/44/94 % of the rich-window gain in OLMo-1B/Gemma/OLMo-7B; without it the gains are
  +0.0013/+0.0037/+0.0002; Qwen's null flips to +0.0007 without C♯. Result 7, abstract, glance, central claim,
  contribution 4, Limitations and Appendix A restated; new Table C.8.
- **Interval type (Opus M1).** Percentile intervals are shifted toward zero by 28–40 %; pivotal intervals from the
  same 300 draws exclude zero for 1B/Gemma/7B. Both are now reported (Result 7, Table C.8, D.6); "support only for
  Gemma" removed from every headline.
- **Ridge λ (Sonnet M1, Opus M11).** `phase5/ridge_lambda_sensitivity.py` → `ridge_lambda_sensitivity_v4.txt`: λ=1 is
  effectively OLS (base eigenvalues 1.6–730; rich min 0.006); pattern flat for λ ≤ 1, gains inflate at λ ≥ 10 (all
  cells positive at λ = 100). Disclosed in D.6; λ = 1 kept as the pre-specified, conservative end.
- **Conditional-row ECI (Sonnet M2/M4).** `phase5/operator_eci_boot.py` → `operator_eci_boot_v4.txt`: document-cluster
  intervals [0.47,0.67]/[0.49,0.68]/[0.53,0.71] include or graze 0.5; marginals-only null gives 0.84 ± 0.05; B row
  subsampled to 48 events → 0.55. "Keep twins apart" → "do not identify twins" everywhere.
- **Matched-operator control (Opus M2/M3/M5/M6).** Pearson(N, Nᵀ) = 0.977 stated in §6; "directionality contributes
  nothing" rescoped to this corpus; helper-word factorization named as an association operator that lands between the
  groups; first-order/second-order (syntagmatic/paradigmatic) framing added with Schütze & Pedersen 1993 and Sahlgren
  2008; the p-value spread across three ρ ≥ 0.93 predictors stated as estimator variance.
- **False statements (Opus B2/B3).** Result 2: glyph RSA falls from +0.84…+0.86 to +0.51/+0.43/+0.67/+0.28 (six-layer
  maxima, no selection null), respelled-name frequencies 1/2/21/38 vs 6,348/3,101/4,290/1,544 stated. §4: layer-mean
  R²_cv 0.06–0.45 (single layers −0.04…0.76), circle/line adds ≤ 0.03 to layer means, 30/660 single layer-cells
  exceed 0.04 (max +0.10); p-values labelled layer-corrected.
- **Reproducibility (Opus B4).** MIRROR_MANIFEST.md lives in the mirror, not the working repo; the excluded
  multi-context and merged-count files are named in §Reproducibility.
- **Multiplicity (Opus M7):** 288 Wikipedia held-out tests stated; nothing survives. **View choice (M9):** declared post
  hoc; "neither privileged" dropped. **Twin exclusion (M10):** stated in Result 7 and D.6. **Harmonic bootstrap (M4):**
  two significantly negative cells reported; bootstrap coverage (16 cells) stated. **Synthetic scope (M12):** abstract
  and Result 5 rescoped to the toy world. **Fifths-specific alias base rate (M13):** 5.0 % / 0.9 % added beside 31 %.
  **Control sets (M8):** new Appendix D.7. **Figure 6 (M14):** percentile intervals drawn. **Identifiability
  (Sonnet M3)** and **memorization alternative (Sonnet §3)** stated in Result 8 and Discussion. Minor items: status
  macro "Bounded screen" → "Bounded"; convergence sequence given non-monotonically; 1T peak / 2T drop; 90 runs;
  222–2,425 exposures; r = .003 arm labelled; dose-response n stated; PMI range 0.1–0.54; ECI interval in glance;
  Gemma mirror fidelity caveat; Phase-II b/B and shared permutation bank stated; brenner2026grid title fixed in
  LITERATURE_AUDIT.md.

## Not acted on

- Raising the document bootstrap above B = 300 (Opus fix 2): not rerun; the disagreement between interval types is
  reported instead. Moving §7 and §8.2–8.3 to appendices (Opus structure): §7 kept as a short section, §8.2–8.3 kept as
  compressed subsections; further cutting is a venue decision.
- Appendix A trim (Sonnet): kept; the audit trail is the point of this preprint.

## Prose

At the user's request the whole manuscript was rewritten for plain reading (abstract, glance, introduction, every
Result box, section openings, discussion): idea before number, one claim per sentence, terms explained where first
used. No number changed except as listed above.
