# Adversarial review brief — "One Concept, Multiple Geometries", tag paper-v2 (commit 24754d7)

You are an independent, adversarial reviewer of a research manuscript. Review it as a hostile-but-fair expert
referee would for a strong ML/interpretability venue: find what is wrong, overclaimed, unsupported, internally
inconsistent, misleading, or badly organized. Do not praise it.

## Context you should know

- `paper-v1` (tag `fb3e3e8`) reported a "14–34 % held-out KL reduction beyond circle, line, orthography and
  frequency" in all four models. An earlier review found that its "circle" predictor was actually the chromatic
  cyclic distance and that merged target classes were given an arithmetic line centroid. Everything in Phase V was
  recomputed with a corrected feature set (`V4_CORRECTION_REPORT.md`); the corrected effect is small and
  model-dependent. `paper-v2` is the corrected and restructured manuscript. `git show paper-v1:paper/main.tex` gives
  the old draft if you want to check that no v1 claim survives.
- Every Phase-V number in `paper/main.tex` has already been mechanically verified against the v4 artifacts (about
  430 checks, `notes/OPUS5_V4_VERIFICATION.md` covers the earlier ledger). Spot-check a sample of your own choosing
  (at least 20 numbers across Results 1–9, the abstract and Appendix C) but spend most of your effort on the
  science, statistics, logic and framing, not on re-transcribing tables.
- The 19 tests in `tests/test_phase5_v4.py` pass and every `results/phase5/fingerprint/*_v4*.json` validates.
  Do not re-run them and do not re-run any analysis chain.

## What to review

- `paper/main.tex` (source of record) and `paper/main.pdf` (24 pages). Read the whole paper.
- Results: `results/phase5/fingerprint/*_v4*.json` (`dkl`, `dkl_p`, `kl.theory`, `r2gain`, `resid_r`, `docboot`,
  `feature_definitions`, `loo.fold_scalers`, `randomization`), `results/phase5/crosscorpus_compare_v4_*.{json,txt}`,
  `results/phase5/thinning_seed_variance.csv`, `results/phase5/ckpt_trajectory_v4.txt`, `results/phase5/v3_v4_comparison.md`,
  `results/phase5/operators.txt`, `results/corpus/wiki/{report.json,seam.txt,cluster_boot.txt}`, `results/phase2/*`,
  `results/phase3/*`, `results/phase4/*`. `paper/make_figs.py` shows which files feed each figure.
- Code: `phase5/theory_features.py`, `phase5/fingerprint.py`, `phase5/crosscorpus_compare_v4.py`,
  `phase5/ckpt_fingerprint.py`, `phase5/operators.py`, `phase2/*.py`, `scripts/corpus_cluster_boot.py`. Check that
  Appendix D describes what the code does (features, fold-local scaling, ridge, hold-out, null, p estimator,
  bootstrap, seeds).
- Logs and audit trail: `MANUSCRIPT_AUDIT.md`, `V4_CORRECTION_REPORT.md`, `PHASE5_RESULTS.md`, `PHASE5_LOG.md`,
  `PHASE2_RESULTS.md`, `RESULTS.md`, Appendix A of the paper.

## What we want from you

1. **Claim audit.** For the abstract, "Results at a glance", the central-claim box, contributions, and every Result
   box (1–9): is each claim supported at the stated strength? Flag overclaims, hedges that hide a negative result,
   and negative results that are buried. In particular judge whether §8 (held-out test, other corpora, checkpoints)
   is now stated honestly, and whether the paper's headline (operator-dependent geometry + the orthographic alias)
   is actually earned by the evidence in §3–§6.
2. **Statistical audit.** The relabeling nulls (right null? conditioning on the observed corpus?), the finite-sample
   p estimator, multiplicity across 36-cell grids, the document-cluster bootstrap and the tension between "significant
   by relabeling" and "cluster interval includes zero", the training-row target prior, the matched-operator control,
   leave-one-source-row-out with 15 rows, the DiD specificity test and its "directionless" verdict, the thinning-seed
   variance, the checkpoint residual test. Is the ridge with fixed λ=1 on ~154 training pairs effectively OLS, and
   does the paper say so? Is anything post hoc that is not declared as such?
3. **Feature-set audit.** Is the corrected v4 theory baseline (Appendix D.6) actually the "scientifically strongest
   defensible" baseline, or is something obvious missing or double-counted? Are the set-valued merged-class line
   features a principled treatment of a log-sum-exp target class? Could the residual gain in three models be
   explained by something the baseline still omits?
4. **Logic and framing.** Does "operator-dependent geometry" follow from PMI vs conditional rows, or is it partly
   definitional? Does the matched-operator control really separate construction from directionality? Is the
   orthographic-alias result (§4) as general as the paper says? Is the synthetic evidence (§7) used for more than it
   can bear? Is "One concept, multiple geometries" a finding or a truism, and does the paper make that case?
5. **Structure and length.** The paper was just restructured to lead with §3–§6 and demote Phase V to §8. Does the
   new order work? What should be cut, moved to an appendix, or shortened? Is the glance box / abstract / central
   claim consistent with the body and with each other? Is 24 pages defensible?
6. **Presentation.** Figures (readability, labels, consistency with text), tables, notation, Appendix A completeness,
   references (`paper/refs.bib` — anything wrong or fabricated), typos.

## Rules

- READ ONLY. Do not modify, create, delete or move any repository file except your own report. Do not run GPU jobs
  or long computations; small Python checks on results files (seconds) are fine.
- Every finding must cite the file and line (tex) or file/key (results/code) it rests on.
- Grade every finding: BLOCKING (claim false or unsupported as stated), MAJOR (materially misleading or a real
  statistical problem), MINOR (presentation, wording, small inconsistency).
- End with a one-line verdict — ACCEPT / MINOR REVISION / MAJOR REVISION / REJECT — followed by a ranked list of the
  five most important things to fix and, separately, your honest one-paragraph assessment of how good the paper is
  and what venue it fits.
- Be concrete and blunt. No summaries of what the paper does.

Write the full report as Markdown to the output file named in your instructions.
