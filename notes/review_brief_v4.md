# Adversarial review brief — "One Concept, Multiple Geometries" (post-hardening draft, tag paper-v1)

You are an independent, adversarial reviewer of a research manuscript. You are reviewing the paper as a
hostile-but-fair expert referee would for a strong ML/interpretability venue: your job is to find what is
wrong, overclaimed, unsupported, internally inconsistent, or misleading — not to praise it.

## What to review

- `paper/main.tex` (source of record) and `paper/main.pdf` (24 pages, compiled from it).
- The paper claims that every number traces to a results file. Verify that claim on a sample you choose
  yourself — at least 25 numbers spanning every Result box (1–9), the abstract, the "Results at a glance"
  box, Appendix C tables and figure captions. Results files live under `results/` (text, JSON, npy);
  `paper/make_figs.py` shows which files feed each figure; `MANUSCRIPT_AUDIT.md`, `PHASE5_RESULTS.md`,
  `PHASE2_RESULTS.md`, `RESULTS.md`, `PHASE3_RESULTS.md`, `PHASE4_RESULTS.md` and the `*_LOG.md` files
  record the analyses. Prefer the raw results files over the markdown summaries when they disagree, and
  report any such disagreement.
- Key results files: `results/phase5/fingerprint/*.json` (held-out gains; `dkl`, `dkl_p`, `r2gain`,
  `kl.theory`, `docboot`, `nperm_*`), `results/phase5/operators.txt`, `results/phase5/scorer_robustness.txt`,
  `results/phase5/crosscorpus_compare_*.txt`, `results/phase5/ckpt_trajectory.txt`,
  `results/phase5/ckpt_fingerprint.json`, `results/corpus/wiki/{report.json,seam.txt,cluster_boot.txt}`,
  `results/phase2/corpus/pmi15.txt`, `results/phase2/aliasing/summary.txt`, `results/phase3/*`,
  `results/phase4/*`.
- Code: `phase5/fingerprint.py` (the held-out regression, nulls, bootstraps), `phase5/operators.py`,
  `phase5/scan_conditional.py`, `phase2/*.py`, `scripts/corpus_cluster_boot.py`. Check that the methods
  described in Appendix D match what the code does (estimator, features, z-scoring, hold-out, null,
  p-value formula, bootstrap).

## What we want from you

1. **Numerical audit.** For each number you checked: paper value, file value, match / mismatch.
2. **Claim audit.** For each Result box and for the central claim, abstract and "Results at a glance":
   is the stated claim supported by the evidence at the stated strength? Flag overclaims, hedges that hide
   a negative result, and negative results that are buried.
3. **Statistical audit.** Nulls (are they the right nulls?), multiplicity, the finite-sample p-value
   estimator, bootstrap validity (shard-cluster, document-cluster), the training-row target-prior control,
   the matched-operator comparison (§6), the leave-one-source-row-out design, template robustness,
   the cross-corpus difference-in-differences and its "inconclusive" verdict.
4. **Methods/code consistency.** Anything Appendix D says that the code does not do, or vice versa.
5. **Logic and framing.** Does the reframed central claim (conditional-row vs association, not
   directional vs symmetric) follow from the matched-operator results? Is anything in the paper still
   written as if directionality mattered? Is the synthetic evidence (§9) used for more than it can bear?
   Is the checkpoint section (§10) appropriately hedged?
6. **Presentation.** Figures: anything unreadable, mislabeled, or inconsistent with the text; tables;
   notation; references (any citation that looks wrong or fabricated — check `paper/refs.bib`).

## Rules

- Do NOT modify any file in the repository. Read only. Do not run GPU jobs or long computations; small
  Python checks on results files (seconds) are fine.
- Do not re-run the full experimental pipeline. The numbers you need are in the results files.
- Every finding must cite the file and line (tex) or the file/key (results) it rests on.
- Grade every finding: BLOCKING (the claim is false or unsupported as stated), MAJOR (materially
  misleading or a real statistical problem), MINOR (presentation, wording, small inconsistency).
- End with a verdict in one line: ACCEPT / MINOR REVISION / MAJOR REVISION / REJECT, followed by a
  ranked list of the five most important things to fix.
- Be concrete and blunt. No summaries of what the paper does; we know what it does.

Write the full report as Markdown.
