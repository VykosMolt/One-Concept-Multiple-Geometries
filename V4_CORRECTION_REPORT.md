# V4_CORRECTION_REPORT.md — correction of the Phase-V held-out baseline (paper-v1 → paper-v2)

Scientific correction log, 2026-08-30 → 2026-09-02. Not a defence of paper-v1: the corrected experiment says
what it says, and paper-v2 reports that.

## Trigger

An Opus 5 manuscript review of paper-v1 (`notes/review_opus5_v4.md`, 120 numbered findings, ledger in
`notes/OPUS5_V4_VERIFICATION.md`) found two substantive defects in the flagship Phase-V regression
(§7 of the paper, `phase5/fingerprint.py`) and one misstatement of the template-robustness numbers (B3).
Both substantive defects were independently confirmed on 2026-08-30 before any code was changed.

## B1 — the "circle" predictor was chromatic distance

`phase5/fingerprint.py` (v3, both views) and `phase5/ckpt_fingerprint.py` built the feature named `circ` from
`pc = (7·S) mod 12`, i.e. the semitone pitch class, and then took the cyclic distance in `pc`. That is the
**chromatic** cyclic distance, not the circle-of-fifths distance. Landmarks: C→G has fifths distance 1 and chromatic
distance 5; C→D♭ has fifths distance 5 and chromatic distance 1. Over the 165 target-aggregated pairs the two
distances correlate at Spearman −0.19 (−0.11 over the 105 spelled pairs). Every held-out, residual, cross-corpus
and checkpoint result in paper-v1 §7, §8, §10 (and the rich model's cosine "harmonics", which were harmonics of
the chromatic distance) therefore controlled for the wrong geometry, while the abstract, central claim,
contributions and discussion said "beyond explicit circle … baselines". `phase5/operators.py` and
`scripts/corpus_cluster_boot.py` use `keys15.candidate_geometries` and were not affected; Phases I–IV were not
affected.

## B2 — merged target classes were given an arithmetic line coordinate

In the target-aggregated view (12 classes; enharmonic target spellings merged by log-sum-exp) the v3 line feature
was the distance to the *mean* line coordinate of the class. The merged classes {C♭ (−7), B (+5)}, {G♭ (−6), F♯ (+6)}
and {D♭ (−5), C♯ (+7)} thereby received coordinates −1, 0, +1 — points on the line occupied by no spelling.
A log-sum-exp class is dominated by whichever spelling the model prefers, so a single invented centroid is not a
defensible representation. A reviewer sensitivity check (nearest-spelling line features added) already reduced the
eight flagship v3 gains by roughly two thirds.

## Implementation corrections (commit `a56ad3e`, tests `tests/test_phase5_v4.py`)

- **Feature definitions** (`phase5/theory_features.py`, schema `phase5-theory-v4.1`, feature list and formulas
  stored inside every result JSON under `feature_definitions`): true fifths-circle distance
  d₅ = min(|f(z_i)−f(z_j)| mod 12, 12−…) with f(z)=7z mod 12; chromatic cyclic distance as a *separate* predictor;
  spelled view: |s_j−s_i| and s_j−s_i, glyph-class equality, source/target flat and sharp indicators, same root,
  alphabet distance, unsigned signature-size difference, edit distance, source/target log frequency, fold-fitted
  tokenizer residuals. Target-aggregated view: a class T is a *set* of spellings; the line is represented by
  min/max_{t∈T}|s_t−s_i| and the signed endpoints min/max_{t∈T}(s_t−s_i); set-valued orthographic controls
  (min alphabet distance, min/max signature-size difference and edit distance, contains-flat/contains-sharp,
  any-same-glyph/root), source log frequency and log summed class frequency, source tokenizer residual and mean
  target residual. Rich variant adds d₅², the relevant line-distance square, d₅×line, chromatic×line, cos(2πd₅/12),
  cos(4πd₅/12) (the k=3 cosine is exactly dependent in this design and was removed). Exact algebraic duplicates
  found during the contract freeze were dropped so that ridge regularization is not silently weakened.
- **Fold-local preprocessing**: for every held-out source row the tokenizer projection
  (token_count ~ 1 + is_flat + is_sharp) is fitted on the 14 training source keys, and every continuous column —
  including the corpus column and the target prior — is centred/scaled with statistics from training-pair
  observations only, then applied unchanged to the held-out row; the held-out response enters no preprocessing.
  v3 (and the first, unaccepted v4 preview run of 2026-08-30) standardized globally. Per-fold scalers, feature
  names and rank audits are recorded in each JSON (`loo.fold_scalers`, `loo_feature_names_by_fold`,
  `loo.rank_audits`).
- **Deterministic randomness**: Python's salted `hash()` was replaced by SHA-256 over canonical JSON of
  (master seed 20260830, corpus, model, family, extraction, view, rich, targetprior, templates, stream, replicate);
  identical output was verified across `--jobs` settings and `PYTHONHASHSEED` values. All v4 nulls, document
  bootstraps and thinning replicates were regenerated under this scheme.
- **Null estimator** unchanged from the hardening pass: p = (b+1)/(B+1), B = 5,000 (Wikipedia), 2,000
  (cross-corpus, templates, checkpoints); reported only to the resolution of B.
- **Size-matched baselines**: five independently seeded Wikipedia thinnings per corpus
  (`cond_wikipedia_thin_<corpus>_v4[_s1..4].npz`); `phase5/crosscorpus_compare_v4.py` averages them and reports the
  seed SD (`results/phase5/thinning_seed_variance.csv`).
- **Provenance and guards**: `phase5/v4_provenance.py` (81-input SHA-256 manifest, preflight, output allowlist),
  `phase5/validate_v4.py` (fail-closed checks on B, estimator, schema, finiteness, bootstrap completeness),
  atomic writes, v3-overwrite guard, path-traversal/symlink rejection. All v3 artifacts are byte-identical to
  paper-v1. Invalid preview artifacts from the first v4 attempt were quarantined outside the repository
  (`~/Documents/Research/pitch_fourier_v4_preacceptance_quarantine_20260831`), not deleted.
- **B3 and the other numerical findings** of the Opus review were verified item by item
  (`notes/OPUS5_V4_VERIFICATION.md`: 63 confirmed, 12 reviewer errors, 4 partial, 1 provenance-limited, 40 that
  depended on the final v4 run) and the manuscript corrected accordingly.

The chain `phase5/rerun_v4.sh` was launched only after an independent implementation verifier and an independent
reviewer had returned READY on the committed snapshot; it ran 2026-08-31 19:23 → 2026-09-01 01:22 UTC on CPU
(20 workers, no GPU), log `results/phase5/rerun_v4.log`, marker
`results/phase5/V4_COMPUTE_AND_COMPARISON_COMPLETE.marker`.

## Result changes (target-aggregated view, modulation family; ΔKL nats per row; OLMo-1B / Gemma / Qwen / OLMo-7B)

Full machine-readable comparison of 57,348 inherited numbers: `results/phase5/v3_v4_comparison.csv` (summary
`.md`).

| cell | paper-v1 (v3) | paper-v2 (v4) | v4 p | status |
|---|---|---|---|---|
| window, base | +0.037 / +0.029 / +0.016 / +0.036 (all p ≤ .0008) | +0.0041 / +0.0053 / −0.0001 / +0.0038 | .015 / .0012 / .334 / .0030 | Qwen lost |
| document, base | +0.045 / +0.036 / +0.023 / +0.039 | +0.0052 / +0.0041 / +0.0000 / +0.0007 | .0076 / .0118 / .303 / .235 | Qwen, 7B lost |
| window, rich | +0.023 / +0.019 / +0.007 / +0.018 | +0.0035 / +0.0061 / −0.0002 / +0.0033 | .034 / .0004 / .402 / .0026 | Qwen lost |
| document, rich | +0.034 / +0.026 / +0.013 / +0.022 | +0.0039 / +0.0031 / −0.0003 / −0.0003 | .044 / .022 / .553 / .628 | Qwen, 7B lost |
| window, rich + target prior | +0.027 / +0.022 / +0.011 / +0.025 | +0.0039 / +0.0045 / −0.0003 / +0.0030 | .0004 / .0002 / .608 / .0010 | Qwen lost |
| document, rich + target prior | +0.035 / +0.029 / +0.018 / +0.029 | +0.0039 / +0.0019 / −0.0004 / +0.0004 | .0002 / .0010 / .708 / .0446 | Qwen lost; 7B marginal |
| theory-only KL₀ (base / rich) | 0.117–0.134 / 0.105–0.123 | 0.045–0.063 / 0.042–0.059 | | |
| reduction of KL₀ (window, rich) | 6–19 % window, 12–28 % document ("14–34 %" base) | 5.9 % / 13.0 % / −0.4 % / 7.3 % | | |
| doc-cluster 95 % CI, window (v3: base model only) | all eight base cells excluded 0 | rich: only Gemma excludes 0 ([+0.0016, +0.0071]); base: 1B, Gemma, 7B | | |
| doc-cluster 95 % CI, document | all excluded 0 (base) | none excludes 0 (base or rich) | | |
| residual r (window / document) | +0.60 / +0.59 / +0.50 / +0.60 (document) | window +0.24 / +0.28 / +0.07 / +0.31 (p .019/.0006/.236/.0004); document n.s. | | |
| matched operators (window) | sym +0.039/+0.030/+0.016/+0.040, rev +0.029/…, PMI +0.021/…, all p ≤ .0016 | sym +0.0052/+0.0071/+0.0005/+0.0069; rev +0.0019/+0.0065/+0.0004/+0.0036; PMI −0.0017/−0.0000/+0.0008/−0.0022 | sym Qwen n.s.; rev 1B, Qwen n.s.; PMI null | |
| templates (Table C.7) | every leave-one-out aggregate significant | 19/32 leave-one-out and 9/32 single-template cells significant; Qwen null throughout | | |
| cross-corpus direct gains | "every corpus gives the gain in every model" | aggregated grids: 8/0/6/0 of 36 cells significant (OLMo-Mix wiki / 9-sh DCLM / Dolmino / 54-sh DCLM); 54-sh DCLM modulation-window negative in all four models | | |
| specificity (72 DiD cells) | 10 significant, all OLMo-favouring ("inconclusive") | 8 significant, 4 favour OLMo and 4 oppose; 33 positive / 39 negative | | fingerprint not supported |
| checkpoint residual (document) | first significant at 294B (p .037), stable from 4T (p ≤ .017) | no stage-1 point significant (window max +0.22, p .083); stage-2 window +0.21/+0.23/+0.18 (p .023/.013/.052); document never significant | | threshold claim withdrawn |
| twin-difference alignment | 16 of 108 line-controlled cells significant (paper-v1 misprinted 17) | 16 of 108 | | unchanged |

Harmonic-relation and chord families: no cell is significant under the corrected base or rich model in any
model (v3 had reported 7B chord gains of +0.064/+0.071 and target-prior-rescued harmonic gains); with the
training-row target prior three Qwen cells reach p < .05 at +0.0009 to +0.0011 nats and nothing else does.

## Claims weakened or withdrawn

- "Corpus conditionals reduce held-out KL by 14–34 % beyond circle, line, orthography and frequency" —
  withdrawn. The corrected reduction is 5.9 / 13.0 / −0.4 / 7.3 % (window, rich) and the baseline in v1 did not
  contain the fifths circle.
- "The gain replicates across all four models" — withdrawn. Qwen2.5-3B is null in every corrected cell;
  OLMo-7B's document gain is null; only Gemma's rich-window cluster interval excludes zero.
- "Document-cluster bootstrap intervals exclude zero in every cell" — withdrawn.
- "Larger, not smaller, with a training-row target prior" — reduced to: the prior sharpens the relabeling test in
  the three models that already show the effect and does not rescue Qwen.
- "Every matched operator is significant (all p ≤ .001)" — withdrawn.
- "The gain is generic across corpora; even the 633-pair DCLM sample gives it in every model" — withdrawn.
- "Specificity inconclusive, all significant DiD cells OLMo-favouring" — replaced by a directionless 4/4 split;
  the training-data fingerprint is neither established nor suggested.
- "Residual correspondence first significant at 294B tokens and stable from 4T" — withdrawn; the corrected
  trajectory has no significant stage-1 point.
- "The two scorers agree throughout the target-aggregated view" — one of eight cells now disagrees.
- The document conditional is no longer described as the stronger regressor; the earlier conjecture about
  document residuals is dropped.

## Claims that survive (corrected wording)

- Operator-dependent geometry: PMI-type association identifies enharmonic twins and is periodic; conditional-row
  operators keep spellings apart while remaining fifths-smooth; the contrast is one of construction, not
  directionality (Phase-I/II/V descriptive statistics, unaffected by B1/B2).
- Conditional-row operators align with restricted next-key behaviour far better than association operators
  (Spearman ≈ 0.80 vs 0.1–0.5), unaffected.
- A modest, model-dependent held-out contribution of the local-window conditional beyond the corrected rich
  baseline: positive by the relabeling test in OLMo-1B, Gemma and OLMo-7B (p = .034 / .0004 / .0026), not in Qwen;
  cluster-bootstrap support only for Gemma. Document conditional: OLMo-1B and Gemma only, pointwise.
- Cross-corpus heterogeneity is the honest reading of Result 6; no corpus or family was promoted post hoc.

## Unaffected results (not inherited from B1/B2)

Phase I (corpus PMI fifths structure, shard-cluster bootstrap, helper factorization, key-name aliasing and the
respelling intervention), Phase II (15-key behaviour follows the line; last-token artefact; enharmonic identity in
7B only; merged-scorer results), Phase III (output-code control), Phase IV (sparse-alias control), the operator
comparison of §6 (descriptive Spearman/ECI values; `phase5/operators.py` used the correct geometries), and the
checkpoint line/circle partials and twin asymmetries of §10. Independent of v4, the same pass corrected: the
merging-experiment conflation (12-key merged target vs 15-key merged scorer), the corpus-size non-convergence of
the canonical fifths partial (+0.27 at 0.11B words → +0.63 at 3.1B, no plateau), the "14 of 16" template count
(15 of 16), the Poisson-SD attribution, the last-token line nulls, subspace shares, the Gemma context contrast,
the slash-notation count, the 7B top-1 statistic, the Sadek et al. venue, and the §9 wording items in the ledger.

## Versioning

- `paper-v1` = `fb3e3e8` (immutable; mirror commit `e4423ac`): the pre-correction manuscript with the wrong circle
  baseline and the class-mean line encoding. Retained as the auditable snapshot.
- `a56ad3e`: corrected v4 implementation, frozen before the recompute.
- `paper-v2`: the corrected manuscript, figures, results and this report; mirror rebuilt from it. The paper's
  Appendix A lists B1, B2, the global-standardization and RNG corrections as their own rows.
- After the v4 integration the manuscript was re-ordered so that operator dependence and the aliasing result lead and
  Phase V is one section (§8, "Beyond correspondence"); section/result/figure numbers above refer to paper-v1 where
  they cite the old draft. The mapping is recorded in `MANUSCRIPT_AUDIT.md` (Restructure).
