# PHASE5_DESIGN.md — training-data fingerprint test (written before any Phase-V comparison was run)

## Question
Does each model's spelled-key next-key behaviour reproduce the conditional statistics of spelled text beyond what
circle, line, orthography and frequency already explain? Three claim levels (from the brief): WEAK (both line-like),
MEDIUM (full corpus rows predict full model rows better than theory baselines, held-out), STRONG (residual, idiosyncratic
corpus structure predicts residual model behaviour, with corpus specificity and/or checkpoint-time alignment).

## Objects
- Q^{(m,f)}: model m's 15×15 behavioural matrix q(j | context family f with source key i) from Phase II
  (families C harmonic, D chord, E modulation; 4 templates each; total log-prob scorer, row-normalized over the 15
  spelled candidates). Rows = source spelling i, columns = target spelling j. Enharmonic spellings are NOT merged
  ("spelled tonal view"); a "neutralized view" merges target columns into 12 classes for a secondary analysis.
- C^{(c,F)}: corpus c's directional conditional matrix under extraction family F (defined below), rows = source
  spelling, columns = target spelling, add-0.5 smoothing, row-normalized; raw counts kept.

## Extraction families (fixed before comparison; competing hypotheses, not a menu)
A. Directional local window: target j mentioned within N = 16 words AFTER source i (Karkada window, directed; from
   the Phase-I ordered-pair distance tallies) — and N = 40 from a new scan.
B. Cue-conditioned directional window (N = 40): intervening text contains a cue from one of the families
   modulation / chord / key-signature / enharmonic / generic-relation (as in Phase II corpus_conditional, extended).
C. High-precision relational patterns: regexes `from X major (to|into) Y major`, `X major (to|→) Y major`,
   `modulat\w+ (to|into) Y major` within 12 words after X major, `X major and Y major` — sparse, high precision.
D. Document-conditioned: P(j appears anywhere later in the document | i appears), directional by first occurrence.
Corpora: Wikipedia 20231101 (local); OLMo-Mix-1124 wiki (checked for identity with the local dump) and a DCLM sample;
Dolmino-Mix-1124 (Stage 2) samples — subject to download feasibility, recorded in PHASE5_LOG.md.

## Nuisance / theory features per ordered pair (i, j), i ≠ j
circle distance; line distance |s_i − s_j|; signed difference s_j − s_i; same glyph class; has-accidental(i),
has-accidental(j); same root letter; edit distance; log source frequency; log target frequency (corpus counts);
token count(i), token count(j) for the model's tokenizer.

## Central test (per model × family × corpus × extraction family)
1. Fit log Q and log C separately on the nuisance/theory features (ridge on 210 ordered off-diagonal entries; rank
   version too); take residuals Q_resid, C_resid; report Spearman(C_resid, Q_resid) with a null from 2000 joint
   relabelings of the 15 source/target keys applied to C (rows and columns permuted together).
2. Leave-one-source-row-out prediction of Q: for each i, fit on the 14 other rows (a) theory-only features, (b) corpus
   row only (log C_i,· as a single predictor plus intercept), (c) theory + corpus; predict the held-out row over its 14
   targets; score by Spearman(pred, actual) and by the KL of the softmax-calibrated prediction; average over rows.
   The central number: ΔCV = score(theory + corpus) − score(theory), with a relabeling null (permute the corpus
   matrix's key labels jointly) and a bootstrap over corpus counts.
3. Twin difference vectors: ΔQ_i = log q(·|alias1) − log q(·|alias2) and ΔC likewise, over the 13 non-alias targets;
   cosine (centered) and Spearman; bootstrap over corpus counts; null by permuting targets.
4. Cross-corpus: repeat with each corpus; for OLMo models, test whether OLMo-consumed data beat Wikipedia (paired
   over templates/rows).
5. Checkpoints (OLMo-2-0425-1B, revisions stage1 1B/21B/49B/105B/294B/1007B/1993B/4001B tokens and Stage-2 final):
   the same Q matrices per checkpoint; emergence of the line, of twin asymmetries, and of residual alignment.
6. Operators side by side on the same corpus: symmetric PMI (Phase I), helper-word factorization (Phase I),
   directional conditional (this phase), NTP-style centered support (context n-gram → next key) where constructible;
   which predicts key-name geometry, prompt-final geometry, and behaviour.

## Pre-registered readings
COARSE GEOMETRY ONLY if residual correlations vanish and ΔCV ≈ 0; CONDITIONAL STATISTICS PREDICT BEHAVIOUR if ΔCV > 0
with p < .05 across models/families; TRAINING-DATA FINGERPRINT if residuals align and OLMo's own data beat proxies;
TEMPORAL ACQUISITION MATCH if residual alignment grows across checkpoints; STATISTIC-DEPENDENT GEOMETRY if symmetric
operators recover periodic structure while conditional operators recover the open/spelled structure with quantitative
support; CORPUS–MODEL LINK FAILS otherwise.
