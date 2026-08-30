# RESULTS.md

**Verdict: INTERESTING MISMATCH (with the instrument mostly at fault).** In English Wikipedia the co-occurrence
statistics of the 12 major keys carry a genuine, ordering-specific, *periodic* circle-of-fifths kernel (partial
Spearman with fifths distance +0.62 canonical / +0.66 enharmonic-merged, p < 0.001 under free and block-preserving
relabeling nulls), and Karkada's own embedding construction reproduces it (+0.52). Four base LLMs (1–7B) put only a
small, layer-selection-fragile fraction of it into the geometry of the key-name token (best-layer partials
+0.21/+0.29/+0.23/+0.34; significant after layer-selection correction only for OLMo-2-7B, p = 0.045/0.017) — that
geometry is dominated instead by an *orthographic* "name carries an accidental glyph" block, which, being contiguous on
the circle of fifths, puts 79.6% of its Fourier energy into the fifths fundamental and would have been read as a circle
of fifths by the standard months/weekdays Fourier instrument. What the models do carry — in their next-key predictions
and in the residual stream of next-key contexts (partials +0.3…+0.5, matched-context p ≈ 0.01–0.04) — is an **open line
of fifths ordered by signed accidental count** rather than the corpus's periodic circle; this survives merging
enharmonic spellings at ≈⅔ strength. Outcome D on the original question, resolved into a topology mismatch (periodic
statistic, open-boundary model) plus a methodological warning.

*(Written during the night of 2026-08-28/29 and rewritten after two independent Opus reviews — verbatim reports in
`notes/review_code.md`, `notes/review_science.md`; every correction is dated in `RESEARCH_LOG.md`. All numbers trace to
files under `results/`. "HH:MM" stamps in the log are sequence markers, see its header.)*

## 1. Corpus: the statistic is real, fifths-shaped and periodic

Wikipedia 20231101.en, 41 shards, 6.41M docs, ≈3.1B words, Karkada window (L=16, f(d)=17−d).
- **Karkada's M\* is saturated for keys** (off-diagonal entries 1.97–2.00; ρ median ≈1560): its 12×12 "spectrum" is
  numerical residue and flip-flops with corpus size. PMI (pre-registered secondary; the SGNS target of Levy &
  Goldberg 2014) is used for keys; M\* is fine for months (circulant fraction 0.97, |λ| profile
  [0.50, 0.26, 0.12, 0.08, 0.03, 0.01]).
- PMI kernel in fifths order κ = 8.64, 7.89, 7.59, 7.16, 7.07, 7.13, 6.48 (d′ = 0…6): decreasing with fifths distance
  (one 0.06 wobble at d′=5, tritone dip), jagged in semitone order (`figures/summary/fig1_corpus_kernels.png`).
  The Fourier bin P5 dominates (0.50 vs P1 0.17) but that number depends on the PMI diagonal κ(0); the robust
  statement is that the deviation of κ from its mean peaks at k = 5/7.
- **Controls**: partial Spearman with −fifths distance, controlling for the accidental block, log-commonness, letter
  identity and alphabet distance: **+0.62** (canonical spellings), **+0.66** (enharmonics merged); p = 0.0005 under both
  the free and the block-preserving relabeling null. White-key sub-block ordered along F–C–G–D–A–E–B: RSA +0.63, rank 78
  of 5040 orderings (p = 0.016; merged: +0.67, p = 0.0095). Removing every document that mentions "circle of fifths" /
  "perfect fifth" changes nothing (+0.64).
- **Periodic, not open**: circle vs line-of-fifths (signed accidental count; the two differ on 15 seam-crossing pairs):
  canonical family circle|line +0.40, line|circle +0.27, P(circle > line) = 0.87 (Poisson bootstrap); the canonical set
  starves the seam pairs because Wikipedia is spelling-consistent (Db|F# weighted count 17 vs Db|Gb 160), and with
  enharmonics merged **circle|line = +0.58 ± 0.02, line|circle = −0.07 ± 0.03, P(circle > line) = 1.00**, seam-pair
  ρ(circle) +0.62 (`scripts/corpus_seam.py`).
- **Theory-faithful prediction**: W = Φ√|Λ| from the full V=3000 PMI over the 14,235 key-containing documents gives the
  key rows partial fifths +0.52…+0.56 for d ∈ {100…3000}, circle|line +0.57 / line|circle −0.31. Zeroing the 12×12 key
  block before factorizing changes the key rows by <0.001 (the block is ≈1% of the matrix; helper words carry it —
  consistent with Karkada Fig. 4, though this ablation is close to tautological).
- Convergence: the fifths partial rose from +0.27 (0.1B words) to +0.63 (3.1B) and had not plateaued; F# major has 232
  mentions in all of Wikipedia; bootstrap SDs (Poisson on weighted counts) understate uncertainty ≥3×.

## 2. Key-name token geometry: an orthographic block, and almost no cycle

24 contexts × 12 keys × all layers, three token positions, four models (dtype check: bf16 vs fp32 Δ ≤ 0.002).
- **Raw Fourier profile**: P5 (fifths fundamental) share 0.26–0.55, P1 at null — in every template, position and
  spelling (canonical, all-sharp, all-flat, "-flat/-sharp" words, unicode). *This is the result the standard instrument
  would report.*
- **It is a categorical feature.** The five keys with an accidental are contiguous on the circle of fifths, so a binary
  indicator is a width-5 boxcar in fifths coordinates with 79.6% of its energy in the k=5/7 pair (exact). Diagnostics:
  projecting the indicator out drops P5 to 0.11–0.14, *at* the projection-matched relabeling null (0.12; z ≈ +0.4) — not
  below it, as an earlier draft said; the k=7 mode is rank-1-like (isotropy 0.4–0.6 vs 0.98 for a circle); a synthetic
  calibration (d = 2048) reproduces the observed numbers with boxcar+noise and excludes a ≥50% genuine circle
  (which would give projected P5 ≈ 0.18–0.22, isotropy ≈ 0.8–0.98). Note the projection removes 47.9% of a true
  circle's P5 *energy* (13.5% of its share), so it is not mild; the share-based conclusion stands.
- **The block is orthographic, not musical** (`scripts/decouple_orthography.py`, fig7): respelling four *white* keys as
  B#, Fb, E#, Cb makes "has an accidental glyph" (9/12) differ from "is a black key" (5/12). Gram RSA with the black-key
  block goes from +0.70…+0.86 to −0.14…+0.12 in all four models, while RSA with the glyph block rises to +0.5 (1B),
  +0.4 (Gemma), +0.7 (Qwen), +0.3 (7B). (B#/Fb/E#/Cb are rare strings, so glyph and rarity stay entangled; key-signature
  membership is ruled out.)
- **What is left after controls** (partial Spearman with −fifths distance; ctrl: glyph block, commonness, letter
  identity, alphabet distance; nulls = 1000 relabelings, free and block-preserving, and max-over-layers correction —
  `scripts/partial_nulls.py`, `results/nulls_multictx.txt`):

| model | glyph-block RSA (max) | best-layer partial fifths | p (max-over-layers) free / block | final-layer partial | white-key fifths-line RSA (best), exact-null p / layer-corrected p |
|---|---|---|---|---|---|
| OLMo-2-1B (fp32) | +0.86 | +0.21 (L13) | 0.16 / 0.21 | −0.16 | +0.41, 0.037 / 0.19 |
| Gemma-2-2B (bf16) | +0.84 | +0.29 (L23) | 0.10 / 0.09 | −0.13 | +0.61, 0.003 / **0.028** |
| Qwen2.5-3B (bf16) | +0.84 | +0.23 (L14) | 0.15 / 0.19 | −0.16 | +0.42, 0.025 / 0.24 |
| OLMo-2-7B (bf16, CPU offload) | +0.86 | +0.34 (L30) | **0.045 / 0.017** | +0.09 | +0.78, <0.001 / **0.001** |
| corpus PMI (canonical / merged) | +0.47 (mutual) | +0.62 / +0.66 | 0.0005 | — | +0.63, 0.016 / — |
| theory embedding (d=300) | +0.57 | +0.52 | 0.0005 | — | +0.45 |

So the "scale curve" of the token geometry is three noise-level points and one significant one (7B), and even that one
is line-like (circle|line ≈ 0, line|circle +0.2…+0.3), is not an isotropic Fourier circle, and is lost at the final layer.
Static input embeddings show nothing (P5 0.20, z 1.2). Major and minor keys behave alike.

## 3. Where the fifths kernel does appear: next-key contexts, output side, as a line

**Predictions.** L[x,y] = log P(" y major" | context(x)), six contexts, all four models (`scripts/predictive_matrix.py`):
RSA with corpus PMI +0.44…+0.62; partial fifths (same controls) +0.20/+0.23/+0.33/+0.33 ("modulates from x major to")
and up to +0.42/+0.50 ("…tonic chord is usually followed by the chord of", Qwen/7B). Adding the model's own column
marginal as a control *raises* these (+0.23/+0.30/+0.46/+0.43; chord-after +0.58/+0.61), so they are conservative.

**Topology.** Circle vs line (each controlling for the other + block + commonness):

| model | canonical targets: circle\|line / line\|circle | enharmonic-merged targets ("modulates to") | PC1 tracks |
|---|---|---|---|
| OLMo-2-1B | −0.39 / +0.69 (then_key −0.40 / +0.76) | −0.11 / +0.46 | signed accidental count 0.78 |
| Gemma-2-2B | −0.38 / +0.74 | −0.07 / +0.52 | 0.73–0.78 |
| Qwen2.5-3B | −0.41 / +0.79 (seam-pair ρ(circle) −0.81) | −0.20 / +0.66 | 0.76–0.80 |
| OLMo-2-7B | −0.33 / +0.74 | −0.06 / +0.56 | 0.70–0.78 |
| corpus PMI (merged) | **+0.58 / −0.07** | — | cos(circle) 0.83 (but confounded with commonness, 0.85) |

The line survives the model-marginal, flat-name and accidental-sign controls (line|circle +0.34…+0.58) and jackknife
over keys (reviewer: +0.33…+0.84, p < 0.0005 in all cells). Merging the two spellings of each black key removes ≈¼–⅓ of
it (canonical-spelling scoring forced e.g. F#'s dominant to be "Db"); the rest is real. One nuance: with merged targets
the chord-progression context in the larger models is circle-like (Qwen +0.35/+0.18, 7B +0.44/+0.14) — enharmonic
identity is honoured for chord relations but not for key succession. Behaviour tracks this: few-shot fifth relations
(raw / length-normalized mean-per-token scoring; chance 0.08): dominant 0.08/0.17, 0.33/0.33, 0.75/0.67, 0.92/0.83 for
1B/2B/3B/7B; the raw scorer under-selects 2-token names (a lower bound); a single-prior "calibrated" scorer over-corrected
and is reported only as a failed variant (`results/behavior/fewshot_three_scorers.txt`).

**Residual stream at the predicting position** (`scripts/predict_position.py`, `results/nulls_predpos.txt`), same
controls, max-over-layers nulls, *matched contexts* (concept token and final token from the same sentence):

| model | context | concept token: best partial, p | predicting token: best partial, p | non-predicting control (final token) |
|---|---|---|---|---|
| OLMo-2-1B | then_key | +0.33 (0.058) | **+0.46 (0.018)** | −0.01 |
| Gemma-2-2B | then_key | +0.42 (0.031) | +0.38 (0.039) | +0.16 |
| Qwen2.5-3B | modulates_to | +0.37 (0.039) | **+0.47 (0.020)** | +0.14 |
| OLMo-2-7B | modulates_to / then_key | +0.46 (0.015) / +0.52 (0.007) | +0.53 (0.010) / +0.47 (0.020) | +0.37 |

So most of the difference between "+0.2 in the key name" and "+0.5 at the predicting state" that an earlier draft
attributed to *position* is **context**: a key-relation context (one sentence about modulation or a following piece)
raises the key token's own geometry to +0.33…+0.52, and the predicting position adds ≈0.1 in 1B/3B and nothing in
2B/7B. The non-predicting control is ≈0 for 1B/3B but +0.37 for 7B. At the final layer the predicting state is line-like
in every model (line|circle +0.4…+0.7, circle|line negative); a circle-like intermediate stage (circle|line ≈ +0.3)
appears in only 1 of 12 model×context cells at a corrected p < 0.05 (7B "modulates to", L17–18) — not a general
"circle mid-network → line at output" mechanism, contrary to an earlier draft.

## 4. Positive control

Months: corpus M\* |λ| profile vs OLMo-2-1B Fourier profile — best-layer cosine 0.97 (Karkada template) / 0.99
("It happened in x, which"), Spearman 0.94–1.00; per-layer minima 0.82–0.94; a flat (2/11,…,1/11) null profile would
give 0.75. P1 z ≈ 6, contextual sharpening to P1 share 0.52 at the last layer.

## 5. Strongest evidence against the reading above

1. Scale: 1–7B only; the 7B key-name geometry is significant and the 7B non-predicting control is high — a 13B–70B model
   may put a real circle into the key token, which would demote the "instrument at fault" story to "at small scale".
2. PMI ≠ M\*: the corpus conclusions rest on PMI; at the full-vocabulary level M\* and PMI factorizations agree (RSA +0.43
   vs +0.44), but the 12×12 comparison is PMI-only.
3. The circle-vs-line contrast rests on 15 pairs whose corpus counts are ~2 orders of magnitude below the rest; the
   canonical family only reaches P = 0.87; the merged family is the clean evidence.
4. 12 concepts → 66 pairs; controls are rank-based partials with 4–5 covariates (VIF ≤ 1.4; circle/line VIF ≈ 2.5).
   Relabeling nulls (free and block-preserving) agree, but power is limited: an effect of +0.2 cannot be resolved
   at one layer, only in aggregate.
5. Wikipedia ≠ the models' training mixes; the corpus signal had not converged with corpus size.
6. Mode isotropy (0.4–0.65) reads as a 15–30% circle admixture while the projected-P5 diagnostic reads as ≈0–15%; the two
   are not perfectly concordant and isotropy is uncalibrated for anisotropic residual noise.

## 6. What is novel, what is not

- Novel (no prior work found): key-name residual-stream geometry in text LLMs; a competing-orderings test of the
  co-occurrence→geometry theory; the corpus-circle vs model-line (periodic vs open-boundary) dissociation, which
  connects to Temperley's line of fifths and Chew's spiral array rather than the Krumhansl–Kessler torus.
- Not novel, but a useful service note: a categorical feature contiguous on one cyclic ordering fakes the fundamental of
  the competing ordering (textbook DFT of the diatonic/black-key set peaking at coefficient 5 — Quinn, Amiot); the
  *orthographic* origin of that feature in LLMs is the sharper point.
- Verified algebra (MATH.md): under x → 7x only P1 ↔ P5 swap; P2, P3, P4, E6 are invariant.

## 7. Probable artifacts and corrections made after review

Raw "P5 > P1" everywhere (orthographic block + tokenization: sharps are 2 tokens, " Db/ Eb/ Ab" distinct single tokens);
"projected P5 below the null" (wrong: at the matched null); "13.5% of a circle's energy" (share, not energy: 47.9%);
"κ monotone" (one wobble); M\* "2.00 ± 0.01, ρ ≈ 1800" (±0.03, median ≈1560); Gemma/Qwen block maxima +0.84; Qwen best
layer L14; flat-null cosine 0.75; a `partial()` NaN bug (layer-0 rows); a `cof_docs` counter; the tonic template with
a comma anchor (tokenization merge); a 42-shard merge (fixed); the "black/white-key block" name (orthographic);
the "circle-like mid-network" and "locus, not context" claims (demoted, §3); the few-shot raw scorer (length-biased,
lower bound; mean-per-token reported alongside).

## 8. Experiments performed

Corpus: `corpus/scan_wiki.py`, `merge.py`, `analyze.py`; `scripts/corpus_report.py`, `corpus_confounds.py`,
`adjacency.py`, `convergence.py`, `corpus_seam.py`, `directed_corpus.py`; `corpus/scan_vocab.py` + `scripts/theory_embedding.py`.
Model: `scripts/extract_all.py` + `spectra.py` (10 families, 30 templates, 4 positions); `confounds.py`; `multicontext.py` +
`multictx_analyze.py` (24 contexts); `decompose_rsa.py`; `partial_nulls.py`; `decouple_orthography.py`;
`behavior_keys.py`, `behavior_fewshot.py`, `behavior_fewshot_calibrated.py`; `predictive_matrix.py` (canonical + merged
targets); `predict_position.py` (+ control context); `circle_vs_line.py`. Synthetic: `tests/test_synthetic.py` +
boxcar/circle calibration. Figures: `figures/summary/fig1–fig7`.

## 9. Compute

One RTX 5070 Ti laptop GPU (12 GB); forward passes only. Per model: ≈1 min (1B fp32) to ≈21 min (7B bf16 with ≈5 GB
CPU offload) per full pipeline plus CPU nulls (2000 relabelings per statistic). Wikipedia scan ≈40 core-minutes;
vocabulary pass ≈2 min. Wall-clock dominated by ≈45 GB of downloads. Two independent Opus reviews (~2.5 h agent time).

## 10. Paper seed?

Yes, narrowly: *"Corpus co-occurrence of musical keys is a periodic circle of fifths; small LLMs express it as an open
line of fifths on the output side and barely at all in the key-name geometry, which is dominated by an orthographic
accidental feature that mimics a circle of fifths under the standard Fourier instrument."* Needed before it is a paper:
a 13B+ point on the same pipeline; a tokenization-free replication (synthetic corpus + small transformer with a
controlled periodic kernel and no accidental tokens); a causal test of the seam (why do modulation contexts break at
F#/Db while chord contexts do not); and the OLMo-mix corpus. See `NEXT_STEPS.md`.


*(Hardening correction 2026-08-30: the Poisson-on-counts bootstrap SDs quoted above (±0.02/±0.03; P(circle > line) = 0.87
for the canonical family) ignore document clustering and are withdrawn. A shard-cluster bootstrap (41 shards, B = 1000;
`scripts/corpus_cluster_boot.py`, `results/corpus/wiki/cluster_boot.txt`) gives fifths partial +0.62 [+0.44, +0.68]
canonical / +0.66 [+0.49, +0.71] merged; merged circle|line +0.58 [+0.39, +0.64], line|circle −0.07 [−0.20, +0.07],
P(circle > line) = 1.00; canonical circle|line +0.40 [+0.05, +0.55], line|circle +0.27 [−0.08, +0.58], P = 0.63.)*
