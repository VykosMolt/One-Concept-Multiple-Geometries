# PHASE2_LOG.md — chronological log of Phase II (machine-clock timestamps from `date`)

Phase I is frozen at git tag `phase1-final` (commit 8d7ae5c). Phase II code lives in `phase2/`, results in
`results/phase2/`, figures in `figures/phase2/`; Phase-I files are not modified except that this log,
`PHASE2_LITERATURE.md` and `PHASE2_RESULTS.md` are added.

Central Phase-II question: why does the same tonal domain show periodic-circle (corpus), open-line (next-key
prediction) and orthographic (key-name geometry) structure under different data sources, tasks and contexts?
Hypotheses H1–H9 as given in the Phase-II brief. Priority order: 15-key enharmonic model test → context
factorial (esp. chord-progression circle effect) → joint circle/line/orthography decomposition with nulls →
sequence-level behaviour → 15-key corpus → major/minor → task-aligned corpus statistics → arbitrary output codes
→ causal → larger models.

Claim discipline (from the brief): no "circle→line transition", "predicting-position locus", "scaling trend",
"black/white-key semantic feature", "P5 means circle", "tokenization artifact" (say orthographic/spelling),
"topological transition". Use: periodic fifth structure, open fifth coordinate, spelling-sensitive geometry,
accidental-glyph feature, neutral vs tonal pitch class, context-conditioned geometry, candidate-sequence
prediction, Fourier aliasing, partial fifth effect.

Sat Aug 29 09:48:15 AM CEST 2026
## Phase II start

### Setup facts (before any Phase-II model result)
- Corpus counts for the 15 spellings (Wikipedia major keys): Cb 38, Gb 162, Db 438, Ab 813, Eb 2945, Bb 2666, F 4290,
  C 6348, G 4725, D 6149, A 4286, E 3101, B 1544, F# 232, C# 121. Weighted enharmonic-pair co-occurrence: Cb–B 104
  (Cb's largest partner), Gb–F# 201, Db–C# 82 — rare spellings co-occur with their enharmonic twins, plausibly via
  explicit "enharmonically equivalent to" prose; to be checked at document level. Seam-pair counts remain 1–2 orders
  below common pairs; the 15-key corpus analysis must be reported with bootstrap uncertainty, not as geometry.
- Tokenization: Gemma-2 has all 15 symbol spellings as single tokens except F#/C# (2); OLMo-2 and Qwen2.5 split
  Cb/Gb/Bb/F#/C# (2 tokens). Word spellings: naturals 1, flats 2 (Gemma 3), sharps 3. No spelling is token-uniform;
  token count is carried as a candidate geometry and the word spelling is a secondary control.
- Candidate geometries implemented in phase2/keys15.py: circle_fifths (periodic neutral pc), line_fifths (open),
  chromatic, Krumhansl–Kessler major-major (1 − profile correlation; periodic, shaped), Chew spiral-array key-center
  distance (open helix, quarter-turn per fifth), glyph_class, has_accidental, edit_distance, same_letter, alphabet,
  n_accidentals (unsigned), tokcount, commonness.
- Context families (phase2/contexts.py; 4 templates each; written before results): A spelling, B enharmonic identity,
  C harmonic relation, D chord progression, E modulation/next key, F generic mention.
- Candidate-geometry collinearity over 105 pairs (15 keys): circle_fifths vs line_fifths ρ = 0.25 (vs 0.68 in the 12-key
  design); KK major–major ≈ circle (ρ 0.99); Chew spiral-array key distance ≈ line (ρ 1.00) for these parameters; glyph_class
  ≈ edit_distance (0.84) and ≈ line (0.65); has_accidental vs circle 0.47. Consequence: with major keys only, the classical
  spaces (KK, spiral array) cannot be discriminated from circle/line; they are kept for the major/minor extension. The
  orthographic family (glyph class, edit distance, has_accidental) is moderately collinear with the line — partial and
  nested tests must be reported with that in mind.

### 09:56 Track 5 (15-key corpus) and Track 7 (aliasing) — first results
- 15-key Wikipedia PMI (phase2/corpus15.py): the three enharmonic pairs are the *closest* pairs of all (ECI 0.04; PMI 9.9–11.0
  vs median 7.3); circle|line +0.42, line|circle +0.38 (both p < 0.001 under free and glyph-preserving relabeling;
  P(circle > line) = 0.89 by Poisson bootstrap). So the spelled corpus statistic is BOTH periodic and open: rare spellings
  co-occur with their line neighbours (key-signature prose) AND with their enharmonic twins.
- Targeted document scan (phase2/scan_enharmonic_docs.py): only 41 co-mentions of enharmonic pairs within 140 characters
  exist in all of Wikipedia; 49% contain the word "enharmonic" and most others are slash notation ("Gb major/F# major")
  or harp-notation discussions (C-flat major for harp parts in B major). => The corpus's NPC identity at the seam is
  carried by explicit metalinguistic equivalence statements, not by harmonic usage. (The Phase-I 12-key merged-family
  circle result rests on non-enharmonic seam pairs such as B–Db and is unaffected.)
- Aliasing (phase2/aliasing.py): contiguous blocks of width 4/5/6 on a 12-cycle put 70/80/83% of their energy in the
  fundamental pair; 29% of all 4094 binary partitions put > 50% of their energy in a single paired mode and 6.6% > 70%;
  random 3- and 4-class partitions: P(> 50%) = 8.5% / 2.7%; root-letter identity under semitone labelling puts 34% in
  P1 (chromatic-looking). Alternating labels put 100% in E6. So a dominant Fourier pair is a common outcome for
  innocent categorical features, and which pair it lands in is decided by the labelling.

### 09:57 first 15-key geometry cells (OLMo-2-1B, symbol spelling) and a control added BEFORE reading further
At the key token, layer 15, the three enharmonic pairs are among the closest ~15% of all 105 pairs in families A, D, F
(ECI 0.15–0.17; corrected p_min 0.03–0.04 free, ≤ 0.01 glyph-preserving), 0.26–0.33 in B, C, E; final layer 0.30–0.63.
At the prompt-final position ECI ≈ 0.5 (null). Before interpreting: every enharmonic pair is also a pair of
alphabet-adjacent letters with different glyphs (C/B, G/F, D/C), a purely orthographic property shared by e.g. Ab–G,
Eb–D, Bb–A, F#–G, C#–D. Added (i) alphabet distance to the control set and (ii) an explicit comparison of the enharmonic
pairs' rank with the 11 letter-adjacent cross-glyph non-enharmonic control pairs; all fits re-run with this.

### 09:59 Track 6 — cue-conditioned corpus statistics (phase2/corpus_conditional.py)
Ordered key-mention pairs within 24 words over key-containing Wikipedia docs, split by intervening cue words. Symmetric
log-association per class, 15 spellings, same controls. Results (circle|line / line|circle, Poisson-bootstrap SD):
all 29,485 pairs +0.59/+0.23; other 25,127 +0.47/+0.19; chord 1,087 +0.64/+0.21; signature 3,200 +0.68/0.00;
modulation 327 +0.25/+0.10 (sparse); enharmonic 53 (sparse). ECI 0.03–0.07 in every class except modulation (0.43).
=> No cue class of the corpus is line-dominant; even key-signature prose is periodic (because rare spellings are
discussed together with their enharmonic twins). H9 ("corpus contains both, model picks the task statistic") is NOT
supported by this construction: the open coordinate in the models' next-key predictions has no counterpart in the
corpus's conditional statistics that we can find. Directed-interval histograms peak at ±1, +2 on the line in every class.

### 10:01 OLMo-2-1B 15-key geometry with controls (phase2/geometry_fit.py; results/phase2/geometry_olmo2_1b.log)
Key token, layer 15: ECI (mean percentile rank of the 3 enharmonic pairs) 0.15 / 0.15 / 0.17 in families A_spelling /
D_chord / F_generic (control pairs — letter-adjacent, cross-glyph, non-enharmonic — 0.58–0.66; corrected p_min free
0.03–0.04, glyph-preserving ≤ 0.01), 0.26–0.33 in B_enharmonic / E_modulation / C_harmonic (p_min free 0.20–0.38,
glyph 0.01–0.08). Final layer 0.30–0.63. At the prompt-final position ECI ≈ 0.5 (null) everywhere.
circle|line partial (ctrl: glyph class, edit distance, same letter, alphabet, token count, commonness, line): best
+0.20 (A, L13; p_max 0.07), +0.27 (B final, L11; p_max 0.026); line|circle: best +0.30 (E last, L9; p_max 0.006/0.002),
+0.33 (E final, L16; p_max 0.002/0.014), +0.26–0.27 (B, F last; p ≈ 0.02–0.03). CV R² gains over orthography ≤ 0.04.
Reading (1B only): a neutral-pitch-class identity signal for enharmonic twins exists at the key token (layer 15),
strongest in spelling/chord/generic families — the opposite of H4's prediction that enharmonic/harmonic contexts should
strengthen it; the open line is the dominant *metric* structure and is strongest in modulation contexts (E) at both
positions. Orthography explains most CV variance (R² 0.14–0.55); circle/line add little.

### 10:13 15-key geometry, all four models (symbol spelling; results/phase2/geometry_*.log; figures/phase2/heatmap15_symbol.png)
Key token, corrected (max-over-layers) p: line|circle significant in every family for Qwen2.5-3B (+0.26…+0.37, p ≤ 0.034)
and OLMo-2-7B (+0.30…+0.34, p ≤ 0.014); in OLMo-2-1B for E/F/B (+0.26…+0.30, p ≤ 0.03); Gemma-2-2B weak (best +0.24, B,
p 0.05). circle|line at the key token: only Qwen C_harmonic +0.30 (p 0.022) and B_enharmonic +0.24 (p 0.05) reach
significance. Enharmonic collapse (ECI) at the key token: OLMo-1B A/D/F 0.15–0.17 (p 0.03–0.04); Qwen A 0.16 (p 0.07),
C 0.18 (p 0.08); Gemma ≥ 0.20 (n.s.); OLMo-7B 0.35–0.48 (n.s.) — the largest model shows no enharmonic-twin identity in
its key-name geometry while having the strongest line. Prompt-final position: 7B circle|line +0.26…+0.36 significant in
A, B, C, D, E (D_chord +0.36, p 0.004; E +0.36, p 0.006; C +0.31, p 0.012; B +0.32, p 0.006; A +0.26, p 0.024; F +0.21,
p 0.03) alongside line|circle +0.22…+0.34; Qwen final circle|line significant in B (+0.28, p 0.018) and E (+0.27, p 0.02).
CV-R² gains from circle or line beyond orthography+commonness are ≤ 0.04 in every cell: the tonal coordinates are small
partial effects riding on an orthographic/commonness-dominated geometry (ortho-only CV R² 0.1–0.55).

### 10:13 H4 context-contrast test (phase2/context_contrast.py; a-priori middle-half layer band; 2000 template regroupings)
circle|line, relational(B,C,D,E) − (A,F): key token 1B +0.02 (p .32), Gemma −0.01 (.69), Qwen +0.04 (.12), 7B +0.06 (.09);
prompt-final 1B +0.04 (.06), Gemma −0.03 (.85), Qwen +0.07 (.02), 7B +0.05 (.08). Per family at the final position the
enharmonic family (B) has the highest circle|line in 1B (p .01), Qwen (+0.09) and 7B (+0.22, p .03), and E_modulation in
1B (p .03) and 7B (+0.21, p .05); Gemma shows nothing. Band-averaged partials are all within ±0.1: context effects on
the periodic component are small and only marginally significant in 3 of 4 models → H4 weakly supported at most.
Line at the key token: E_modulation is the highest family in Qwen (+0.21, p < .001), 7B (+0.21, p .04), 1B (+0.13, p .11)
→ a consistent (3/4 models) context effect: modulation/next-key prose strengthens the open coordinate in the key-name
token. ECI: no family effect except B_enharmonic in 1B (p .03) and 7B key token (p .01, but 7B ECI is ≥ 0.65 there).

### 10:14 word-spelling control (OLMo-2-1B) and an instrument caveat
With "-flat"/"-sharp" word spellings the enharmonic-twin collapse disappears at the key token (ECI 0.37–0.43, all n.s.),
and circle|line partials of +0.28…+0.41 that pass the free relabeling null (p .006–.036) FAIL the glyph-preserving null
(p .09–.71): the shared "-flat"/"-sharp" tokens create a glyph block collinear with the line, and the free null is the
wrong null there. Caveat discovered: the 'last' position is the accidental token for 2-token keys (Cb/Gb/Bb/F#/C# in
OLMo/Qwen; F#/C# in Gemma) and the letter token for 1-token keys, so key-token geometries mix token types across
keys. A span-mean position is being extracted for all four models (extract15 --v2) and will be reported alongside;
the prompt-final position (same token for every key) is unaffected.

### 10:17 Track 8 respelling decomposition (21 labels: 9 respelled tonic pairs + D, G, A; phase2/respell_decomp.py)
Key token, layers at ½ and ¾ depth, OLMo-1B / Gemma / Qwen: mean percentile rank of the 9 twin distances 0.49–0.73 versus
letter-adjacent cross-glyph control pairs 0.58–0.67 (random-9-pair p ≥ 0.42) — with rare spellings included (B#, E#,
Fb, D#, G#, A#) enharmonic twins are NOT close in any model; the 15-key ECI signal in OLMo-1B (standard spellings) is
diluted or absent here, so it is at most a standard-spelling effect. Cross-fitted 8-dim subspaces: the "spelling"
subspace (span of twin differences fitted on the other five families) captures 0.53–0.73 of the between-label variance
in the held-out family (random 8-dim: 0.004) and the "semantic" subspace (span of twin means) 0.46–0.66 — heavily
overlapping, i.e. neither is a clean factor; after projecting out the cross-fitted spelling subspace the twin ECI drops
to 0.03–0.32. Reading: spelling directions are context-general (they transfer across families), and what is left after
removing them treats twins as near-identical, but the two "subspaces" as fitted are not separable factors (their
variance shares sum to > 1). H5 (separable circle/line/orthography subspaces) is not supported in this form.

### 10:17 Track 11 arbitrary output codes — first pass (phase2/codebook.py; header listed in LINE order — flagged)
Identity check (code for X major): OLMo-1B 0.20–0.27 (cannot use a 15-entry codebook; excluded), Gemma 0.53–0.87,
Qwen 0.73–1.00, 7B pending. Under codes (partials control commonness only): Qwen dominant/chord/enharmonic tasks
line|circle +0.36…+0.60 (p < 0.01), circle|line −0.12…−0.27; enharmonic twins far apart (ECI 0.70–0.90) even when the
prompt asks for the enharmonic equivalent; Gemma similar (+0.25…+0.55). Modulation task ≈ 0 both ways.
CAVEAT noticed immediately: the codebook header listed the 15 keys in signed line order, so list-position proximity can
manufacture a line. Re-running with a randomized header order (three codebooks per model) before drawing any
conclusion. Written before seeing the randomized results.

### 10:19 Track 5 — 15-key sequence-level behaviour (phase2/behavior15.py + behavior_fit.py; results/phase2/behavior/summary.txt)
Scorers pre-registered: total log-prob, length-normalized (mean per token), enharmonic-merged mass (B/C/D only; it
collapses twins by construction, so its ECI is not evidence and its circle|line is a 'neutral-pitch reading').
Total scorer, mean over 4 templates (line|circle / circle|line; #templates significant under glyph-preserving null):
 OLMo-1B: B +0.30/+0.07 (4/4, 1/4) C +0.33/−0.03 D +0.27/−0.10 E +0.44/+0.02;  Gemma: B +0.50/−0.14 C +0.32/0.00
 D +0.36/−0.14 E +0.54/−0.07;  Qwen: B +0.31/+0.11 C +0.42/+0.11 D +0.40/−0.01 E +0.52/+0.07;
 OLMo-7B: B +0.14/+0.28 (3/4, 3/4) C +0.38/+0.16 (4/4, 4/4) D +0.50/+0.13 (4/4, 3/4) E +0.56/+0.13 (4/4, 3/4).
Length-normalized scoring lowers line|circle by 0.05–0.15 and lowers ECI (twins less far) but changes no sign.
Enharmonic collapse in behaviour (ECI, total/mean): OLMo-7B B_enharmonic 0.11/0.05 — the model answers the enharmonic
twin (top-1 interval +12 in 8–9 of 15 keys); Qwen B 0.51/0.38 (partial); OLMo-1B 0.70/0.58 and Gemma 0.83/0.75 (fail).
Outside the B family every model keeps twins far apart (ECI 0.67–0.90). Top-1 intervals are ±1 line steps
(dominant/subdominant) in C/D/E for all models. Under merged scoring circle|line becomes +0.14…+0.52 in every model:
i.e. the seam of the line IS the spelling; once both spellings are pooled, what remains is circle-like.
Reading: H2/H6 — behaviour is spelling-sensitive (TPC line) in all models; OLMo-7B additionally has an NPC circle
component and knows enharmonic identity when asked; the merged-scorer result says the line is 'circle + a spelling
barrier', not a different metric.

### 10:20 7B respelling decomposition; 7B codebook OOM
OLMo-2-7B respell design (key token, L24/L31): twin ECI 0.56–0.71 vs controls 0.63–0.67 (p ≥ 0.73) — no enharmonic
collapse; cross-fitted spelling subspace captures 0.60–0.65 of held-out variance, semantic 0.50–0.55 (random 8-dim
0.002); ECI after removing the spelling subspace 0.04–0.17. Same pattern as the three smaller models.
The 7B line-order codebook run hit CUDA OOM (15-way batched scoring with CPU offload); scoring is now chunked; only the
randomized-order design will be run for 7B (the line-order design is the flawed one anyway).

### 10:21 Track 11 — randomized-header codebooks (Qwen done; Gemma/7B queued)
With the header in signed line order, list position and line distance are identical, so the first-pass 'line under
codes' was uninterpretable. With randomized header order (3 codebooks, identity accuracy 0.87–1.00 for Qwen), the
behaviour follows codebook LIST POSITION (partial +0.27…+0.65 controlling commonness, circle, line) and the line
partial controlling list position is small (dominant +0.04 / +0.23 (p .01) / +0.08; chord +0.09 / +0.15 / +0.03;
enharmonic ≈ 0); circle ≈ 0; ECI ≈ 0.5. Conclusion: under arbitrary output codes the models mostly answer by a prompt
heuristic (nearby list entries); neither the line nor the circle transfers robustly to code words. This does NOT show
the line is a surface-spelling artefact (the task is too weak to carry the relation), and it does not show the
opposite; Track 11 is inconclusive and recorded as such.

### 10:24 Gemma-2-2B randomized codebooks: list-position partial +0.35…+0.66, line|circle −0.11…+0.13 (all n.s.),
circle ≈ 0, ECI 0.30–0.69 — same as Qwen. 7B pending. Track 11 recorded as inconclusive in PHASE2_RESULTS §13.

### 10:36 OLMo-2-7B randomized codebooks (identity 0.87–1.00): list-position partial +0.17…+0.52; line|circle small
(dominant +0.04…+0.06 n.s.; chord +0.14/+0.22/+0.23, p .07/.02/.02); circle|line ≈ 0 except chord cb0 +0.16 (p .04);
ECI 0.29–0.37 in two codebooks, 0.65–0.81 in the third. A weak residual line in the chord task for 7B only; overall
Track 11 remains inconclusive (list-position heuristic dominates in all three models that can use a codebook).
The first span-mean geometry fits failed on a file-name mismatch (tag/spelling split); re-run with corrected names.

### 10:39 SPAN-MEAN KEY-TOKEN GEOMETRY (extract15 --v2; geometry_fit position 'mean') — corrects §3 of the draft
Averaging over all tokens of the key name instead of taking the last token: line|circle at the key token is ≈ 0 in
every model and family (OLMo-1B −0.01…+0.04; Qwen −0.01…+0.04; OLMo-7B +0.06…+0.12, p_max ≥ 0.20; Gemma +0.30 only at
layer 0 = input embeddings, glyph-preserving p 0.34–0.57). circle|line ≤ +0.12 everywhere (n.s.). ECI 0.19–0.21
(OLMo-1B, p_min .056–.086), 0.20–0.28 (Qwen, p .09–.21), 0.21–0.37 (7B, p .10–.41), 0.25–0.40 (Gemma) against control
pairs ≈ 0.55: a weak, marginal enharmonic-twin proximity, consistent across models and families but not individually
significant after layer correction.
=> The 'last'-token line in Qwen/7B across all six families (draft §3) was carried by the accidental token being the
final token of the 2-token keys (Cb, Gb, Bb, F#, C#); it is withdrawn as a key-name-geometry claim. The span-mean has
its own bias (it pulls 2-token keys toward their letter-token twin, controlled by same_letter/alphabet). The prompt-
final position is the only token-uniform probe and its results stand (7B circle in relational families; Gemma line at
final in B/C/D/E; Qwen circle in B/E; 1B line in E).

### 10:40 H4 contrast at the span-mean position: all family contrasts within ±0.03 (p .02–.77, effect sizes
negligible); the key-token "modulation → line" effect (Qwen +0.21 at the last token) is −0.10 at the span mean, i.e. it was
a last-token effect. Key-token context claims withdrawn; prompt-final-position and behavioural claims retained.
