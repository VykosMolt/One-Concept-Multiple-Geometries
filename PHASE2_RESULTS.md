# PHASE2_RESULTS.md

**MIXED / MODEL-SPECIFIC** — with one clean positive (the open tonal-pitch-class line is the dominant *tonal*
structure in next-key behaviour in all four models), one clean negative (the corpus contains no task-conditioned line;
every cue class of Wikipedia is periodic), one instrument correction of my own (the "line in key-name geometry" of the
two larger models was a last-token artefact: it vanishes when the key name is probed by its span mean), a small and
position-specific context effect (relational contexts nudge the periodic component at the token-uniform prompt-final
position, clearly only in OLMo-2-7B), and no evidence for separable circle/line/orthography subspaces. Enharmonic
identity — the defining property of the neutral-pitch-class circle — is known behaviourally by OLMo-2-7B only, and in
key-name geometry appears at most as a marginal twin proximity (p ≈ .06–.2) once orthographic controls and a token-
uniform probe are used.

*(Phase I frozen at tag `phase1-final`. Phase-II code in `phase2/`, results in `results/phase2/`, figures in
`figures/phase2/`, chronology in `PHASE2_LOG.md`, literature in `PHASE2_LITERATURE.md`. Every statistic below has a
relabeling null — free and glyph-class-preserving — and best-over-layers values carry max-over-layers p-values;
context contrasts use template-regrouping nulls on an a-priori layer band. All queued runs finished; nothing is pending.)*

## 1. What survived Phase I (starting point)
Robust periodic fifths structure in Wikipedia key co-occurrence (partial +0.62/+0.66, p < 0.001, two null families;
theory embedding +0.52); M\* saturation and the diagonal-dependence of the naive P5 profile; key-name geometry largely
orthographic (accidental-glyph feature, 79.6% of its Fourier energy in the fifth pair under the semitone labelling);
only OLMo-2-7B with a layer-corrected fifths signal in key-name geometry; a spelling-sensitive open line in next-key
prediction (≈⅔ surviving enharmonic merging); chord-progression contexts more circle-like in Qwen/7B.

## 2. What the reviews killed (not resurrected here)
"Circle→line transition", "predicting-position locus", "scaling trend", "black/white-key semantic feature",
"P5 means circle", "topological transition". None reappears below; the 15-key design was chosen precisely because the
12-key design cannot separate circle from line.

## 3. The 15-key enharmonic test (Track 1)
Design: 15 standard major-key spellings on the signed line of fifths (Cb … C#, s = −7…+7); the pairs (Cb,B), (Gb,F#),
(Db,C#) share a neutral pitch class. Circle distance 0 vs line distance 12 on these pairs; over the 105 pairs the circle
and line geometries correlate only ρ = 0.25 (vs 0.68 in the 12-key design). Krumhansl–Kessler major–major similarity is
ρ = 0.99 with the circle and Chew's spiral array ρ = 1.00 with the line for these 15 keys, so the classical spaces add no
discrimination until minor keys are added (§9). Orthographic geometries (glyph class, edit distance, letter identity,
alphabet distance, token count) and commonness are always in the control set; the "enharmonic collapse index" ECI is the
mean percentile rank of the three twin distances (0.5 = null), compared against 15 letter-adjacent, cross-glyph,
non-enharmonic control pairs (Ab–G, Eb–D, Bb–A, …), because every enharmonic pair is also such a pair.

**Key-name geometry (symbol spellings; 4 templates × 6 context families; results/phase2/geometry_*.log, fig
`heatmap15_symbol.png`).** Max-over-layers relabeling p in parentheses.

| model | line\|circle at key token (range over families) | circle\|line at key token | ECI at key token (best family) | circle\|line at prompt-final (significant cells) |
|---|---|---|---|---|
| OLMo-2-1B | +0.09…+0.30; sig. in E (p .006), F (.018), B (.03) | ≤ +0.20 (n.s.) | 0.15–0.17 in A/D/F (p .03–.04; controls 0.58–0.66) | B +0.27 (p .026) |
| Gemma-2-2B | +0.10…+0.24; B marginal (p .05) | ≤ +0.17 (n.s.) | 0.20 (A, p .12) | none |
| Qwen2.5-3B | +0.26…+0.37; sig. in all six (p ≤ .034) | C +0.30 (p .022), B +0.24 (p .05) | 0.16 (A, p .07), 0.18 (C, p .08) | B +0.28 (p .018), E +0.27 (p .02) |
| OLMo-2-7B | +0.30…+0.34; sig. in all six (p ≤ .014) | ≤ +0.19 (n.s.) | 0.35–0.48 (n.s.) | A +0.26, B +0.32, C +0.31, D +0.36, E +0.36, F +0.21 (p .004–.03) |

Cross-validated (leave-one-key-out) rank regression: orthography + commonness explain R²_cv 0.1–0.55 of pair
distances; adding the circle or the line raises R²_cv by ≤ 0.04 in every cell. The tonal coordinates are small partial
effects on an orthography-dominated geometry.

**Span-mean probe (the token-uniform correction; results/phase2/geometry_*_mean.log, fig
`keytoken_last_vs_mean.png`).** The table above uses the last token of the key name, which is the accidental token
for 2-token keys (Cb, Gb, Bb, F#, C# in OLMo/Qwen; F#, C# in Gemma) and the letter token otherwise. Averaging over the
key-name span instead: line|circle ≈ 0 in every model and family (OLMo-1B −0.01…+0.04; Qwen −0.01…+0.04; OLMo-7B
≤ +0.12, p ≥ .20; Gemma +0.30 only at layer 0/input embeddings, glyph-preserving p ≥ .34); circle|line ≤ +0.12
(n.s.); ECI 0.19–0.28 (OLMo-1B, Qwen), 0.21–0.37 (7B), 0.25–0.40 (Gemma) against control pairs ≈ 0.55, p_min .06–.4.
**The "line in key-name geometry across all six families" for Qwen/7B is therefore withdrawn**: it was carried by the
accidental token. What remains at the key name is orthography, commonness, and a weak, consistent-in-sign but
non-significant enharmonic-twin proximity. (The span mean has the opposite bias — it pulls 2-token keys toward their
letter token — which is why same-letter and alphabet distance are in every control set.)

**Word spellings ("C-flat major") as a control (OLMo-1B, Qwen):** the twin collapse disappears (ECI 0.37–0.43) and
apparent circle partials of +0.28…+0.41 pass the free null but fail the glyph-preserving null (p .09–.71): shared
"-flat"/"-sharp" tokens make the glyph block collinear with the line. **Instrument caveat:** the key-token position is
the accidental token for 2-token keys and the letter token for 1-token keys; a span-mean position was extracted for
all four models (next paragraph) and the prompt-final position is unaffected.

**Enharmonic identity in geometry is model- and spelling-specific.** OLMo-2-1B (symbol spellings, layer 15) places the
three twins among the closest 15% of pairs while the orthographic control pairs sit at 0.6; Qwen/Gemma show a marginal
version; OLMo-2-7B none. In the 21-label respelling design (9 twin pairs including rare B#/E#/Fb/D#/G#/A#), twins are
not close in any model (ECI 0.49–0.73 ≈ controls).

## 4. Context factorial (Track 2) — H4 test with template-regrouping nulls
Families A spelling, B enharmonic identity, C harmonic relation, D chord progression, E modulation, F generic mention
(4 templates each, written before results). Statistic: family mean over an a-priori middle-half layer band; null: 2000
random regroupings of the 24 templates into 6 pseudo-families.

| model | circle\|line: relational (B–E) − (A,F), key token / final | line\|circle, highest family at key token | ECI family effect |
|---|---|---|---|
| OLMo-2-1B | +0.02 (p .32) / +0.04 (p .06); B highest at final (p .01), E (p .03) | E +0.13 (p .11) | B at key token (p .03); E at final (p .00) |
| Gemma-2-2B | −0.01 (.69) / −0.03 (.85) | none | none |
| Qwen2.5-3B | +0.04 (.12) / **+0.07 (.02)** | **E +0.21 (p < .001)** | none |
| OLMo-2-7B | +0.06 (.09) / +0.05 (.08); B +0.22 (p .03), E +0.21 (p .05) at final | **E +0.21 (p .04)** | B at key token (p .01) |

Reading: the periodic component is *nudged* upward by relational contexts at the token-uniform prompt-final
position in three models (band-averaged effect +0.04…+0.07; only Qwen reaches p < .05), and the enharmonic family is
where it is largest. The key-token effects in the table (including "modulation strengthens the line") do **not**
survive the span-mean probe: at the span mean every family contrast is within ±0.03 (Qwen E line +0.21 at the last
token → −0.10 at the span mean). H4 is therefore weakly supported at the predicting position only; Gemma shows no
context sensitivity anywhere. The Phase-I chord-progression circle effect is not reproduced as a *family* effect in
geometry (D_chord is not distinguished from C/E); see §6 for behaviour under merged scoring.

## 5. Circle vs line vs orthography decomposition (Tracks 3, 9)
Nested CV regression (§3) and cross-fitted subspaces (Track 8): an 8-dim "spelling" subspace fitted from twin
differences in five context families captures 0.53–0.73 of the between-label variance in the sixth family (random
8-dim: 0.002–0.004) and the "semantic" (twin-mean) subspace 0.46–0.66; the two overlap heavily (shares sum > 1). After
projecting out the cross-fitted spelling subspace, twin ECI falls to 0.03–0.32 in all models. So (i) spelling directions
are context-general (they transfer across families), (ii) circle/line/orthography are *not* separable subspaces at the
resolution of this design — H5 not supported; H3 supported (orthography aliases strongly and is the largest factor).

## 6. Sequence-level behaviour (Track 5; results/phase2/behavior/summary.txt, fig `behavior15.png`)
15 × 15 candidate-sequence matrices in families B, C, D, E; scorers: total log-prob, length-normalized, and
enharmonic-merged mass (B/C/D only; collapses twins by construction). Total scorer, family means; #templates
significant under the glyph-preserving null in brackets:
- line|circle: OLMo-1B +0.27…+0.44 [4/4 in all families]; Gemma +0.32…+0.54 [4/4]; Qwen +0.31…+0.52 [4/4];
  OLMo-7B +0.14 (B) … +0.56 (E) [3–4/4].
- circle|line: ≈ 0 for OLMo-1B, Gemma, Qwen (−0.14…+0.11; ≤ 1/4 significant except Qwen C 3/4 at +0.11); OLMo-7B
  +0.13…+0.28 [3–4/4 in all families].
- Enharmonic identity in behaviour ("the enharmonic equivalent of X major is"): OLMo-7B answers the twin (ECI 0.11
  total / 0.05 length-normalized; top-1 interval +12 in 8–9 of 15 keys); Qwen partially (0.51/0.38); OLMo-1B (0.70) and
  Gemma (0.83) do not. Outside family B every model keeps twins far apart (ECI 0.67–0.90).
- Length normalization lowers line|circle by 0.05–0.15 and brings twins closer, never changes a sign.
- Merged scoring turns circle|line positive in every model (+0.14…+0.52): the line is "circle plus a spelling barrier",
  not a different metric — the Phase-I "≈⅔ survives merging" statement is the same fact from the other side.
Top-1 answers in C/D/E are ±1 line steps (dominant/subdominant) in all models.

## 7. Corpus comparison (Tracks 5, 6)
- 15-key Wikipedia PMI: the three enharmonic pairs are the *closest* pairs of all (ECI 0.04; PMI 9.9–11.0 vs median
  7.3); circle|line +0.42, line|circle +0.38 (both p < .001, two nulls; P(circle > line) 0.89 by bootstrap). The spelled
  corpus statistic is both periodic and open.
- But a targeted document scan finds only **41** co-mentions of enharmonic pairs in all of Wikipedia; 49% contain the
  word "enharmonic" and most others are slash notation ("Gb major/F# major") or harp-notation prose. The corpus's
  NPC identity at the seam is metalinguistic (statements of equivalence), not harmonic usage.
- Cue-conditioned statistics (ordered pairs within 24 words split by intervening cue words): every class is circle-
  dominant — all +0.59/+0.23, other +0.47/+0.19, chord +0.64/+0.21, **key-signature prose +0.68/0.00**; modulation prose
  (327 pairs) +0.25/+0.10 is too sparse. **The corpus contains no task-conditioned line** (H9/"corpus contains both"
  not supported by this construction). The models' open coordinate has no counterpart in Wikipedia's conditional
  statistics that we could find; it is model-side.

## 8. Strongest null-corrected statistics
Corpus circle (Phase I merged family) partial +0.66, p < .001 both nulls; 15-key corpus ECI 0.04, p < .001.
Behaviour line|circle: 4/4 templates significant under the glyph-preserving null in 14 of 16 model×family cells.
Geometry at the token-uniform prompt-final position: OLMo-7B circle|line +0.31…+0.36 in C/D/E (p ≤ .012) and
+0.32 in B (p .006); Gemma line|circle +0.34…+0.44 in B/C/E (p ≤ .02). OLMo-7B behavioural enharmonic identity: ECI
0.05–0.11 (null 0.5). Key-token geometry: nothing survives the span-mean probe. Everything else is p ≈ .02–.1 or null.

## 9. Major/minor extension (Track 4)
Not run. Justification: enharmonic identity is known behaviourally by one model only, and Krumhansl–Kessler /
spiral-array geometries are indistinguishable from circle/line on major keys; adding 15 minor keys would double the
extraction cost without a model that demonstrably knows relative/parallel relations (Phase-I few-shot: relative-minor
accuracy 0.08–0.83, only 7B above 0.5). Recommended as the first follow-up with 7B only.

## 10. Causal work (Track 10)
Not attempted: the prerequisite (a robust, replicated context contrast in geometry) was met only weakly (§4).

## 11. Track 7 — Fourier aliasing, formalized (phase2/aliasing.py; MATH.md §3, §10)
For any feature f over Z_12, F_k = Σ_x f(x) e^{−2πikx/12}; a contiguous block of width w on a 12-cycle puts 18/37/55/70/
80/83% (w = 1…6) of its centered energy in the fundamental pair; the accidental indicator is the w = 5 block in fifths
coordinates, hence 79.6% in the k = 5/7 pair under semitone labelling. Across all 4094 binary partitions of 12 labels,
29% put > 50% of their energy into a single paired mode and 6.6% > 70%; random 3-/4-class partitions 8.5% / 2.7%;
root-letter identity under semitone labelling puts 34% into the *chromatic* pair. Statement (elementary, but the
false-positive demonstration is the point): *Fourier power measures variation with respect to a chosen group labelling
of the concepts; it does not identify the source of that variation, and common categorical features alias into the
fundamental of whichever ordering makes them contiguous.*

## 12. Late results
**Arbitrary output codes (Track 11), randomized headers, three codebooks per model.** Models that can use a 15-entry
codebook (Qwen identity 0.87–1.00, Gemma 0.60–0.73, OLMo-7B 0.87–1.00; OLMo-1B 0.2, excluded) answer dominant /
chord / enharmonic / modulation questions mostly by *codebook list position* (partial +0.17…+0.66 after controlling
commonness, circle and line). Controlling list position, the line partial is ≈ 0 for Qwen and Gemma (one Qwen codebook
+0.23, p .01) and small for OLMo-7B (chord task +0.14/+0.22/+0.23, p .07/.02/.02; dominant ≈ +0.05); the circle is ≈ 0
throughout (one 7B chord codebook +0.16, p .04). A first pass with the header in line order had shown a strong "line"
that was entirely list position; it is discarded. Track 11 is inconclusive: removing spelled outputs removes most
tonal structure along with the spelling, but a prompt heuristic dominates the task, so this neither supports nor
refutes H6.
**Span-mean key-token geometry:** done; see §3 (it withdrew the key-token line claim).

## 13. Strongest evidence against the preferred interpretation
- The dominant geometry everywhere is orthographic + commonness; circle and line add ≤ 0.04 CV-R². A sceptic can say
  the tonal coordinates are decoration on a string-feature space.
- Key-name geometry depends on the probe convention: last token (accidental-token bias) shows a line, span mean
  (letter-token bias) shows nothing; neither is neutral. Only the prompt-final position and behaviour are probe-clean.
- Context effects are small (≤ 0.07 band-averaged) and only Qwen reaches p < .05; Gemma shows none.
- The corpus 15-key circle at the seam is carried by ~41 explicit equivalence mentions.
- The arbitrary-code test (Track 11) is dominated by a list-position heuristic; it neither confirms nor refutes that
  the line is a spelling effect.
- Four models, one seed of templates per family, 15 (or 105-pair) designs: power is limited; several "consistent
  across three models" statements are consistent in sign, not individually significant.

## 14. Relationship to prior work
- **Karkada et al.**: their statistic → embedding-geometry mapping holds for the periodic corpus block; transformer
  key-name geometry follows their block only through the orthographic/commonness components; the open coordinate the
  models use for prediction has no counterpart in the symmetric or the cue-conditioned corpus statistics.
- **Temperley / Moss et al.**: the models' behavioural space is the tonal-pitch-class line (spelling-sensitive), the
  corpus's is the neutral-pitch-class circle — the two spaces they distinguish, here recovered from text; but Moss et
  al.'s warning that the representation format decides what is recoverable applies to us twice over (spelled vs
  merged targets; last-token vs span-mean probes).
- **Marjieh et al.**: same form/meaning entanglement pattern (accidental glyph ≈ their Levenshtein term); our added
  point is that the "meaning" term is itself ambiguous between two theory-motivated spaces and that the open one wins
  in behaviour for all four models.
- **Krumhansl–Kessler / Chew**: not discriminable from circle/line on major keys; untested for minor keys.
- **Hu et al. 2026**: context transforms representations here too, but the transformations are small and only the
  modulation→line effect replicates across three models; we make no shared-geometry claim.

## 15. Narrowest defensible novelty claim
*In four text-trained LLMs the next-key behaviour over 15 spelled major keys follows Temperley's open line of fifths
(spelling-sensitive, with its seam at the enharmonic pairs) rather than the neutral-pitch-class circle that the
training-like corpus statistic exhibits in every context class; the key-name residual geometry itself carries no
robust tonal coordinate once orthography, frequency and token-position artefacts are controlled; enharmonic identity
is known behaviourally by the 7B model only, and only that model shows a periodic component at the predicting
position in relational contexts.* The Fourier-aliasing false positive (and now the last-token-position false positive)
are methods notes.

## 16. Paper or technical note?
A technical note with one solid, well-controlled behavioural finding (line, not circle; corpus periodic) and a methods
warning; not a paper. It would become a paper with (a) a model that knows enharmonic identity in more than one family
(7B+ scale, or instruction-tuned), (b) a synthetic tokenization-free replication, and (c) a causal test on the line.

## 17. Single most informative next experiment
Train a small transformer on a synthetic spelled corpus with a *controlled* periodic (NPC) co-occurrence kernel and
spelled outputs, then test whether next-key prediction acquires the open line purely from the output vocabulary being
spelled (H6) — this is the only way to settle whether the line is a property of prediction over spelled labels or of
learned tonal knowledge, and it removes every orthographic confound at the source.

*(Correction 2026-08-29, found by the second paper review: the "29% of the 4094 binary partitions put > 50% of their energy
into a single paired mode" figure in §11 was a floating-point tie artefact — 276 partitions sit exactly at 0.5. Exact
counts: 24.7% strictly above one half, 6.7% exactly at one half, 31.4% at or above; the 6.6% > 70% figure is exact.
`phase2/aliasing.py` is now tie-aware; `results/phase2/aliasing/summary.txt` regenerated.)*
