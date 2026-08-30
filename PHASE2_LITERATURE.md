# PHASE2_LITERATURE.md — expanded audit for Phase II (what was read, what it establishes, what is left for us)

Legend: **Read** = full text or substantial excerpt read tonight; **Abstract/summary** = abstract plus secondary
summaries; **Prior knowledge** = not re-read tonight, cited from memory and to be checked before any submission.

## Music theory / cognition of tonal space

**Temperley, *The Line of Fifths* (Music Analysis 19/3, 2000) — Read (pp. 289–294 of the author's PDF).**
Defines *neutral pitch classes* (NPCs, 12 classes under octave + enharmonic equivalence, the object of pc-set
theory) versus *tonal pitch classes* (TPCs: octave equivalence only, so Ab ≠ G#). The line of fifths is "similar to
the circle of fifths, except that it extends infinitely in either direction" — Weber's 1821 spiral straightened;
Riemann's Tafel and Longuet-Higgins' space contain it as one axis. TPC identification is an early cognitive stage:
"a pitch of NPC 0 can be TPC C, B# or Dbb"; the *pitch variance rule* prefers labelling nearby events close together on
the line, formalized by minimizing line-of-fifths variance around a decaying centre of gravity. Spelling is claimed
to be cognitively real, inferred from context, and consequential for key-finding. **For us**: the "line" we measure
in model predictions is a legitimate cognitive object (TPC space), not an artifact by definition; the circle is
its mod-12 (NPC) projection. Our question becomes *which of the two spaces a given corpus statistic or model task
lives in*, which is exactly Temperley's NPC/TPC distinction.

**Moss, Neuwirth & Rohrmeier, *The line of fifths and the co-evolution of tonal pitch-classes* (J. Math. & Music 17,
2023) — Abstract/summary only (publisher PDF paywalled; author page and code repo DCMLab/lineoffifths consulted).**
Dimensionality reduction on a large historical corpus of tonal-pitch-class counts (c. 1360–1940, TP3C corpus)
recovers the line of fifths as the fundamental organizing dimension of tonal material, with a historically growing
exploitation of the fifths range and a notion of pitch-class co-evolution. TPCs are integer positions on the line;
NPCs are their mod-12 reduction; MIDI-style neutral encodings discard the distinction, so what is recoverable
(line vs circle) depends on the representation chosen. **For us**: text is spelled (TPC-encoded); the choice between
symbol/word spellings and enharmonic merging is the same representational decision, and our 15-key design is a
TPC design whose mod-12 projection is the NPC circle.

**Krumhansl & Kessler (Psych. Review 1982) — Prior knowledge; profile values used in phase2/keys15.py.** Probe-tone
profiles for major and minor keys; inter-key distances from profile correlations; MDS recovers a torus on which
fifth-related, relative and parallel keys are near. Not "the circle of fifths": for major keys alone the KK
similarity is a monotone function of circular fifths distance (ρ = 0.99 with our circle geometry), so it only
becomes a distinct hypothesis when minor keys are added (relative/parallel relations). Enharmonic equivalence is
assumed (NPC space).

**Krumhansl & Toiviainen, *Tonal Cognition* (Ann. NY Acad. Sci. 2001) — Abstract/summary.** Self-organizing-map model
trained on the profiles recovers the same toroidal key map (neighbours on the circle of fifths; parallel and relative
pairs proximal). Establishes that a neural model given NPC-space input recovers the torus.

**Chew, *Towards a Mathematical Model of Tonality* (MIT 2000) / Spiral Array — Abstract/summary + model implemented.**
Pitches on a helix P(k) = (r sin kπ/2, r cos kπ/2, kh), k the line-of-fifths index; chords and keys as weighted
centres; the model "deliberately does not assume enharmonic equivalence" — it is an open TPC space with a
quarter-turn per fifth. For the 15 major keys with standard weights the spiral-array key distance is ρ = 1.00 with
the plain line (monotone in |Δk| over the ranges we use), so it is a *shaped* line, not a new hypothesis, until
chords/minor keys are included.

**Sadek & Bakarji, PMLR 303 (2026) — Abstract.** Autoencoder on the WTC recovers circle-of-fifths key geometry from
symbolic music. "A neural network contains a circle of fifths" is therefore not novel; our object is *text* and the
NPC/TPC question.

## Statistics → geometry

**Karkada et al. (ICML 2026) — Read in Phase I.** Translation symmetry of M* ⇒ Fourier geometry of word embeddings;
periodic vs open boundary conditions treated separately (Cor. 2 vs Prop. 3). Our Phase-I finding: M* saturates for
keys; PMI carries the structure; the corpus block is periodic (NPC-like), the models' predictive statistic is open
(TPC-like). Karkada's LLM claims are qualitative and made on months/years/states only.

**Levy & Goldberg (NeurIPS 2014) — Prior knowledge.** SGNS implicitly factorizes shifted PMI (word × context
matrices); it justifies PMI as *the* embedding-model target, but it says nothing about transformer residual states
or about *conditional* next-token distributions, which are the object of the language-model objective. The
distinction between an association (PMI) matrix and a conditional-probability objective is exactly the Track-6
question: symmetric association may be periodic while the conditional prediction is spelling-sensitive.

**Modell, Rubin-Delanchy & Whiteley (2025) — Skimmed in Phase I.** Cosine similarity encodes on-manifold geodesic
distance; complementary.

## Fourier features ≠ computation; instrument cautions

**Nanda et al. (ICLR 2023); Zhong et al., *Clock and Pizza* (NeurIPS 2023) — Prior knowledge.** Fourier features on
cyclic groups arise in transformers trained on modular arithmetic, and identical-looking Fourier structure can be
produced by different algorithms. **Feucht et al. (2026)** and the **Convergent-Evolution** paper (2604.20817) extend
the caution to natural cyclic concepts. **Our Track-7 point is narrower**: a *non-cyclic categorical* feature
(accidental glyph) aliases into a specific harmonic of the semantic labelling, producing a false-positive "circle of
fifths" in a realistic LLM experiment; the mathematics is elementary (DFT of an indicator), the demonstration is new.

## Semantic / string entanglement and context-dependent geometry

**Marjieh, Veselovsky, Griffiths & Sucholutsky, *What is a Number, That a LLM May Know It?* (2025) — Abstract + full-text
extraction.** Six LLMs (GPT-4o, Llama-3.1 8B/70B, DeepSeek-V3, Claude-3.5-Sonnet, Mixtral-8x22B); similarity
judgments over integer pairs fitted by s = α + β·Levenshtein + γ·LogLinear (R² 0.73 combined vs 0.61 log-linear vs
0.21 Levenshtein alone); string vs integer framing shifts the balance (R² 0.62 vs 0.72) but does not remove the
string term; linear probes on Llama-3.1-8B residuals track log-linear (0.92) and Levenshtein (0.65) distances;
downstream errors (string-bias 37% for Llama-8B). **Relation to us**: our accidental-glyph block is the "Levenshtein
term" of key names and our circle/line are two *competing semantic* terms — the analogue of their single log-linear
term. Our contribution must be that (i) the *semantic* space itself is ambiguous between two theory-motivated
geometries (NPC circle vs TPC line) and (ii) context selects between them — not merely that form and meaning
coexist.

**Hu, Niu & Varma, *Language Models Represent and Transform Concepts with Shared Geometry* (arXiv:2607.04525, July
2026) — Abstract + full-text extraction.** 23 models in six families; concepts as point-cloud manifolds; context
displacement vectors r(w,τ) − r(w,τ₀) for categorization/perception/situation/affect/knowledge framings; CKA and
Grassmann alignment show displacement structure is shared across models (transported displacements predict held-out
ones above chance; alignment correlates with MMLU, r ≈ 0.78–0.81); displacement magnitude anti-correlates with
lexical density, direction variability with abstractness. No music, no cyclic sets, no form/meaning distinction.
**Boundary**: "context transforms concept geometry, and the transformation is shared across models" is taken. What
would be new here is that the context-conditioned axes coincide with *independently specified competing tonal
spaces* (NPC circle vs TPC line vs orthography) and predict candidate-sequence behaviour.

**Gurnee & Tegmark (ICLR 2024) — Prior knowledge.** Standards for a "represented coordinate": linear probes that
generalize across prompt templates and entity types, robustness to prompt variation, causal relevance. Our nulls
(free + block-preserving relabeling, max-over-layers) and template-holdout are the corresponding standard here.

## Searches (tonight)
- "language model pitch spelling / tonal pitch class / line of fifths LLM" → only symbolic-music/MIR work (TP3C corpus,
  Tonal Diffusion Model, Tonnetz coherence) and audio-LLM pitch benchmarks; nothing on text LLMs and TPC/NPC.
- Krumhansl–Toiviainen, Chew, Temperley located as above.

## Narrow novelty statement — after Phase II (what was earned; see PHASE2_RESULTS.md §15)
Earned: *next-key candidate-sequence behaviour of four text-trained LLMs over 15 spelled major keys follows the open
tonal-pitch-class line (Temperley), whereas the training-like corpus statistic is periodic (neutral pitch classes) in
every context class we can condition on; enharmonic identity is known by one model (OLMo-2-7B).* Not earned (dropped):
"three separately identifiable coordinate systems" (subspaces overlap; H5 unsupported), "context selects which one
dominates hidden-state geometry" (key-token effects were probe artefacts; only small predicting-position effects
remain), and any claim that the key-name geometry itself carries a tonal coordinate. Methods notes: Fourier aliasing of
categorical features; last-token vs span-mean probe bias for multi-token concept names.
