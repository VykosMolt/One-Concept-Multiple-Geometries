# LITERATURE_AUDIT.md — what was actually read/checked

(Running note; updated through the night. "Read" = full text fetched and relevant sections
read verbatim; "Skimmed" = abstract + targeted grep of full text; "Search only" = title/abstract
from search results.)

## Core theory

### Karkada, Korchinski, Nava, Wyart, Bahri — *Symmetry in language statistics shapes the geometry of model representations* (arXiv:2602.15029, ICML 2026). **Read** (main text §2–3, App. A.2–A.3, B.1–B.3, E.2 verbatim from arXiv HTML).
- M* = (P_ij − P_iP_j)/(½(P_ij+P_iP_j)), Wikipedia 20231101.en, V=25k, L=16, f(d)=L+1−d (Eq. 1, 19).
- Theory for word2vec-like models: W W^T = |M*|; centered S-block Gram = P|M*|_S P; Fourier
  geometry with amplitudes √|λ_k| (Prop. 1, App. E.2). Only *relative* amplitudes predicted (B.3).
- LLM evidence: Gemma-2-2B, prompt "The month of the year is x", final-token residual, PCA pictures
  only; no quantitative Gram-vs-M* metric for the LLM; Limitations: contextual sharpening unexplained.
- Concept families: months, years, US states. **No music, no competing generators.** Nothing in the
  paper addresses a family with two automorphic orderings; our test is out-of-sample for them.
- Fig. 4: month circle survives ablating the month–month block (helper words). Directly relevant
  caveat: a 12×12 block can under-determine the geometry.

### Karkada, Simon, Bahri, DeWeese — *Closed-form training dynamics reveal learned features and linear structure in word2vec-like models* (arXiv:2502.09863). **Skimmed** (abstract; the |Λ| / asymmetric-factorization result is restated in 2602.15029 App. B, which I read).

### Prieto, Stevinson, Barsbey, Birdal, Mediano — *From Data Statistics to Feature Geometry: How Correlations Shape Superposition* (arXiv:2603.09972). **Skimmed** (abstract). Toy model (BOWS) trained on bag-of-words statistics; correlated features → constructive interference → clusters and cycles; Karkada cite them for cyclic month correlations in Wikipedia and "autoencoders approximately perform PCA". Same mechanism class (pairwise statistics → geometry). Does not address competing orderings.

### Engels, Michaud, Liao, Gurnee, Tegmark — *Not All Language Model Features Are One-Dimensionally Linear* (arXiv:2405.14860, ICLR 2025). **Skimmed** (abstract + summary). Circular weekday/month features in GPT-2 and Mistral-7B via SAEs, verified by PCA and interventions on modular arithmetic tasks. Establishes the phenomenon we use as a positive control; no music.

### Feucht et al. — *Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts* (arXiv:2605.01148). **Skimmed** (abstract). Cyclic representation ≠ cyclic computation: Llama-3.1-8B computes base-10 addition then maps back. Reason we never claim "the model reasons around the circle of fifths" from geometry alone.

### Modell, Rubin-Delanchy, Whiteley — *The Origins of Representation Manifolds in LLMs* (arXiv:2505.18235). **Skimmed**. Cosine similarity encodes on-manifold geodesic distance. Complementary to Karkada; no corpus-statistics prediction of harmonic content.

## Adjacent / newer work found by search

- **Du, He, Su — *Uncovering Symmetry Transfer in LLMs via Layer-Peeled Optimization* (arXiv:2605.12756, May 2026).** Skimmed. Proves for a layer-peeled surrogate that cyclic-shift symmetry of target distributions forces circulant logit/Gram matrices. Again circulant ≠ which ordering; consistent with MATH.md §2.
- **Convergent Evolution: How Different Language Models... (arXiv:2604.20817, Apr 2026).** Skimmed §1, §3. Fourier spikes (periods 2,5,10) appear in transformers, RNNs, LSTMs *and* word embeddings, but spectral energy dissociates from linear separability. Used as a caveat (MATH.md §8). Explicitly lists months/weekdays as "natural next step".
- **Singh & Chopra — *Geometry of Human Perceptual Domains Emerges Transiently in LLM Representations* (arXiv:2605.27970, May 2026).** Skimmed. "Pitch" here = frequency-in-Hz prompts ("...musical tone VALUE Hz"), evaluated by RSA against human similarity; finds arc-like ordinal structure at mid layers, attenuating later. Not pitch classes, not keys, no circle of fifths, no corpus statistics. Overlap with our project: only the observation that structure is mid-layer and transient.
- **Sadek & Bakarji — *The Circle of Fifths as Latent Geometry in Bach's WTC* (PMLR v303, 2026).** Search only. Autoencoder on symbolic music → CoF in PCA. Music-data, not language.
- **Chuan, Agres, Herremans — *From Context to Concept* (arXiv:1811.12408; NCA 2020).** Search only + prior knowledge. word2vec on polyphonic music slices → implicit circle of fifths from cosine distances. Music tokens, not natural language. Together with ChordRipple (Huang et al. 2016, chord2vec: major/minor chords arranged in CoF shapes) this establishes that *music-token* co-occurrence produces CoF geometry — expected because harmonic progressions move by fifths. Our question is whether *natural-language* co-occurrence does.
- **Madjiheurem et al. — chord2vec / Chord Embeddings (arXiv:2102.02917).** Search only. Same domain (music tokens).
- **Do Music Foundation Models Embed Pitch in Helical Structure? (arXiv:2607.29086).** Skimmed abstract. Audio MFMs, isolated notes → helix with octave periodicity. Audio, not text.
- **LLMs' Internal Perception of Symbolic Music (arXiv:2507.12808).** Skimmed abstract. Generation-based; not representation geometry.
- **Exploring the Internal Mechanisms of Music LLMs: Root and Quality (OpenReview)** and **MI-MIDI (arXiv:2608.06638)**: probing MusicGen / text-to-MIDI models for root/quality/key; one uses "DFT at ω=7" to project a 12-d key profile onto the circle of fifths — the standard music-theory DFT trick (Quinn/Amiot), same algebra as MATH.md §3 but applied to pitch-class *profiles*, not LLM hidden states.
- **Kantamneni & Tegmark — *Language Models Use Trigonometry to Do Addition* (arXiv:2502.00873)**; Zhou et al. *Pre-trained LLMs Use Fourier Features to Compute Addition* (arXiv:2406.03445). Prior knowledge. Number helices with periods 2,5,10,100. Methodology precedent for Fourier analysis along a concept axis.

## Prior-art verdict (so far)
- The Fourier-along-concept-axis measurement, the co-occurrence→geometry theory, and the LLM month circle are all established.
- CoF geometry in *music-trained* embeddings is established (Chuan et al., ChordRipple, Sadek & Bakarji).
- I found **no** prior work measuring pitch-class / key geometry in text-trained LLM hidden states, and **no** prior test of the Karkada theory on a family with two competing automorphic orderings. The "DFT at ω=7 / ω=1" distinction is standard in mathematical music theory (Quinn 2006/2007 "General equal-tempered harmony"; Amiot 2016 *Music Through Fourier Space*) — the novelty, if any, is the corpus-statistics prediction, not the Fourier bookkeeping.
- Weakening prior art: Karkada Fig. 4 (block ablation) means a 12×12 block prediction is not what the theory strictly says; and 2604.20817 warns spectral energy ≠ usable geometry.

## Late-night additional searches (after the black-key confound was found)
- Searched for prior discussion of categorical/contiguous-arc confounds in "Fourier feature" claims for
  cyclic concepts (months/weekdays): nothing found beyond the Convergent-Evolution paper's spectral-vs-
  separability dissociation (2604.20817) and Feucht et al.'s representation≠algorithm point. The specific
  point that a categorical feature contiguous on one ordering mimics the fundamental of the *competing*
  ordering appears to be unremarked in the LLM literature (it is elementary in Fourier analysis; and in
  mathematical music theory the DFT of the diatonic/black-key set is well known to peak at coefficient 5
  — Quinn 2006/7, Amiot 2016 — which is the same fact in pitch-class-set form).
- "What Makes a Good Layer?" (arXiv:2608.14819): audio MFMs; key probe via DFT at ω=7 — the same
  algebra, on audio, not text.
- "Language Models Learn Universal Representations of Numbers…" (arXiv:2510.26285): numbers only.
- No paper found probing key-name or pitch-class-name geometry in text LLMs.
- **Line of fifths vs circle of fifths.** Temperley, *The Line of Fifths* (Music Analysis 19(3), 2000)
  argues tonal pitch spelling lives on an open line of fifths rather than the enharmonic circle; the
  "line of fifths and the co-evolution of tonal pitch-classes" (J. Math. & Music, 2022, found in search)
  continues this. Prior knowledge + search only; relevant because the models' predictive geometry turned
  out to be the line (open BC), the corpus statistic the circle (periodic BC).

## Added after the independent scientific review (citations the review flagged as missing; prior knowledge, not re-read tonight)
- **Chew, *Towards a Mathematical Model of Tonality* (MIT PhD 2000) / "The Spiral Array" (2002).** Pitch classes and keys on an
  *open helix generated by fifths* — the geometric formalization of the open line-of-fifths vs closed circle distinction that
  our models exhibit. Should be cited next to Temperley 2000.
- **Krumhansl & Kessler (1982); Krumhansl, *Cognitive Foundations of Musical Pitch* (1990).** Probe-tone key space is a *torus*
  (circle of fifths × relative/parallel). MATH.md §4's "not a torus" is about a single 12-point orbit and does not contradict
  this, but the standard key torus must be acknowledged wherever "torus" is mentioned.
- **Levy & Goldberg (NeurIPS 2014), "Neural word embedding as implicit matrix factorization."** SGNS factorizes shifted PMI;
  this is the canonical justification for using PMI when Karkada's bounded M* saturates — the switch is not a concession.
- **Nanda et al. (ICLR 2023) "Progress measures for grokking"; Zhong, Liu, Tegmark, Andreas (NeurIPS 2023) "The Clock and
  the Pizza."** Origin of Fourier features on cyclic groups in transformers and the canonical demonstration that a
  Fourier-looking representation can implement a non-Fourier algorithm (stronger than our §8 caveat).
- **Marjieh, Sucholutsky, van Rijn, Jacoby, Griffiths (2023/24), "LLMs predict human sensory judgments across six modalities"**
  — includes pitch (elicited judgments, not residual streams). **Gurnee & Tegmark (ICLR 2024), "Language models represent
  space and time"** — general precedent for concept-axis geometry probing.
Novelty narrowing per the review: "first measurement of *key-name* residual-stream geometry in text LLMs" (not "pitch");
the contiguous-arc caution is a service note (textbook Quinn/Amiot in pitch-class-set form), not a contribution.

## Paper-stage additions (2026-08-29; abstracts/summaries only unless stated — to be read in full before submission)
- **Karkada et al. 2602.15029, Appendix D "Combined Model of Seasonality and Binary Semantic Attributes"** — verified from the
  arXiv HTML table of contents: a joint model in which a periodic attribute and binary semantic attributes coexist in
  the PMI structure. Cited in the paper for the point that binary attributes can occupy the cyclic sector.
- **Fu, Zhou, Belkin, Sharan, Jia — *Convergent Evolution: How Different Language Models Learn Similar Number
  Representations* (arXiv:2604.20817)** — authors verified from arXiv; the 2604.20817 entry above (title paraphrased
  then) is this paper. Cited for "Fourier sparsity is necessary but not sufficient for separability".
- **Wurgaft et al. 2026, *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior*
  (arXiv:2605.05115)** — abstract only: causal interventions along representation-manifold paths reproduce natural
  behaviour; cited as prior art for "representation geometry corresponds to behaviour geometry".
- **Shai et al. 2026, *Transformers learn factored representations* (arXiv:2602.02385)** — abstract only: factors in
  orthogonal residual subspaces when conditionally independent; cited as prior art for factored representations.
- **Brenner, Knösche, Scherf 2026, *Grid-World Representations in Transformers Reflect Predictive Geometry*
  (arXiv:2603.16689)** — abstract only: representations align with analytically derived predictive vectors; cited as
  prior art for predictive-statistics → geometry in a synthetic world.
- **Bassi & Tomar 2026, *Geometry of Ordinal Representations in Language Models* (arXiv:2607.04167)** — abstract only:
  1-D ordinal manifolds when the variable is locally computable from token identity; cited for ordinal geometry.
- **Park et al. 2026, *The Information Geometry of Softmax: Probing and Steering* (arXiv:2602.15293)** — abstract only;
  in refs.bib, not cited in the current draft.
- **Marjieh, Veselovsky, Griffiths, Sucholutsky 2025 (arXiv:2502.01540)** — ID verified; see PHASE2_LITERATURE.md.
- **Moss, Neuwirth, Rohrmeier 2023, J. Math. Music 17(2):173–197** — bibliographic data verified; still abstract/summary
  only (paywalled), so the paper attributes to it only the NPC/TPC encoding point recorded in PHASE2_LITERATURE.md.
