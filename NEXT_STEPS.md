# NEXT_STEPS.md (updated after the two independent reviews)

0. **Rename everywhere: the block is an accidental-glyph (orthographic) feature.** Any follow-up must use the
   decoupled spelling set (B#, Fb, E#, Cb) or a tokenization-free synthetic setup; see item 4.

Ordered by expected information per GPU-hour.

1. **Scale curve on one model family.** OLMo-2 1B → 7B → 13B/32B (CPU offload or 4-bit for the
   representation *sign* only), same 24 contexts, report token-geometry partial-fifths per layer,
   predictive-matrix partial-fifths, and few-shot relation accuracy. Tonight's prediction, revised
   after Qwen2.5-3B: relation accuracy and *predictive* partial-fifths rise with scale, while the
   key-name token geometry stays block-dominated with at most a transient mid-layer fifths
   component. If a 7B+ model puts a real circle (isotropic k=7, survives indicator projection,
   circle > line) into the token geometry, that falsifies tonight's reading.
1b. **Where does the relation live? (revised)** With matched contexts the fifths kernel is in the residual
   stream of *key-relation contexts* at both the key token and the predicting token (+0.33…+0.53, corrected
   p 0.01–0.06), line-like at the output; a circle-like intermediate stage is significant in one cell only
   (7B "modulates to"). Next: (i) design contexts that vary relation type (modulation vs chord progression —
   merged-target scoring shows chord contexts are circle-like in Qwen/7B) and test where the seam becomes a
   barrier; patch those layers and see whether seam-crossing answers (dominant of F#) change;
   (ii) unembedding-row geometry of the single-token keys; (iii) test whether the black/white block
   is causally used for relation answers (activation patching of the indicator direction);
   (iv) repeat with the *corpus* restricted to next-key statistics (directed, short-window
   co-occurrence "in X major … in Y major") to see if the predicting-state geometry matches a
   directional statistic better than the symmetric window.
2. **Predictive vs representational statistics** (started tonight, scripts/predictive_matrix.py):
   if P(next key | key) matches corpus PMI while the token geometry does not, the Karkada mapping
   for LLMs should be stated for the *output* distribution / unembedding side, not the residual
   stream of the concept token.
3. **Corpus at OLMo-mix scale.** Wikipedia's fifths partial was still rising at 3.1B words
   (0.27→0.63). Stream ~50B tokens of allenai/olmo-mix-1124 (DCLM + wiki) with the same scanner;
   check convergence and whether music-forum text (chord charts) adds chromatic (P1) structure.
4. **Remove the categorical confound at the source.** Use a 12-key set where every key has the
   same token count and no accidental token: e.g. solfège-free "key of do…" is not available in
   English; alternatives: (a) German note names (H, B, Fis, Es …) in a German model/corpus;
   (b) numbered keys in a synthetic corpus + tiny transformer trained from scratch with a
   controlled fifths-smooth kernel — the clean causal test of statistics→geometry with competing
   generators, no tokenization confound at all.
5. **Causal test on Gemma-2-2B layers 20–24**: project out / add the k=7 subspace and measure
   change in few-shot dominant accuracy (representation ≠ algorithm caveat applies).
6. **Instrument improvement**: report Fourier profiles *after* removing the categorical
   (indicator) directions as the default; add mode-isotropy to the standard output. A boxcar on a
   contiguous arc of a cycle always mimics the fundamental of that cycle's *other* generator.

## After Phase V (2026-08-29)
- The corpus→behaviour link is generic and directional (conditional operator, neutralized view, modulation family);
  a fingerprint test with real power needs (a) a DCLM sample ≥ 50× the current one, (b) a second checkpointed run
  (Pythia/OLMo-1 with published data order) for cumulative temporal alignment, (c) more than two OLMo models.
- The genre effect (web prose > encyclopedia per pair) is a lead: which extraction family/genre best predicts behaviour?
- Held-out rank scorers are insensitive here; use KL/R² (pre-register both from the start).
