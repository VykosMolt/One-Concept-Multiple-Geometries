# RESEARCH_LOG.md — chronological log

Convention: each entry is timestamped (local, 2026-08-28/29), states what was observed,
NOTE ON TIMESTAMPS: the "~HH:MM" stamps from about 01:00 onward were written from an internal estimate
that ran fast; the machine clock at the end of the session read 02:51 when the OLMo-2-7B pipeline started
(see results/olmo7b_pipeline.log), so entries labelled ~05:00–~10:00 actually happened between ~01:00
and ~02:50. The ORDER of entries is correct; treat the stamps as sequence markers, not clock times.
what was changed, why, and whether the change was made BEFORE or AFTER seeing the
relevant result. Nothing is deleted from this log; superseded entries are annotated.

---

## 2026-08-28 ~evening — session start

**Status: pre-registration of initial plan (before any data or model results).**

Initial question (from brief): does a Karkada-style co-occurrence Fourier spectrum for
the 12 major keys predict the Fourier spectrum of OLMo-2-1B hidden states along the
semitone axis, in a domain where chromatic (k=1/11) and fifths (k=5/7) generators compete?

Pre-registered primary statistic (before seeing any data):
- Corpus: M*_S (12x12) from Karkada Eq. 1 with L=16, f(d)=L+1-d, Wikipedia 20231101.en.
  Circulant projection kappa(d) = mean_i M*_{i,i+d}; lambda_k = DFT(kappa).
  Predicted representation mode energies: E_k ∝ |lambda_k| (Karkada B.2/B.3 — |M*|, relative
  amplitudes only; for LLM residuals the paper itself only claims qualitative carry-over).
- Model: OLMo-2-0425-1B base; hidden states at all layers; concept-axis DFT of the 12 centered
  vectors; paired energies P1..P5 plus E6; full spectrum reported, not just P1 vs P5.
- Positive control: months (Karkada template "The month of the year is x"), weekdays.
- Synthetic tests of instrumentation before any model data.

Pre-registered comparison: rank/Spearman + cosine between normalized corpus |lambda_k| spectrum
(k=1..6 paired) and normalized representation spectrum, per layer; with permutation-null
(random relabelings of the 12 concepts) for the representation spectrum concentration.

Pre-registered expectations (honest priors, written before results):
- Months: representation P1 dominant at mid layers; corpus kappa roughly exponential in
  circular distance so |lambda_1| > |lambda_2| > ... (Karkada Fig 5).
- Keys: genuinely uncertain. Key names co-occur in text mostly through (a) lists of keys
  (chromatic or fifths order both plausible), (b) related-key discussions (relative/parallel/
  dominant/subdominant → fifths-adjacent), (c) works catalogues (Op. numbers). Prior ~50/50
  that corpus kappa is fifths-peaked vs chromatic-peaked; and the enharmonic-spelling
  choice (Db vs C#) could dominate everything because "C#" strings co-occur with "#"-spelled
  keys (sharp side) and "b" with flat side — this is a *fifths-correlated* nuisance.

Environment: RTX 5070 Ti Laptop 12 GB, Python 3.12 venv, torch cu128, transformers.
Repo: ~/Documents/Research/pitch_fourier.


## 2026-08-28 late — instrumentation verified (before any corpus/model data)

`tests/test_synthetic.py` passes: Parseval, conjugate symmetry, chromatic/fifths fundamentals,
mixtures (P5 share = mixing weight exactly), isotropic-noise null = (2/11,…,1/11), SNR sweep,
non-circulant single-concept displacement, random-relabel null (95th pct of P1 share for a pure
circle after random relabel ≈ 0.40 — so a P1 or P5 share must exceed ~0.4 to be individually
significant at 12 points), circulant projection + kernel DFT + |M| prediction, and the
λ^(5)_k = λ^(g)_{7k} identity.

Discovery while verifying the math (logged as a design consequence, not a result): under x→7x
only P1↔P5 swap; P2, P3, P4, E6 are invariant. So the "competing generator" question is
entirely carried by the P1:P5 ratio, and the other four bins are ordering-independent
checks of the corpus→model prediction. See MATH.md §3.

Corpus pipeline decisions (before seeing counts):
- Key mentions matched case-sensitively as "<Letter><acc>? (major|minor)" with acc in
  {#, ♯, -sharp, ' sharp', b, ♭, -flat, ' flat'}; every spelling counted separately.
- Sentence-initial matches (preceded by .!?:"( or newline) are excluded for ALL keys uniformly,
  to handle "A major factor…" / "A minor issue…" polysemy without breaking translation symmetry.
  Excluded counts are recorded per key so the asymmetry can be inspected.
- Positions are whitespace-word indices; multiword mentions sit at their first word.
- Three tallies kept per shard: all docs; docs without explicit "circle/cycle of fifths",
  "perfect fifth(s)", "circle of fourths"; and only those docs.
- Also recorded: within-16-word raw pair counts by distance, doc-level co-occurrence, and the
  sequence of consecutive key mentions (≤40 words apart) for a "which generator do lists use"
  statistic.

## 2026-08-29 ~00:30 — corpus pipeline validated on 1/41 of Wikipedia (shard 0)

Observed (shard 0 only, 112M words): months M* is highly circulant (off-diagonal circulant
fraction 0.945) with |λ| paired profile [0.50, 0.27, 0.12, 0.08, 0.03, 0.005] — monotone in
harmonic order, as expected for an exponential-type kernel (Karkada Fig. 5). Weekdays: P1-dominated
(0.875) but circulant fraction only 0.21 (weekend/weekday asymmetry breaks translation symmetry —
expected). Keys: too sparse on one shard (F# major 12 mentions; many zero cells → M* = −2), so
the key spectra from this test run are NOT interpretable and are not used. Full 41-shard scan
pending download.

No changes to statistics or preprocessing were made after seeing these numbers.

## 2026-08-29 ~00:45 — saturation of M* noticed (on months/weekdays, shard 0; BEFORE any key result)

M* = 2(ρ−1)/(ρ+1) is bounded in [−2,2] and saturates for ρ ≫ 1. Months have ρ≈10–30 → M*≈1.0–1.8;
weekdays have M*≈1.94–1.98 everywhere (κ nearly flat, λ_{k≠0} ≈ 0.07 and below). Key names will
almost certainly be even more saturated (they co-occur only inside music articles, ρ ~ 10²–10³),
so the circulant kernel shape lives in tiny variations near 2, and any zero cell (−2) is a
gigantic outlier. Decision (made now, before seeing key spectra): the primary statistic remains
M* (faithful to Karkada, whose theory is about M* not PMI), and a *pre-specified secondary*
statistic is PMI = log ρ (unbounded; Karkada B.4 show M* ≈ PMI only near ρ≈1). Both will be
reported for every family; if they disagree qualitatively that is itself a finding about the
theory's sensitivity to the choice of target matrix. Implemented as `build_M(..., stat="pmi")`.
Progressive scanning of Wikipedia shards started as they download.

## 2026-08-29 ~01:10 — exploratory (not pre-registered): sequential interval statistics, 6/41 shards

`scripts/adjacency.py` on consecutive key mentions (≤40 words apart), interval = second − first mod 12,
excluding identical keys. major→major (n=3161): fifth(±5/7)=0.33, whole-tone(±2)=0.23,
minor-third(±3)=0.17, major-third(±4)=0.13, semitone(±1)=0.11, tritone=0.03. Uniform would be
0.18 per pair / 0.09 tritone. minor→minor identical pattern. major→minor peaks at +9 (relative
minor) and +7; minor→major peaks at +3 (relative major) and +10.
Interpretation: at the level of raw sequential text, semitone adjacency is *suppressed* and
fifth adjacency *enhanced*; but this is unigram-confounded (common keys C/G/D/F are mutually
fifth-related) — M* corrects for unigram rates, so this is only a hint. Recorded BEFORE any
M* key spectrum or any model result.

## 2026-08-29 ~01:40 — tokenization inspection (BEFORE extraction)

OLMo-2 tokenizer (100,278 vocab): months are single tokens. Keys: C, D, E, F, G, A, B, Db, Eb,
Ab are single tokens (" Db"/" Eb"/" Ab" exist as tokens, presumably from abbreviations); Bb → " B"+"b";
every sharp (C#, D#, F#, G#, A#) → letter + "#"; Gb → " G"+"b". "-flat" is one token, "-sharp" is
two ("-sh","arp"); unicode ♭/♯ are 2 byte-tokens each. So in the canonical major set, x=6 (F#)
and x=10 (Bb) are 2-token spans, the rest 1-token; in the canonical minor set x=1 (C#), 6 (F#),
8 (G#), 10 (Bb) are 2-token. A token-length artifact would appear as a non-circulant
perturbation at those positions (spreads energy over all modes, cf. synthetic test 7), not as
a specific k=5/7 signal; but it can dilute/inflate any bin. Controls: (i) 'anchor' position =
the identical " major"/" minor" token right after the concept; (ii) span-mean; (iii) an all-
flat set (Db,Eb,Ab single-token, Gb,Bb two-token) and all-sharp set (all 5 accidentals two-token);
(iv) the "-flat"/"-sharp" word spellings. Extraction positions recorded: last, mean, anchor,
final; layer 0 = embedding output. Model loaded in float32.

## 2026-08-29 ~02:05 — FIRST MODEL RESULT (major keys) and a confound identified immediately after

Observed (OLMo-2-1B, "The piece is written in the key of {x} major", anchor token " major"):
P5 share 0.22 (L1) → 0.33 (L9), z≈3.7 vs 2000-relabeling null; P1 ≈ 0.15, z≈−0.7 (at/below
null). Span-mean and last-token positions agree. Static embeddings: nothing (P5 0.20, z 1.2).
Bare "{x} major": weak (P5 ≈ 0.22, z ≤ 1.9). Months control: P1 share 0.33–0.38, z≈6, rank
order P1>P2>P3≈P4>P5>E6 matching the corpus |λ| rank order.

Confound noticed AFTER seeing this: the black keys {Db,Eb,F#,Ab,Bb} = fifths positions
{7,9,6,8,10} — contiguous on the circle of fifths (that is just the key-signature structure).
A binary "has accidental" feature is therefore a width-5 boxcar in fifths coordinates; its DFT
puts |B_1|²=13.9 vs ≈1 in other bins, i.e. ≈79% of its energy in P5 (semitone k=5/7) and ≈0.4% in
P1. So P5-dominance is exactly what a *categorical* black/white-key feature would produce — and
that feature is also partly a tokenization feature (sharps = 2 tokens; " Db"/" Eb"/" Ab" are
distinct single tokens). This does not make the result wrong (the corpus statistics may carry
the same block structure, and the theory would then predict it), but it means "P5 > P1" alone
cannot be read as "circle-of-fifths geometry".

Added diagnostics (decided now, applied to all families): (a) project the centered
black-key indicator out of H and recompute spectra; also project out the token-count indicator;
(b) mode isotropy: for hhat_k ∈ C^d, singular values of [Re; Im] ∈ R^{2×d}; a genuine
circle has s2/s1 ≈ 1, a boxcar/categorical feature has s2 ≈ 0; (c) white-key-only test: the 7
white keys are a *line* in fifths order F–C–G–D–A–E–B and a different line in chromatic order
C–D–E–F–G–A–B; compare RSA of their Gram with the two line-distance matrices (no accidental
confound possible within white keys).

## 2026-08-29 ~02:25 — black-key confound CONFIRMED by projection + synthetic calibration

Major keys, "…key of {x} major", anchor " major", layer 9: raw P5=0.33, P5 after projecting out
the black-key indicator = 0.13 (null 0.18; z=+0.5), isotropy s2/s1 of the k=7 mode = 0.57.
Synthetic calibration (same d=2048, isotropic noise, signal fraction tuned to reproduce raw
P5≈0.33): boxcar+noise → projected P5 = 0.12, iso7 = 0.50; genuine circle+noise → projected
P5 = 0.23–0.25, iso7 = 0.98; 50/50 mixture → 0.19, 0.76. The data sit on the pure-boxcar line.
Also: the projection removes only 13.5% of a *noise-free* circle's P5 energy, so it is not
"too aggressive" — a circle would have survived.

Conclusion so far: OLMo-2-1B's contextual key representation at this prompt carries a strong
categorical black-key/white-key (accidental / key-signature-complexity) feature and essentially
no smooth circle-of-fifths or chromatic circle. The apparent fifths-dominance is the boxcar.
This is Outcome G ("something weird") relative to the original framing, and it reframes the
corpus question: does the corpus M* also carry a black/white block structure (which the theory
would then correctly "predict" — for the wrong reason)?

## 2026-08-29 ~02:40 — confound diagnostics across all key families (anchor position)

Every 12-key family/template (minor canon ×4, all-sharp ×2, all-flat ×2, "-flat/-sharp" words,
major chords ×2) shows the same pattern: raw P5 elevated (0.27–0.55), P1 at/below null; after
projecting out the black-key indicator P5 falls to 0.05–0.19 (null 0.18), z_proj between −1.8 and
+1.5; k=7 isotropy 0.3–0.6 (categorical-like) vs ≈0.9 for k=1 (noise-like). The "-flat"/"-sharp"
word spelling gives the largest raw P5 (0.55) — because the accidental is a separate, highly
salient token — and the largest residual after projection (0.19–0.24, z≈1.5), consistent with
a three-level natural/flat/sharp categorical rather than a circle. White-key fifths-line RSA is
≈0 or negative everywhere; white-key chromatic-line RSA grows with depth to ≈+0.4 at the final
layers (identical across families, as it must be — same seven strings and template).
Interpretation: no template produces a smooth fifths circle; all produce an accidental
categorical feature, which is contiguous on the fifths circle and hence masquerades as P5.
Tonic template "The musical note {x}, which" gives degenerate anchor states (to be checked —
likely a tokenization merge or a massive-activation token); not used.

Tokenization artifact found (2026-08-29 ~02:45): in "The musical note {x}, which", "F#," tokenizes
as " F" + "#," so the anchor token differs for F# (the anchor becomes " which"). The extractor's
span/anchor detection is offset-based and handled this "correctly" but the anchor is then not
identical across concepts. Rule adopted going forward: anchors must be whitespace-separated
words (" major", " minor", " which" after a space), never punctuation glued to the concept.
Template tonic_canon__t2 is excluded from all analyses.

## 2026-08-29 ~03:05 — FULL WIKIPEDIA CORPUS RESULT (41 shards, 6.41M docs, 3.12B words)

(A first merge accidentally included a partial-merge file as a 42nd shard; fixed — merge.py now
only reads train-*.json — and everything below is from the corrected merge.)

Months: M* circulant fraction 0.968, |λ| profile [0.499, 0.256, 0.118, 0.081, 0.033, 0.012]
(PMI: [0.41, 0.25, 0.14, 0.11, 0.07, 0.03]). Bootstrap sd < 0.001. Matches Karkada's exponential-
kernel picture; and matches the *rank order* of OLMo's month spectrum (P1>P2>P3≈P4>P5>E6).
Weekdays: P1 share 0.90 but circulant fraction only 0.27 (weekend block).

Major keys (canonical spellings; 37.5k mentions, min per key 232 = F# major):
  M*  : profile [0.199, 0.061, 0.132, 0.151, 0.383, 0.075]  circ frac 0.33
  PMI : profile [0.172, 0.145, 0.141, 0.031, 0.505, 0.006]  circ frac 0.50
Minor keys (26.5k mentions): M* [0.20, 0.03, 0.11, 0.13, 0.46, 0.08]; PMI [0.19, 0.14, 0.11, 0.01, 0.55, 0.00].
Enharmonic-merged and mode-merged variants: same picture, P5 ≈ 0.34–0.55, P1 ≈ 0.16–0.20.
Bootstrap sd on P5 ≈ 0.01–0.05.
So at face value: corpus predicts fifths-dominance (P5 ≫ P1), and OLMo shows fifths-dominance.
Outcome "A" — *before* accounting for the black-key confound on both sides. Next: apply the
same black-key projection to M* (scripts/corpus_confounds.py).

## 2026-08-29 ~03:15 — corpus-side confound check → MISMATCH emerging; M* saturation is fatal for M*

scripts/corpus_confounds.py on full Wikipedia:
- M* for keys: κ(d) = 2.00 at every distance (1.99 at tritone); ρ = P_ij/(P_iP_j) ≈ e^7.5 ≈ 1800.
  Karkada's bounded statistic is saturated; its "spectrum" is numerical residue at the 10⁻³
  level (and it flips to P1>P5 after black-key projection, i.e. it is not stable). **M* cannot
  be used for this family.** PMI (pre-registered secondary) is used from here on; this is a
  post-hoc switch of primary statistic, forced by saturation, and is recorded as such.
- PMI kernel, fifths order d'=0..6: [8.64, 7.89, 7.59, 7.16, 7.07, 7.13, 6.48] — monotone
  decreasing in fifths distance (≈ exponential/linear decay + tritone dip); semitone order is
  jagged (d=1: 7.13 < d=2: 7.59 < d=5: 7.89). Same for minor keys and merged variants.
- After projecting the black-key indicator out of the PMI matrix (both sides): P5 stays dominant
  (major 0.30, minor 0.16→ hmm minor drops to 0.157 vs P1 0.255 — noted; merged-modes 0.29 vs 0.23;
  major merged 0.32 vs 0.21). White-key sub-block line RSA: fifths +0.63/+0.66/+0.67 vs chromatic
  +0.14/+0.16/+0.15 (all-mode merged +0.45 vs +0.26). Block means (PMI): black–black 7.7–8.4 >
  white–white 7.3–7.5 > white–black 7.0 — so a black/white block IS present in the corpus too,
  but it is not the whole story: the white-key line is fifths-ordered.
- Explicit-theory removal: the "nocof" tally (docs without "circle/cycle of fifths", "perfect
  fifth(s)", "circle of fourths") differs from "all" by 485 docs; profiles identical to 2 decimals.
  Explicit music-theory descriptions do NOT drive the corpus statistic.

Contrast with OLMo-2-1B (anchor " major", all templates): P5 collapses after black projection
(0.13 vs null 0.18), white-key fifths RSA ≈ 0 (chromatic +0.1→+0.4). Tentative verdict:
corpus (PMI) has genuine fifths-neighbour structure; the model's contextual representation at
this depth carries only the accidental categorical. This is Outcome D unless a stronger
model-side measurement (many contexts averaged; direct Gram-vs-PMI RSA; bigger models) finds
the fifths-neighbour structure. Next steps decided now: (1) multi-context averaged extraction
+ Gram RSA against corpus PMI (with/without black-key projection, and white-keys only);
(2) Gemma-2-2B (Karkada's LLM), Qwen2.5-3B, OLMo-2-7B (CPU offload) as additional models.

## 2026-08-29 ~03:50 — multi-context (24 contexts) + Gram-RSA decomposition: the mismatch is clean

scripts/multicontext.py / multictx_analyze.py / decompose_rsa.py. Model Gram (centered, 24-context
average) vs corpus matrices, Spearman over the 66 off-diagonal pairs, z vs 2000 relabelings.
Major keys, last token, layers 0–16:
  vs black/white block: +0.53 → +0.84 (z ≈ 6–6.7)   ← dominant
  vs commonness (log-unigram outer sum): +0.4 → +0.15 (z ≈ 2–3)
  vs corpus PMI: +0.3 (z ≈ 2.5); vs PMI circulant part +0.37; vs −fifths_dist +0.37 (z ≈ 3); vs −chrom_dist ≈ 0.
  Partial Spearman controlling for black block + commonness: fifths ≈ −0.05…+0.06 (nothing);
  chromatic +0.10…+0.25 (weak); PMI ≈ −0.1. The corpus PMI under the same partialling keeps
  fifths = +0.63. So every bit of model–corpus agreement is mediated by the block/commonness
  components, and the corpus's genuine fifths-neighbour kernel is absent from OLMo-2-1B.
(The earlier "+0.60 after black projection" for minor keys was Gram-RSA against the black-
projected PMI, which still contains commonness; see the decomposition for minor below.)

## 2026-08-29 ~04:05 — decomposition across positions; letter-identity confound added

minor/last, major/anchor, minor/anchor, major/mean (24-context averages): black block ρ = 0.5–0.86
everywhere (z 4–7); −fifths_dist raw ρ 0.2–0.47 but partial (controlling block + commonness)
≈ −0.13…+0.17 (minor/last layer 6 is the largest, +0.17, with PMI partial +0.30 — the only
hint of anything beyond the block, and it fades by layer 14). Chromatic partial is consistently
weakly positive (+0.1…+0.3). Corpus PMI partials: fifths +0.63 (major) / +0.48 (minor).

New confound noticed while reading these: in canonical spelling, semitone neighbours often share
the letter (Db–D, Eb–E, F–F#, Ab–A, Bb–B): 5 of 12 adjacent pairs. A letter-identity token
feature (" Db" vs " D" share a leading character; sharps literally contain the letter token)
therefore produces spurious k=1 (chromatic) structure — the mirror image of the black-key
confound for k=5/7. Added `same_letter` and alphabet-distance control targets to
decompose_rsa.py (decided AFTER seeing the weak chromatic residual; purpose is to test whether
it is letter identity).

## 2026-08-29 ~04:20 — letter controls; two new branches started

With same_letter + alphabet-distance added as controls, the model's chromatic partial drops to
≈0 (major/last: +0.14, +0.03, +0.00, +0.03, −0.07 across layers 2–16); the model's raw RSA with
same_letter is itself ≈0. Final reading for OLMo-2-1B: key-name geometry = black/white block
(+ commonness) only. Corpus PMI keeps fifths partial +0.62 under the same controls.

Branch 1 (behavioral, exploratory): does OLMo-2-1B know dominant / subdominant / relative /
parallel / transposition relations at all? (scripts/behavior_keys.py). If it does not, the
missing geometry is unsurprising: the pairwise statistic exists in the corpus but the model has
not internalized the relation.
Branch 2 (theory-faithful embedding prediction): second corpus pass over key-containing docs
building a V=3000 vocabulary (all key spellings + top words) with the full V×V Karkada-weighted
co-occurrence, to compute W = Φ√|Λ| from the full M*/PMI and read off the key rows — this is
the theory's actual prediction with helper words (Karkada Fig. 4), not the 12×12 proxy.

## 2026-08-29 ~04:30 — behavioral result: OLMo-2-1B does not know key relations

scripts/behavior_keys.py (greedy first-token pitch class): dominant 0/12, subdominant 1/12,
relative minor 1/12, relative major 1/12, semitone-up 3/12, whole-tone-up 0/12, fifth-up 0/12,
parallel minor 8/12 (this last one is just copying the tonic, which is the model's default
behaviour in every relation prompt). So OLMo-2-1B has no usable knowledge of the fifths
relations whose pairwise co-occurrence signature is in the corpus. Interpretation (mechanistic,
not a theory rescue): the 12×12 low-order statistic is present in the data, but a 1B model has
not compiled it into either representation geometry or behaviour; Karkada's carry-over to LLMs
may require the model to have actually learned the relation. Decisive test = larger models.

## 2026-08-29 ~04:45 — convergence curve + quantitative months control

scripts/convergence.py (cumulative shards 1,2,4,8,16,41; words 0.11B→3.12B; min key count 12→232):
- Months: P1 share 0.499 from the first shard on (converged at 0.1B words).
- Keys (PMI): RSA with −fifths_dist +0.25→+0.71; partial fifths (ctrl block+commonness)
  +0.27→+0.63; circulant fraction 0.05→0.50 (zero cells dominate small corpora). Still rising
  at 3.1B words → Wikipedia alone has not converged; a larger corpus (OLMo mix) would be the
  natural next step but the direction is monotone.
- Keys (M*): P5 share 0.33→0.25→0.14→0.07→0.07→0.38 — non-monotone garbage (saturation).
Months positive control, corpus |λ| profile vs OLMo-2-1B profile: cosine 0.94 (embeddings),
0.97 (Karkada template, final token, L16), 0.99 ("It happened in x, which", L15); Spearman
0.94–1.00. A flat null profile would give cosine 0.73. So the instrument + theory reproduce the
known month result quantitatively at the profile level.

## 2026-08-29 ~05:00 — theory-faithful prediction with helper words (scripts/theory_embedding.py)

Second pass over the 14,235 key-containing Wikipedia docs (23.5M words), V=3000 (42 key spellings
+ 2958 most frequent words), full V×V Karkada-weighted counts; PMI (and M*) → eigendecomposition →
W = Φ√|Λ| (Karkada Eq. 30/32, |M| factorization). Key rows (canonical major/minor):
- d=3000 (exact |M|): profile ≈ flat (P5 0.21, P1 0.18; the diagonal self-PMI dominates),
  Gram RSA −fifths_dist +0.44, −chrom_dist −0.08, white-key fifths-line +0.50 / chrom-line +0.03.
- d=300 (top-|λ| modes): P5 0.29, P1 0.15; black-projected P5 0.17 (P1/P5 ≈ 1.06);
  RSA −fifths +0.52, white fifths-line +0.45. Minor keys similar.
- Ablating the 12×12 key–key block before factorizing changes nothing (identical to 3 decimals):
  the key geometry is carried entirely by the 2958 helper words — Karkada's Fig.-4 phenomenon
  reproduces in this domain, and it means the 12×12 block is only a proxy.
- M* instead of PMI gives the same RSA numbers (+0.43) — at the *full-matrix* level M*'s
  saturation matters less because helper-word cells are not saturated.
Reading: the theory's own embedding prediction for keys is "black/white block + a moderate but
real fifths ordering (RSA ≈ +0.5, white-key line +0.45)". OLMo-2-1B reproduces the block and
not the fifths ordering.
d-sweep of the theory embedding (major keys, PMI): partial fifths (ctrl block+commonness+letter+
alphabet) = +0.55 (d=100), +0.52 (300), +0.56 (1000), +0.56 (3000); black_block RSA +0.59→+0.48.
Robust. OLMo-2-1B: partial fifths ≈ 0 at every layer/position.

## 2026-08-29 ~05:30 — Gemma-2-2B (Karkada's LLM; bf16; 24 contexts)

Tokenization: Db/Eb/Ab/Bb single tokens, F# = " F"+"#" (only one 2-token key in the canonical major set).
Fourier: raw P5 0.26–0.35 at all layers, P1 0.14–0.19; after black projection P5 → 0.09–0.17 (null
0.18); k=7 isotropy 0.36–0.75. Same categorical-boxcar signature as OLMo-2-1B.
RSA decomposition (major keys): black block ρ 0.63–0.82 (z 5–6.6) at all layers; commonness
+0.13–0.34. Partial fifths (ctrl block+commonness+letter+alphabet): ≈0 through layer 16, then
+0.18/+0.19/+0.24 (mean position, layers 20/22/24) and +0.18/+0.22/+0.25 (last token), back to
−0.11/−0.13 at layer 26 (final). White-key fifths-line RSA at layers 22–24 (mean): +0.55/+0.61.
Minor keys: partial fifths ≤ +0.18 (layer 24), otherwise ≈0.1.
Reading: Gemma-2-2B has a weak, late, transient fifths-neighbour component (≈ a third of the
corpus's +0.62) on top of the same dominant categorical block; the final layer loses it (cf.
Singh & Chopra 2026's "transient" perceptual geometry). OLMo-2-1B has none. Capacity/training
appear to matter; OLMo-2-7B and Qwen2.5-3B are the next test.
Zero-shot behaviour: Gemma predicts "the"/"a" (wants to continue "the key of…"), so the
zero-shot prompt is unfair for base models; replaced by a few-shot, full-completion-scored
test (scripts/behavior_fewshot.py) for all models. Decided after seeing Gemma's outputs.

## 2026-08-29 ~05:40 — new branch (exploratory): predictive key-transition matrices

Idea: Karkada's theory is about co-occurrence; an LLM's most direct analogue of "co-occurrence" is its
*prediction* P(next key | current key), not the geometry of the key-name token. scripts/predictive_matrix.py
scores all 12 " {y} major" completions after six relation-neutral contexts ("modulates from {x} major
to", "…second movement is in", "A key closely related to {x} major is", …), builds the 12×12 log-prob
matrix, and compares it (RSA + partials, same controls as before) with corpus PMI, fifths/chromatic
distance, block and commonness; also its Fourier profile. This tests whether the corpus statistic is
absorbed *predictively* even when it is absent *representationally*. Written before seeing any output.

## 2026-08-29 ~05:55 — few-shot relation knowledge (scripts/behavior_fewshot.py; 3 demos, disjoint keys, 12-way completion scoring)

OLMo-2-1B: dominant 0.08, subdominant 0.17, relative minor 0.08, relative major 0.08, semitone 0.17,
whole tone 0.00, fifth-up 0.08, minor third 0.00, tritone 0.08 — chance (1/12=0.083) throughout.
Gemma-2-2B: dominant 0.33, subdominant 0.33, fifth-up 0.33, tritone 0.25, semitone 0.25, relative
major 0.17, relative minor 0.08, whole tone 0.08, minor third 0.00. Errors on the fifth relations
are mostly other fifth-related keys (C→F, G→C, Db→Bb…). So Gemma has partial fifths knowledge,
OLMo-2-1B essentially none — consistent with the representational partial-fifths ordering
(Gemma late layers +0.25, OLMo 0). Behaviour and geometry co-vary across the two models.

## 2026-08-29 ~06:05 — dtype fidelity check

OLMo-2-1B multi-context analysis rerun in bf16 vs fp32: Gram-RSA max |Δ| = 0.002 (corr 0.9997–0.9999
across layers), P5 share max |Δ| < 0.0005. bf16 (used for Gemma-2-2B, Qwen2.5-3B, OLMo-2-7B) does
not confound the representation study.

## 2026-08-29 ~06:40 — process note (not scientific)
Three background "waiters" used `pgrep -f <pattern>` where the pattern also occurred in the waiter's
own command line, so they waited forever; the predictive-matrix run was delayed ~40 min. Fixed with
anchored patterns. No results were affected.

## 2026-08-29 ~06:50 — predictive key-transition matrices (scripts/predictive_matrix.py; results/predictive/)

L[x,y] = log P(" y major" | context(x)), normalized over the 12 candidates, symmetrized; 6 contexts.
OLMo-2-1B: RSA with corpus PMI +0.25…+0.58 (best "modulates from x major to" +0.58, "…second movement
is in" +0.58); with −fifths_dist +0.21…+0.42; with block +0.5…+0.7; with commonness +0.55…+0.77.
Partial fifths (ctrl block+commonness+letter+alphabet): +0.20 (modulates), +0.24 (then_key), +0.13
(next_song), ≈0 for "related key"/"chord after". Partial PMI +0.43/+0.46 for the two best contexts.
Top-1 (non-self) answer intervals for "modulates to": +5 ×4, +2 ×3, +7 ×2 — fifth-related 6/12.
Gemma-2-2B: RSA PMI +0.37…+0.56; partial fifths +0.23 (modulates), +0.23 (next_song), +0.16, +0.13,
+0.13, +0.10; partial PMI up to +0.36.
Reading: the corpus's smooth fifths component is partially present in what the models *predict*
(≈ +0.2, a third of the corpus's +0.62), even for OLMo-2-1B whose key-token *geometry* has none.
The Karkada-style statistic → LLM mapping is therefore better stated for the output distribution
than for the concept-token residual stream, at these scales. Both models' predictions are, like
their geometry, dominated by commonness and the black/white block.
Caveat: the predictive matrix is unigram-confounded by construction (P(y) enters every row); the
partial correlation controls for commonness, but the control is the *corpus* unigram, not the
model's marginal. Fourier profile of the symmetrized L: P5 ≈ 0.47–0.50 in every context (like the
corpus PMI's 0.505) — but that number carries the block, so it is not evidence of a circle.

## 2026-08-29 ~07:15 — CIRCLE vs LINE: corpus is periodic, models are open (scripts/circle_vs_line.py)

Motivation (post-hoc, after the accidental confound): "number of sharps/flats in the key signature" is a
*line* along the circle of fifths (signed position C=0,G=+1,…,F#=+6; F=−1,Bb=−2,…,Db=−5). Circle
distance and line distance differ only on the 15 of 66 pairs whose short arc crosses the F#/Db seam
(e.g. B–Db: circle 2, line 10). This is exactly Karkada's periodic-BC vs open-BC distinction.
Partial Spearman, each distance controlling for the other (+ black block + corpus commonness):
  corpus PMI major:      circle|line +0.40   line|circle +0.27      (minor: +0.38 / +0.01)
  theory embedding d=300: circle|line +0.57   line|circle −0.31
  OLMo-2-1B predictive ("modulates to"):  circle|line −0.39   line|circle +0.69  (then_key −0.40/+0.76)
  Gemma-2-2B predictive:                  circle|line −0.38   line|circle +0.74  (then_key −0.35/+0.65)
  with the model's own marginal added as a control: OLMo −0.37/+0.71, Gemma −0.34/+0.73 (unchanged).
  Seam pairs only: ρ(circle) = +0.31 (corpus), +0.26 (theory), −0.42/−0.27 (models' predictions).
  Representations (Gram, last token): early layers weakly line-like (line|circle ≈ +0.2, circle|line
  ≈ −0.2), ≈0 later — consistent with "block + commonness only".
PC1 of the centered matrices: corpus PMI |ρ(PC1, cos θ_fifths)| = 0.83, |ρ(PC1, signed pos)| = 0.31;
theory embedding 0.88 / 0.17; models' predictive matrices 0.16–0.31 / 0.73–0.78.
Reading: the corpus (and the theory's embedding) has a genuine *circle* of fifths — enharmonic
equivalence is honoured (B major is near Db major); the LLMs' predictions instead follow the
*line of fifths* (Temperley): sharp keys and flat keys are opposite ends of an open 1-D manifold
ordered by signed accidental count, and B major is far from Db major. This is a qualitative
topology mismatch (periodic vs open boundary) between the pairwise statistic and the model, on
top of the magnitude mismatch. Plausible mechanism: text is spelled, and spelling ("sharps"/
"flats", "#"/"b" tokens) is what a small LM can read off; enharmonic identity requires music
knowledge the models lack (few-shot relation tests). Wikipedia co-occurrence honours it because
articles discuss related keys across the seam (e.g. Db major ~ C# minor relative pairs, F#/Gb
enharmonic mentions). To be checked on Qwen2.5-3B and OLMo-2-7B (pipelines running).

## 2026-08-29 ~07:40 — is the models' "line" merely a spelling class? No.

Extra controls: flat-name block (names ending in "b": Db,Eb,Ab,Bb) and accidental-sign block
(sharps | C | flats). Models' predictive matrices: line|circle stays +0.45…+0.58 with both blocks
controlled; circle|line −0.21…−0.31. Corpus PMI: circle|line +0.37 with both blocks controlled,
line|circle +0.26. So the models' structure is a *graded* line of signed accidental count (F#
farther from Bb than Db is from … etc.), not just "has a b in the name"; and the corpus's circle
is not just the absence of that class. Figure figures/summary/fig5_circle_vs_line_matrices.png:
in fifths order, corpus = banded diagonal with bright corners (B≈Db, F≈C); models = two blocks
(sharp side vs flat side) with dark corners.

## 2026-08-29 ~07:45 — Qwen2.5-3B (bf16, 36 layers): representation
Same as OLMo-2-1B/Gemma-2-2B: raw P5 0.26–0.33 at all layers, P1 0.17–0.20; after black projection
P5 0.08–0.12 (null 0.18); k=7 isotropy 0.43–0.60; Gram-RSA with corpus PMI +0.35…+0.43 (z≈3) but
black-projected +0.08…+0.22 (z −0.9…+0.9); white-key fifths-line RSA ≤ +0.31 and non-monotone.
Decomposition / few-shot / predictive pending (pipeline running).

## 2026-08-29 ~07:55 — Qwen2.5-3B decomposition (major, last token); repo hygiene
Partial fifths (ctrl block+commonness+letter+alphabet): +0.08 (L0), +0.14 (L4), +0.05 (L8), +0.15 (L12),
+0.23 (L16), +0.13 (L20), +0.13 (L24), +0.14 (L28), +0.18 (L32), −0.16 (L36 final). Black block ρ
0.66–0.83 (z 5–6.6). So Qwen2.5-3B sits between OLMo-2-1B (≈0) and Gemma-2-2B (+0.25): a small
mid-layer fifths component, lost at the final layer, on top of the same categorical block.
Repo: the first commits had accidentally tracked data/ and models/ (28 GB .git); re-initialized with
a .gitignore. No files were lost; the commit history (checkpoints only) was discarded.

## Chronology summary: what was fixed before vs after seeing data (for the reader)

Pre-registered (before any corpus/model result): M* definition, L=16 window and weights, Wikipedia
20231101.en, canonical key spellings and orderings, concept-axis DFT + paired energies, relabeling
null, months/weekdays controls, synthetic tests, OLMo-2-1B base fp32, four token positions,
comparison via normalized profiles (cosine/Spearman). PMI declared as secondary statistic after
seeing months/weekdays saturation but BEFORE any key spectrum.
Exploratory / post-hoc (each dated in the log above): (1) sequential-interval statistics; (2) the
black-key contiguous-arc confound and its diagnostics (indicator projection, mode isotropy,
white-key line RSA) — found after the first P5 result; (3) switching the corpus primary statistic
to PMI because M* is saturated for keys — after seeing κ(d)=2.00; (4) multi-context averaging and
Gram-RSA with decomposition into block/commonness/fifths/chromatic — after the mismatch emerged;
(5) letter-identity controls — after seeing a weak chromatic residual; (6) theory-faithful
full-vocabulary embedding with helper words; (7) behavioural relation tests, zero-shot then
few-shot (the few-shot redesign was made after seeing Gemma answer "the"); (8) predictive
key-transition matrices; (9) circle-vs-line (periodic vs open BC) test — after the accidental
confound suggested that "number of accidentals" is a line; (10) spelling-class controls for the
line finding. Confirmatory follow-ups: Gemma-2-2B, Qwen2.5-3B (and OLMo-2-7B if finished) were
run with the complete pipeline fixed in advance (scripts/run_model_pipeline.sh) after the OLMo-2-1B
analyses were designed; no per-model tuning.
Nothing was deleted; failed/artifact-laden runs (tonic template with comma anchor; M* key spectra;
the 42-shard merge) are kept in results/ and described above.

## 2026-08-29 ~08:30 — Qwen2.5-3B: behaviour, predictive matrices, circle-vs-line — and a revision of the explanation

Few-shot relation accuracy (chance 0.083): dominant 0.75, subdominant 0.83, fifth-up 0.83, whole-tone
0.67, semitone 0.50, relative minor 0.50, relative major 0.33, minor third 0.25, tritone 0.25. Qwen
knows the fifths relations well (its few errors cluster at the F#/Db seam, e.g. dominant of F# → C).
Predictive matrices: RSA with corpus PMI +0.44…+0.59; partial fifths +0.33 (modulates), +0.42
(chord after), +0.31 (next song), +0.28 (then key); partial PMI +0.41. Circle vs line: line|circle
+0.68…+0.82 (with model-marginal control), circle|line −0.23…−0.41; seam pairs ρ(circle) = −0.81/−0.70;
PC1 monotone in signed accidental count (|ρ| 0.76–0.80); survives flat-name and sign-block controls
(line|circle,+both = +0.55 / +0.34).
Representation (from the earlier entry): partial fifths best +0.23 (L14–16), final layer −0.16;
black block ρ up to 0.83.
REVISION: the earlier mechanistic reading ("small models have not learned the relation, hence no
geometry") is falsified by Qwen2.5-3B — it has the relation knowledge (behaviour and predictions)
and still no fifths geometry in the key-name token beyond a small mid-layer component. The
dissociation is therefore between *where* the pairwise statistic is expressed: output-side
predictions (partial fifths +0.3–0.4, with an open-line topology) vs. concept-token residual
geometry (categorical block + commonness, ≤ +0.25 transient). Karkada's LLM carry-over, read as
"concept-token Gram ≈ corpus block", fails for this family across three model families at 1–3B.
Scoring caveat: the 12-way completion uses canonical spellings only, so answers whose correct
spelling is the enharmonic alternative (Gb for F#'s subdominant, C# for F#'s dominant) are scored
against Db/F#; this can only *under*-count accuracy for seam keys.

## 2026-08-29 ~09:10 — predicting-position geometry (NEXT_STEPS 1b, started; written before results)

scripts/predict_position.py: residual at the final token of "The piece modulates from {x} major to"
(and three other next-key contexts), all layers, 12 keys. Question: does the fifths/line structure that
appears in the *logits* also exist as residual-stream geometry at the position that predicts the next
key — i.e. is the mismatch "concept token vs predicting position" rather than "representation vs
output"? Same RSA/partial/circle-line diagnostics as before. Run on OLMo-2-1B, Gemma-2-2B, Qwen2.5-3B;
added to the 7B pipeline.

## 2026-08-29 ~09:35 — PREDICTING-POSITION RESULT: the corpus statistic lives in the state that predicts a key

Residual at the final token of a next-key context, partial fifths (ctrl block+commonness+letter+
alphabet), best layer / final layer, with circle|line and line|circle at those layers:
  OLMo-2-1B  "then_key": +0.46 @L12 (circle|line +0.29, line|circle +0.06) → final L16 +0.34 (−0.20 / +0.60)
             "next_song": +0.37 @L13 → +0.32 final;  "modulates_to": +0.17 @L13 → +0.14 (line from L14 on)
  Gemma-2-2B "then_key": +0.38 @L15 (circle|line +0.25, line|circle −0.02) → final L26 +0.28 (−0.15 / +0.45)
             "next_song": +0.29 @L5; "modulates_to": +0.26 @L26 (line)
  Qwen2.5-3B "modulates_to": +0.47 @L27 (circle|line +0.30, line|circle +0.04) → final L36 +0.28 (−0.31 / +0.64)
             "then_key": +0.29 @L32 (line); "next_song": +0.24 @L33 (line)
Compare: key-name token geometry best ≤ +0.25 and ≈ −0.15 at the final layer; corpus PMI +0.62.
Black-block RSA at the predicting position is also high (0.7–0.86), and k=7 isotropy stays low
(0.25–0.5), so the block is still the dominant single component; but a genuine, controlled fifths
component of +0.3…+0.47 — half to three-quarters of the corpus value — now appears, in all three
models, in the residual stream at the position whose *output* is a key.
Layer dynamics (consistent across models): at the best mid/late layer the fifths component is
circle-like (circle|line ≈ +0.25…+0.30, seam not a barrier); by the final layer it has become the
line (line|circle +0.45…+0.72), matching the logit-level line found earlier. I.e. the model first
computes a circle-of-fifths-like relation and then re-expresses it in spelled-key coordinates for
output. Figure: figures/summary/fig6_predicting_position.png.
Interpretation for the original question: Karkada's "statistics → representation geometry" does
carry over to these transformers, but to the geometry of the state that *predicts* the concept
(the output-side representation), not to the geometry of the concept token itself; and the
periodic topology of the corpus is preserved internally but broken at the output by spelling.
This reframes Outcome D as: *representation of the concept ≠ representation for predicting the
concept; the pairwise statistic predicts the latter.*

## 2026-08-29 ~10:00 — directed corpus statistic (exploratory, negative)

scripts/directed_corpus.py builds D[x,y] = log P(y within K words after x)/P(y) from the ordered-pair
distance tallies. K=4 is too sparse (1,288 ordered pairs); K=16 (13,332 pairs) gives RSA with −circle
+0.64 / −line +0.66, partials circle|line +0.31, line|circle +0.29 (no preference), and correlates with
the models' predictive matrices at +0.29…+0.43 raw but only ≤ +0.15 after block+commonness controls —
i.e. the directed statistic explains the models' predictions no better than the symmetric PMI. Mean D
by directed interval peaks at +5 (fourth up, 8.15) and +7 (fifth up, 8.05), lowest at +6 (tritone) and
+11. Not pursued further.

## 2026-08-29 03:15 (machine clock) — OLMo-2-7B (bf16, CPU offload; pipeline 02:51–03:12)

Token geometry (major, last token, 24 contexts), partial fifths by layer: −0.01 (L0), −0.02 (L4),
+0.14 (L8), +0.22 (L12), +0.23 (L14), +0.20 (L16–20), +0.30 (L24), +0.34 (L26), +0.34 (L30), +0.09 (L32,
final). White-key fifths-line RSA reaches +0.66 (L28). BUT: P5 after black projection stays 0.11–0.14
(null 0.18) and k=7 isotropy 0.5–0.65 → the fifths ordering enters the Gram as a *line-like ordering*,
not as an isotropic Fourier circle: circle|line ≈ 0 (−0.10…+0.03), line|circle +0.2…+0.3 at every
layer; seam-pair ρ(circle) ≈ 0. Black block still the dominant component (ρ 0.79–0.86, z 6.5–6.9).
Few-shot: dominant 0.92, relative minor 0.83, subdominant 0.75, relative major 0.75, semitone 0.75,
fifth-up 0.58, whole-tone 0.50, tritone 0.50, minor third 0.25.
Predictive matrices: partial fifths +0.33 (modulates), +0.37 (then_key), +0.34 (related), +0.50 (chord
after); partial PMI up to +0.52; line ≫ circle (line|circle +0.74…+0.80, circle|line −0.3; seam ρ −0.28/
−0.62; survives spelling-class controls: +0.41…+0.51).
Predicting position: best partial fifths +0.53 @L18 ("modulates to"; circle|line +0.26, line|circle
−0.06 — circle-like), +0.47 @L22 (then_key), +0.41 @L24 (related); final layer +0.38/+0.36/+0.31 with
line|circle +0.29…+0.42. Same circle-mid → line-out pattern as the smaller models, stronger.
Scale curve (best-layer partial fifths): key-name token 1B 0.19 / 2B 0.25 / 3B 0.23 / 7B 0.34;
predicting state 1B 0.46 / 2B 0.38 / 3B 0.47 / 7B 0.53; corpus 0.62; theory embedding 0.52.
Verdict unchanged in kind, sharpened: with scale a line-like fifths ordering does enter the key-name
geometry late in the network (still below the predicting state and the corpus, still lost at the
final layer, never an isotropic circle), while the predicting state and the predictions carry it
at all scales.

## 2026-08-29 ~03:20–04:30 (machine clock) — INDEPENDENT REVIEW (two Opus reviewers) and corrections

Two independent reviewers (code/instrument verification; adversarial scientific review) were run on the
repo. Both reports are preserved verbatim in notes/review_code.md and notes/review_science.md. Summary of
what they found and what was done:

VERIFIED (reproduced by the code reviewer, most to 3 decimals): all synthetic tests and the fourier.py
math (DFT convention, Parseval, mode permutation for all four units, circulant projection, |M| prediction,
project_out, isotropy, nulls); corpus counting vs an independent brute-force implementation of Karkada's
P_ij (exact); merge sums; sentence-initial exclusion; every number in the corpus tables; the partial
Spearman implementation (independent QR version agrees to machine precision); the 15 seam pairs and signed
positions; extraction span/anchor/BOS logic; the layer-9 anchor result (raw P5 0.332, projected 0.133,
iso 0.57) re-extracted on CPU; the boxcar/circle calibration from scratch; all per-model and predicting-
position table entries; theory-embedding W=Φ√|Λ| reading of Karkada Eq. 30/32.

ERRORS FOUND AND FIXED (numbers/prose):
- "projected P5 collapses BELOW the null": wrong — the relabeling null for the projected share is 0.120,
  so 0.13 is AT the matched null (z ≈ +0.4). Code always used the matched null; prose fixed.
- "projection removes only 13.5% of a circle's P5 energy": wrong — it removes 47.9% of the energy;
  13.5% is the share drop. Fixed in MATH.md §10.
- κ in fifths order is not strictly monotone (7.07 at d'=4, 7.13 at d'=5). M* saturation is 2.00±0.03
  (not ±0.01); ρ median 1558 / mean 1683 (not ≈1800). Gemma/Qwen black-block maxima are +0.84 (not
  +0.82/+0.83); Qwen best partial-fifths layer is L14 (+0.233), not L16; flat-null cosine for months is
  0.748 (paired null), not 0.73; "cosine 0.97–0.99" is best-layer (per-layer minima 0.82–0.94); the
  spelling-control line|circle range is +0.34…+0.58 (Qwen then_key +0.34 was omitted).
- Latent bug: partial() returned 0.042 instead of NaN on a constant input (layer-0 rows of the
  predicting-position tables). NaN guard added to all scripts. cof_docs counter under-reported in the
  prefilter branch (counter only; no statistic affected); fixed.
- Bootstrap SDs (Poisson on f-weighted counts, no document clustering) understate uncertainty ≥3×;
  circ_frac_offdiag (0.05→0.50 across corpus sizes) is not converged. Noted in RESULTS.
- results/multictx/*_decompose_*.txt lack the "targets' mutual RSA" rows (grep filter); cosmetic.
- nwords is ≈0.7% high (whitespace approximation in unmatched docs) — a uniform shift of every PMI
  entry (λ₀ only); "3.12B words" → "≈3.1B".
- The corpus P5=0.505 / P1=0.172 Fourier profile depends on the PMI diagonal κ(0): P5>P1 is robust to
  κ(0)±2 but removing the diagonal inverts it; the model profiles in predictive_matrix.py drop the
  diagonal, so the two were not on the same footing. The ordering-informative statement is "the deviation
  of κ from its mean peaks at k=5/7" and the Gram/partial analyses, not the 0.50 number.

SUBSTANTIVE FINDINGS (each verified or re-run by me after the review):
1. Enharmonic spelling starves the seam pairs in the canonical corpus matrix (Db|F# weighted count 17 vs
   Db|Gb 160; Wikipedia is spelling-consistent within a document). With enharmonics merged
   (major_merged@pmi): circle|line +0.58 ± 0.02, line|circle −0.07 ± 0.03, P(circle > line) = 1.000
   (Poisson bootstrap), seam ρ(circle) +0.62, partial fifths +0.66. Canonical: +0.40/+0.27, P = 0.87.
   The corpus circle claim now leads with the merged family (scripts/corpus_seam.py).
2. Partial correlations had no nulls and best-of-layer values had no selection correction. Added
   scripts/partial_nulls.py (free null, block-preserving null, max-over-layers null, exact 5040-ordering
   null for the white-key line). Token-geometry partial fifths, max-over-layers p (free/block):
   1B +0.21 (0.16/0.21), Gemma +0.29 (0.10/0.09), Qwen +0.23 (0.15/0.19), 7B +0.34 (0.045/0.017).
   White-key fifths-line, exact-null p at best layer / max-over-layers p: 1B +0.41 (0.037/0.19),
   Gemma +0.61 (0.003/0.028), Qwen +0.42 (0.025/0.24), 7B +0.78 (0.000/0.001).
   → the "scale curve" is three noise-level points and one significant point (7B); Gemma's late
   "transient fifths component" is not significant after layer selection (its white-key line is).
   Corpus and theory embedding: p = 0.0005 under both nulls (reviewer's computation).
3. Corpus white-key fifths-line RSA +0.63 ranks 78th of 5040 orderings (p = 0.0155; merged: 48th,
   p = 0.0095); the corpus PC1 diagnostic is confounded with commonness (|ρ| 0.85) and is dropped.
4. Predictive partial fifths are conservative: adding the model's own column marginal as a control
   raises every value (1B then_key +0.24→+0.36; Qwen chord_after +0.42→+0.58; 7B +0.50→+0.61).
5. The few-shot scorer ranked completions by unnormalized log-prob over unequal token lengths, under-
   selecting 2-token key names 1.6–7×; accuracies were lower bounds and the "errors at the seam"
   reading was confounded. Re-run with calibrated scoring (log P(cont|prompt) − log P(cont|"The key
   is")): scripts/behavior_fewshot_calibrated.py (results below).
6. (Reviewer 2, replicated by me on all four models — results below) The "black/white-key block" is
   ORTHOGRAPHIC: respelling four white keys as B#, Fb, E#, Cb moves the block from "black keys" to
   "names with an accidental glyph". scripts/decouple_orthography.py.
7. (Reviewer 2, re-run by me on all four models — below) ≈¼–⅓ of the models' "line of fifths" in the
   predictive matrices is canonical-spelling scoring (F#'s dominant must be spelled C#/Db); merged-
   target scoring reduces line|circle but it stays large and significant.
8. (Reviewer 2, re-run by me — below) The predicting-position vs concept-token comparison was
   confounded with context (24 generic contexts vs 1 next-key context); with matched contexts the gap
   shrinks to ≈0.1, a non-predicting control context also scores high, and the "circle-like
   mid-network" stage is significant in 1 of 12 model×context cells (7B "modulates to").
CITATIONS ADDED (LITERATURE_AUDIT.md): Chew's spiral array; Krumhansl–Kessler torus; Levy & Goldberg;
Nanda et al. / Clock-and-Pizza; Marjieh et al.; Gurnee & Tegmark. Novelty narrowed accordingly.

## 2026-08-29 08:30 (machine clock) — review controls re-run on all four models (my own scripts)

Orthographic decoupling (scripts/decouple_orthography.py; 12 contexts; C→B#, E→Fb, F→E#, B→Cb): canonical set —
black-key block RSA +0.7…+0.86 (identical to the glyph block by construction); decoupled set — black-key block
RSA −0.14…+0.12 at every layer in all four models, accidental-glyph block +0.49/+0.51 (1B, L12–16), +0.34/+0.43
(Gemma, L24/26), +0.36…+0.67 (Qwen, L18–34), +0.26…+0.28 (7B, L16–30). The dominant categorical feature in
every model's key-name geometry is "the name carries an accidental glyph", not key-signature membership.
(Caveat kept: B#/Fb/E#/Cb are rare strings, so glyph and rarity remain entangled.) All prior mentions of a
"black/white-key block" should be read as "accidental-glyph block"; the contiguous-arc argument is unchanged
because that set is the same five keys under canonical spelling.

Enharmonic-merged predictive scoring (predictive_matrix.py, logsumexp over both spellings of the five black
keys): line|circle canonical → merged, modulates_to: 1B +0.69→+0.46, Gemma +0.74→+0.52, Qwen +0.79→+0.66,
7B +0.74→+0.56; then_key: +0.76→+0.63, +0.65→+0.54, +0.68→+0.60, +0.75→+0.56. circle|line stays ≤ 0 in these
contexts. So ≈¼–⅓ of the "line" was canonical-spelling scoring; the rest is real. New nuance: in the
chord-progression context ("…the tonic chord is usually followed by the chord of") the merged residual is
circle-like for the larger models (Qwen circle|line +0.35 / line|circle +0.18; 7B +0.44 / +0.14; 7B
related_key +0.22 / +0.09), i.e. enharmonic identity is honoured for chord relations but not for key
modulation/succession contexts. Recorded; not pursued further tonight.

## 2026-08-29 ~08:50 (machine clock) — matched-position nulls and three-scorer few-shot; RESULTS.md rewritten

Matched contexts (concept token vs predicting token from the same sentence; max-over-layers nulls, results/nulls_predpos.txt):
1B then_key +0.33 (p .058) vs +0.46 (p .018); Gemma then_key +0.42 (.031) vs +0.38 (.039); Qwen modulates +0.37 (.039)
vs +0.47 (.020); 7B modulates +0.46 (.015) vs +0.53 (.010), then_key +0.52 (.007) vs +0.47 (.020). Non-predicting control
(final token of "…modulates from x major very often in the"): −0.01 (1B), +0.16 (Gemma), +0.14 (Qwen), +0.37 (7B).
Conclusion: the earlier "locus" claim was mostly a context effect; predicting position adds ≈0.1 in 1B/3B only.
"Circle-like mid-network": 1 of 12 cells significant (7B modulates_to). RESULTS.md rewritten accordingly.
Few-shot scorers: mean-per-token ≈ raw (7B dominant 0.83, subdominant 0.83; Qwen 0.67/0.67; Gemma 0.33; 1B ≈ chance);
multi-token targets are no longer under-selected under mean scoring; the single-prior calibrated scorer over-corrects
(rare spellings win everything) and is reported only as a failed variant.
