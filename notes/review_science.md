# Independent adversarial scientific review (Opus reviewer 2, 2026-08-29) — verbatim summary of findings

A. Statistics: collinearity not a problem (VIF <= 1.42; circle/line VIF 2.4–2.6). Missing nulls on all partials.
Built free and block-preserving relabeling nulls: corpus PMI +0.623 p 0.0005 (both); theory embedding +0.525 p 0.0005;
token geometry 1B +0.19 p 0.10, Gemma +0.25 p 0.03–0.06, Qwen +0.23 p 0.03–0.07, 7B +0.34 p 0.01–0.02. With max-over-
layers correction: 1B p 0.13/0.25, Gemma 0.075/0.11, Qwen 0.13/0.22, 7B 0.030/0.020. => scale curve mostly noise;
Gemma's "transient component" not significant. Verdict OVERCLAIMED (model side), SUPPORTED (corpus).

B. Contiguous-arc confound: 79.6% verified exactly. "13.5% of energy" wrong (47.9% of energy; 13.5% of share). Null for
projected P5 share is 0.120 not 0.18 => "collapses below the null" is false; code used the right null. 50/50 calibration
replicates (proj P5 0.180, iso 0.78). Caveat: iso7 (0.44–0.65) reads as w_circle 0.15–0.30, discordant with projected-P5
reading; iso7 not calibrated for anisotropic residual noise. SUPPORTED on physics, OVERCLAIMED on null comparison.

C. Circle vs line: models' line|circle jackknife-stable (+0.33…+0.84), p < 0.0005; PC1 is a signed axis (0.70–0.78) not
commonness. Corpus PC1 diagnostic confounded (log-frequency 0.85 > cos 0.83) — drop it. Seam pairs have ~2 orders of
magnitude fewer counts (Db–F# = 17); Poisson bootstrap P(circle > line) = 0.95 for the canonical family; merged family
circle|line +0.58, line|circle −0.07 — lead with it. NEW: enharmonic-merged target scoring reduces the models' line by
~1/4–1/3 (1B modulates line|circle +0.69 -> +0.46; Qwen +0.79 -> +0.66), still p < 0.005. SUPPORTED (models = line, after
merged scoring); OVERCLAIMED as a clean dichotomy.

D. Predicting position: (i) final-layer predicting Gram correlates 0.66–0.80 with the predictive log-prob matrix (partly
trivial); (ii) UNFAIR comparison — with matched contexts the concept token scores +0.33/+0.37/+0.46 vs predicting
+0.46/+0.47/+0.53 (gap 0.07–0.13, not 0.2); non-predicting control contexts score +0.31–0.37; (iii) "circle-like
mid-network": 1 of 12 model×context cells significant (7B modulates_to +0.47 @L17, p 0.005). OVERCLAIMED.

E. Corpus side: PMI legitimate (Levy & Goldberg 2014); block handling correct; nocof gives +0.643 / circle|line +0.44;
white-key fifths-line ranks 78th of 5040 orderings (p 0.0155). SUPPORTED.

F. Tokenization: spelling variants cannot decouple black-key from accidental-glyph. Decoupling test (C->B#, E->Fb, F->E#,
B->Cb; OLMo-1B and Qwen, 12 contexts): black-block RSA +0.85/+0.86 -> −0.03/+0.06 (1B), +0.84/+0.71 -> +0.05/−0.03
(Qwen); accidental-glyph block ≈0 -> +0.21…+0.43. The block is ORTHOGRAPHIC, not musical (caveat: B#/Fb/E#/Cb are rare
strings). UNSUPPORTED as "black/white-key block".

G. Prior art gaps: Chew (spiral array), Krumhansl–Kessler (key torus), Levy & Goldberg, Nanda et al. / Clock-and-Pizza,
Marjieh et al. (LLM pitch judgments), Gurnee & Tegmark. Instrument-caution novelty OVERCLAIMED; domain novelty defensible.

H. Verdict sentence overclaims (a) black/white-key block, (b) locus, (c) circle-like mid-network; proposed replacement
emphasising: corpus circle (p < 0.001 both nulls), theory embedding reproduces, models carry a small layer-selection-fragile
fraction in key-name geometry (significant only at 7B), orthographic accidental-glyph block puts 79.6% into P5, models'
predictions and next-key residuals form an open line of fifths surviving merged-spelling scoring at ~2/3 strength.

I. Inconsistencies: "below the null"; "13.5% energy"; flat-null cosine; verdict/table number mismatches; "targets'
mutual RSA" rows missing from saved decompose files; corpus row labelled "predictive" in a table; no partial has a null.
Recommended cheap runs: merged-target predictive on Gemma/7B; decoupling on Gemma/7B; white-key line with exact null and
layer correction (Gemma +0.61 p 0.020, 7B +0.78 p 0.003 survive); partial nulls in scripts; seam-count table.
Top-3 solid: corpus result; contiguous-arc mechanism + calibration; "models' key space is an open line ordered by signed
accidental count".
