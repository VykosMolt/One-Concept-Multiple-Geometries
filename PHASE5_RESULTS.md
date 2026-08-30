# PHASE5_RESULTS.md — training-data fingerprint test

**VERDICT: CONDITIONAL STATISTICS PREDICT BEHAVIOUR — training-data fingerprint NOT ESTABLISHED (specificity test
inconclusive). HARDENING (08-30): the contrast is conditional-row vs association construction, NOT directionality — a
symmetrized conditional from the same counts does equally well; held-out gain survives a training-row target prior
(larger), document-cluster bootstrap and a 5,000-relabeling null with the (b+1)/(B+1) estimator (the earlier 'p<.001'
came from a 300-permutation null and was not justified). See MANUSCRIPT_AUDIT.md.** Local key-mention conditionals predict held-out next-key behaviour beyond circle/line/orthography/
frequency in all four models, from every corpus tried (Wikipedia, OLMo-Mix wiki, three DCLM samples). Whether OLMo
models are better predicted by the data they consumed than Gemma/Qwen are: over all 72 pre-registered DiD cells (9 per
corpus × 4 corpora × 2 views) 10 reach p < .05 and all 10 are OLMo-favouring, but each is small (0.004–0.05 nats;
six separate both OLMo models from both others, four are 7B-driven), and the modulation family shows DiD ≈ 0 on every
DCLM sample in the neutral view. (Correction
2026-08-29: the first version of this file reported only 4 of the 9 cells per corpus — harmonic/modulation × window/
document — a post-hoc restriction found by the paper's adversarial review; see §4.) Along the OLMo-2-1B checkpoint
run the coarse line coordinate appears by 21B tokens and the residual correspondence with corpus conditionals emerges later
(first significant at 294B, stable from 4T) — partial TEMPORAL ACQUISITION evidence from one run, one family.

Central number (the one asked for): in the neutralized view, for the modulation family, the corpus's directional
conditional statistics reduce the held-out KL of model next-key behaviour by 14–34 % beyond a theory model with circle
distance, line distance, signed difference, accidental, source/target frequency and token count (OLMo-2-1B +0.037 nats
of 0.134; Gemma-2-2B +0.029 of 0.127; Qwen2.5-3B +0.016 of 0.117; OLMo-2-7B +0.036 of 0.130; relabeling p ≤ 8e-4 at
B = 5000, b = 0 in six of eight cells),
and explain 19–38 % of the within-row residual variance (ΔR², all p ≤ .007). In the spelled view the same number is
small (theory KL already 0.05–0.07 because the rows are line-dominated); ΔR² +0.10 (1B) / +0.22 (7B) / ≈ +0.02
(Gemma, Qwen). The rank-based ΔCV pre-registered alongside KL was ≈ 0 in the same cells — a metric insensitivity
(11–14 targets per held-out row), reported as such. Whether this residual conditional structure is *OLMo-specific*
(§4) and *temporally acquired* (§5) is answered below.

## 1. Design (PHASE5_DESIGN.md; pre-registered)
Objects: model behavioural matrices Q (15 spelled sources × 15 spelled targets; Phase-II families C harmonic, D chord,
E modulation, 4 templates each; total log-prob scorer) and corpus directional conditional matrices C under extraction
families A (directional 40-word window), B (cue-conditioned window), C (relational regexes; 317 pairs, too sparse),
D (document-conditioned: j anywhere later in the document than i). Nuisance/theory features per ordered pair: circle
distance, line distance, signed difference, glyph-class equality, has-accidental (source, target), same root letter,
edit distance, log source/target frequency, token counts. Tests: (1) residual Spearman after ridge-removing the features
from log C and log Q, joint key-relabeling null (2000); (2) leave-one-source-row-out prediction of Q with theory-only /
corpus-only / theory+corpus predictors — scored by within-row Spearman (ΔCV), by softmax-KL (pre-registered, implemented
only at the v3 re-run; declared in the log) and by within-row R² gain (post-hoc); relabeling null (300) and Poisson
bootstrap (100); (3) enharmonic-twin difference vectors, raw and line-controlled (residualized on target position,
line-distance difference, glyph class), target-permutation null. Views: spelled (15 targets) and neutralized (target
columns merged to 12 pitch classes, class-level features), neither privileged. Three claim levels: WEAK (both line-like),
MEDIUM (corpus rows predict held-out model rows beyond theory), STRONG (residual, idiosyncratic structure predicts
residual behaviour, with corpus specificity and/or checkpoint-time alignment).

## 2. Operators side by side (results/phase5/operators.txt; notes/phase5_ntp_vs_karkada.md)
Karkada's word2vec theory factorizes a symmetric PMI-like association; Zhao et al.'s NTP theory factorizes the
directional context → next-token conditional (its dominant max-margin component only the support pattern, which is
degenerate — all-ones — for a 15-key vocabulary at Wikipedia scale, so smoothed conditionals are the usable proxy).
On Wikipedia over the 15 spellings: symmetric PMI puts twins closest (ECI 0.04; controlled circle|line +0.42,
line|circle +0.39); the directional conditional rows keep twins apart (ECI 0.57; controlled circle|line +0.55,
line|circle +0.12); document co-occurrence as a similarity carries neither (≤ +0.17). What each predicts (Spearman over
105 pairs): behaviour (E family) — conditional +0.80/+0.80/+0.79 (7B/1B/Qwen), helper factorization +0.65/+0.68/+0.64,
symmetric PMI +0.54/+0.39/+0.40; 7B prompt-final geometry — conditional +0.61, helper +0.61, PMI +0.46; key-name
span-mean geometry — nothing above +0.15 (Qwen +0.37 for the helper factorization). So the operator that the NTP theory
factorizes is the best predictor of behaviour and of predicting-state geometry, the symmetric operator the worst, and
the key-name geometry is predicted by none: statistic-dependent geometry at the descriptive level. Note that this is a
comparison of operators computed from one corpus against models trained on other corpora; it does not test either
theory's factorization claim.

## 3. Central test on Wikipedia (results/phase5/fingerprint_wikipedia_v3[_neutral].txt; figures/phase5/heldout_gain_wikipedia.png, fingerprint_wikipedia.png)
**Residual correspondence (not held-out).** Spelled view: OLMo-7B +0.24…+0.47 (p ≤ .02 in 7/9 cells; E×D_doc +0.47,
p .001); OLMo-1B +0.28 (E×A_win40, p .03); Qwen +0.28 (C×A, p .05); Gemma ≈ 0. Neutralized view: E_modulation × D_doc
+0.60 (1B), +0.59 (Gemma), +0.50 (Qwen), +0.60 (7B), all p ≤ .002; × A_win40 +0.47/+0.47/+0.38/+0.47. With a rich
nuisance model (quadratics, cosine harmonics of circle distance, source-row dummies) +0.53/+0.52/+0.39/+0.53 (p ≤ .01);
C_harmonic residuals mostly fall to n.s. under the rich model.
**Held-out prediction (the central number).** Neutralized view, E_modulation: ΔKL +0.037/+0.029/+0.016/+0.036 (A_win40)
and +0.045/+0.036/+0.023/+0.039 (D_doc) nats per row against a theory KL of 0.12–0.13 (all p < .001); ΔR²
+0.29/+0.30/+0.19/+0.33 (A) and +0.36/+0.38/+0.24/+0.38 (D). C_harmonic: ΔR² +0.14/+0.03/+0.15/+0.15 (A_win40;
p .003/.27/.013/< .001). D_chord: OLMo-7B only (ΔKL +0.06–0.07 of 0.36, p ≤ .003) plus Qwen × A_win40 (ΔR² +0.14, p .003).
Rank ΔCV: significant only for OLMo-7B C_harmonic (+0.087/+0.075) and E (+0.047) and Gemma E (+0.02). Spelled view:
E × A_win40 ΔR² +0.10 (1B, p .02), +0.22 (7B, p < .001), ≈ +0.02 (Gemma, Qwen, n.s.); E × D_doc +0.14/+0.12/+0.05/+0.30
(p .01/.03/.19/< .001); ΔKL ≤ +0.012 nats. **Rich-theory robustness (post-hoc; results/phase5/fingerprint_wikipedia_v3_neutral_rich.txt, _v3_rich.txt).** Adding
quadratics and cosine harmonics of circle distance to the theory features: neutralized E_modulation ΔKL +0.023/+0.019/
+0.007/+0.018 (A_win40) and +0.034/+0.026/+0.013/+0.022 (D_doc) nats against a theory KL of 0.11–0.12 (all p ≤ .017;
6–28 % reduction), ΔR² +0.17/+0.18/+0.06/+0.18 (A) and +0.26/+0.29/+0.11/+0.24 (D); C_harmonic gains fall to n.s. in
every model. Spelled view: E × D_doc ΔR² +0.10 (1B, p .03), +0.20 (Gemma, p .01), +0.20 (7B, p .003), Qwen n.s.
So the modulation-family gain is not an artefact of a too-simple theory model; the harmonic-family gain was.
**Twin difference vectors.** Raw ΔC·ΔQ cosines are large everywhere (Cb|B +0.65…+0.92) but that is the line; after
line control, significant alignment survives only in scattered cells (OLMo-1B C×A Cb|B +0.74 p .01; Qwen C×B Gb|F#
+0.89 p < .001; 7B E×A Cb|B +0.71 and Db|C# +0.72, p .02), not consistently across extraction or context families.
**Claim level reached on Wikipedia.** WEAK: yes. MEDIUM: yes in the neutralized view for the modulation family in all
four models under the KL and R² scorers; no under the rank scorer; in the spelled view only for OLMo-1B/7B. STRONG
(residual structure): the residual correspondence is present in all four models, so on Wikipedia alone it is not
evidence of a *training-data* fingerprint — Wikipedia-derived conditional structure beyond circle/line/orthography/
frequency is shared by models trained on four different corpora, i.e. it is a property of how English text about keys
is written, not of one model's data. Specificity is tested in §4.

## 4. Cross-corpus specificity (results/phase5/crosscorpus_compare_neutral.txt, _spelled.txt; figures/phase5/crosscorpus_neutral.png)
Corpora actually consumed by OLMo-2: OLMo-Mix-1124 wiki (both shards, 6.5 GB; 6,857 key documents — the Dolma wiki,
provenance en_simple_wiki_v0, with documents 3× shorter than the 20231101 dump and 5.5× fewer window pairs: A_win40
8,540, D_doc 6,152), an OLMo-Mix DCLM sample (9 shards, 1.75 GB zstd, ≈ 0.02 % of DCLM: 386 key docs, 633 / 467 pairs),
a Dolmino DCLM sample (2 shards: 611 docs, 1,320 / 845 pairs), Dolmino FLAN (49 docs, 6 pairs — unusable). Each corpus
is compared with Wikipedia binomially thinned to the same pair mass per extraction family (phase5/thin_wikipedia.py),
and specificity is the difference-in-differences of per-row held-out ΔKL: (OLMo models − Gemma/Qwen) × (corpus −
size-matched Wikipedia), Wilcoxon over the 15 source rows.
**Generic gain.** Every corpus adds held-out value for every model in the neutralized view — even the 633-pair DCLM
sample (E_modulation ΔKL +0.037/+0.029/+0.030/+0.047 for 1B/Gemma/Qwen/7B, p ≤ .023; C_harmonic +0.018/+0.023/+0.034/
+0.071, p ≤ .007) — and the real sparse corpora beat size-matched Wikipedia for all four models (OLMo-Mix DCLM
E × D_doc +0.024/+0.025/+0.025/+0.030 nats above thinned Wikipedia; Dolmino C × A_win40 +0.041/+0.043/+0.019/+0.019):
web prose about keys is a better predictor per pair than encyclopedia prose, for OLMo and non-OLMo models alike.
**Specificity (DiD), ALL nine cells per corpus (3 context families × A_win40/B_any/D_doc; full tables in
results/phase5/crosscorpus_compare_*.txt).** Significant (p < .05) cells, all OLMo-favouring: neutral — OLMo-Mix wiki
C×A +0.007 (.01), C×B +0.015 (.00), E×D +0.008 (.04); Dolmino D_chord×D_doc +0.049 (.00); 54-shard DCLM D_chord×D_doc
+0.023 (.01). Spelled — OLMo-Mix wiki C×A +0.012, C×B +0.012, E×D +0.004; 9-shard DCLM E×A +0.005 (.01); Dolmino
D_chord×A +0.017 (.02). That is 10 of 72 cells (both views), all positive; 34 of 72 are negative; no multiplicity correction. Per-model
decomposition: six of the ten separate both OLMo models from both non-OLMo models (OM-wiki C×B and E×D neutral, Dolmino
D×D neutral, spelled OM-wiki E×D, DCLM9 E×A, Dolmino D×A); four are carried by the 7B alone; Gemma outgains an OLMo
model in two. (Correction after the second paper review: an earlier version said 63 cells / 26 negative / 'each cell
carried by one model with Gemma gaining as much' — wrong on all three counts.) The modulation family — the family with the robust gain —
shows DiD −0.014…+0.010 (p ≥ .09) on every DCLM sample in the neutral view. Reading: INCONCLUSIVE, not null. The first
version of this section reported only the 4 harmonic/modulation × window/document cells (post-hoc restriction, now
withdrawn).
**Larger DCLM sample (post-hoc expansion, 54 shards, 2,884 key docs, A_win40 3,843 / D_doc 3,740 pairs;
results/phase5/crosscorpus_compare_big_neutral.txt, _big_spelled.txt).** Neutral E_modulation ΔR² +0.18/+0.13/+0.15/+0.19
(window, p ≤ .007), ΔKL +0.021/+0.010/+0.017/+0.024 (n.s. except 7B; theory KL 0.17–0.21 because DCLM frequency features
are poor). Against size-matched Wikipedia the picture is mixed (document conditional above by +0.008…+0.011, window
below for three models by 0.007–0.016 and marginally above for Qwen, +0.002), so the 'genre effect' seen at 633 pairs does not hold uniformly. DiD (OLMo − others): −0.004
(p .98), −0.002 (.45), −0.000 (.76), −0.000 (.85) neutral; +0.005, −0.008, −0.001, −0.000 spelled (p ≥ .33). The null
on OLMo's own 95 % corpus holds at 6× the sample size.
Caveats: even the larger DCLM sample is ≈ 0.1 % of DCLM; the DiD has 15 paired rows of power;
the frequency features are recomputed per corpus, so theory KLs differ across corpora (the size-matched baseline shares
that handicap).

## 5. OLMo-2-1B checkpoints (results/phase5/ckpt_trajectory.txt, ckpt_twins.txt; figures/phase5/checkpoint_trajectory.png)
Nine revisions (stage-1 1B, 21B, 49B, 105B, 294B, 1007B, 1993B, 4001B tokens; stage-2 ingredient-3 endpoint, 51B) plus
the released model, scored with the Phase-II behaviour battery (C, D, E families, fp32). E_modulation:
- line|circle (controlled partial): −0.24 (1B) → +0.19 (21B) → +0.27 (49–294B) → +0.34 (1T) → +0.33 (2T, 4T) → +0.44
  (stage-2 endpoint) → +0.47 (released). circle|line ≈ 0 throughout. The open-line coordinate is present by 21B tokens.
- neutral residual correspondence with Wikipedia D_doc (rich nuisance): +0.15 → +0.14 → +0.12 → +0.27 (105B, p .08)
  → +0.40 (294B, p .03) → +0.26 (1T, p .24) → +0.34 (2T, p .05) → +0.44 (4T, p .004) → +0.46 (S2, p .01) → +0.53
  (released, p < .001); with A_win40 ≈ 0 until stage 2 (+0.27, p .06) and the released model (+0.41, p .002).
- twin asymmetry (|log q(·|Cb) − log q(·|B)| etc.) 4.3 nats at 1B tokens (rare spellings near-unlearned) → 2.1–3.4 →
  2.6; row entropy 1.96 → 2.2. Twin difference *directions* stabilize early (cosine with the released model ≥ 0.86 from
  49B tokens; Cb|B ≥ 0.9 from 21B); their line-controlled alignment with Wikipedia ΔC is scattered (Gb|F# × D_doc
  significant only from 4T on; Db|C# × A_win40 at 49B/294B/4T/S2 but not 105B/1T/2T; Cb|B never).
Reading: the coarse geometry (line) is acquired first and early; the residual, residual correspondence emerges
later and noisily, is first significant at 294B, dips at 1T, and is stable from 4T through stage 2 to the released
model — consistent with TEMPORAL ACQUISITION of the fine structure, but with one non-monotone point and p-values that
straddle .05 through most of stage 1. Caveats: one training run, one family with a clean signal (C_harmonic residuals
are weak and non-monotone); the released 'main' weights belong to a different run (§6) and are plotted as a separate
point. Stage-2 replicates (§6) show the trajectory endpoint is stable across seeds (residual 0.45–0.48).
Temporal alignment with cumulative data order is not reconstructible from published artifacts (no data-order indices for
the 1B; official OLMo-core scripts exist only for the 32B); the only temporal intervention available is Stage 1 → 2.

## 6. Stage-2 as intervention (results/phase5/ckpt_trajectory.txt; three ingredient endpoints as replicates)
Stage 1 is OLMo-Mix-1124 (3.9T tokens; 94.9 % DCLM-Baseline, 0.09 % Wikipedia); Stage 2 is the Dolmino 50B mix
(47 % DCLM high-quality, 21 % math, 17 % FLAN, 7 % Wikipedia, 6 % pes2o, 2.5 % StackExchange) — a ~75× increase in the
Wikipedia share. All three stage-2 ingredient runs (same stage-1 endpoint, 51B tokens each, different seeds) were
scored. E_modulation, stage-1 endpoint (4T) → ingredients 1/2/3: line|circle +0.33 → +0.42/+0.42/+0.44 (3/3 up,
+0.09…+0.11); circle|line −0.06 → −0.03/+0.02/−0.01; Wikipedia neutral residual (rich) D_doc +0.44 → +0.48/+0.45/+0.46
(p .008/.018/.010), A_win40 +0.26 → +0.28/+0.27/+0.27; twin asymmetry 2.63 → 2.55/2.65/2.62; C_harmonic line +0.28 →
+0.30/+0.33/+0.38. So the Stage-2 mix reliably strengthens the open-line coordinate of modulation behaviour and leaves
the residual correspondence essentially unchanged (+0.01…+0.04, within replicate spread ≈ 0.03) — the intervention that
multiplied the Wikipedia share by 75 did not measurably raise Wikipedia-specific alignment. Twin-difference alignment is
likewise replicated across ingredients (E Gb|F# × D_doc +0.86/+0.77/+0.90, p ≤ .01; other cells n.s.).
**The released model is a different run.** The released `main` weights are orthogonal to every branch checkpoint
(cosine ≈ 0.00 for q_proj/down_proj/lm_head, 0.006 for the embedding) with ≈ 2.2× larger singular values, while the
three ingredient endpoints are cosine 0.96–1.0 to the stage-1 endpoint and 0.10 apart from each other; configs are
identical. It is not a soup of the ingredients (results of phase5/soup_check.py). The released model's values (line
+0.47, residual +0.53/+0.41) are therefore a second, independent OLMo-2-1B model, not the end of this trajectory.

## 7. Counter-evidence and limitations
- The residual correspondence is shared by four models with different training corpora (§3) and is obtained from every
  corpus tried (§4): it is a property of English text about keys, not of one model's data. The one OLMo-favouring
  signal (OLMo-Mix wiki, 2/12 cells, ≈ 0.01 nats, 7B-driven) is small, unreplicated in DCLM and not multiplicity-robust.
- Sparse DCLM samples beat size-matched Wikipedia for all models — a genre effect (discursive web prose vs encyclopedia
  lists) that the design did not anticipate and that a fingerprint reading would have to explain away.
- The rank scorer ΔCV pre-registered as "the central number" is ≈ 0 in most cells; the KL scorer that shows the gain was
  pre-registered but implemented after the first results were seen. Both are reported.
- Twin-difference alignment (the sharpest spelled-text signature) is not consistent across families or checkpoints.
- Extraction family C (regex patterns) was too sparse to use (317 pairs); B_modulation/C give NaN corpus-only fits.
- Checkpoints: one run, 9 points, two families; the released model's provenance relative to the ingredient endpoint is
  unresolved; the 1T dip is unexplained.
- Wikipedia conditionals come from 9,168 key-mentioning documents; Cb/Gb rows rest on 38/162 source mentions.

## 8. Relation to the papers
Karkada et al. explain Fourier geometry from a symmetric PMI-like operator; here that operator is the worst predictor
of behaviour and predicts no key-name geometry. Zhao et al.'s NTP theory names the object that does predict behaviour
(the directional conditional), but the support-pattern component that carries their geometric result is degenerate at
this vocabulary size; the phase's positive result is about smoothed conditionals, not about their max-margin geometry.
Neither theory is tested as a factorization claim here.

## 9. Files
phase5/scan_conditional.py, fingerprint.py (--neutral, --rich), operators.py, ckpt_fingerprint.py, ckpt_twins.py,
curl_ckpts.sh, curl_revs.sh, ckpt_behaviour.sh, fig_*.py, summary_table.py; results/phase5/*.txt, fingerprint/*.json,
cond_*.npz; figures/phase5/*.png; PHASE5_DESIGN.md, PHASE5_LOG.md, notes/phase5_ntp_vs_karkada.md.
