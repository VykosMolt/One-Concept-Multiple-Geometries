# PHASE5_LOG.md — chronological (machine clock)

Sat Aug 29 02:13:14 PM CEST 2026 — Phase V start. Design in PHASE5_DESIGN.md.
OLMo-2-0425-1B has 195 stage-1 and 72 stage-2 checkpoint branches on HF; selected stage1 tokens 1B, 21B, 49B, 105B,
294B, 1007B, 1993B, 4001B and stage2-ingredient3 final (51B). Downloads started (≈ 3.4 GB each).
OLMo-Mix-1124 wiki = 2 files (6.48 GB); Dolmino-Mix-1124 wiki lists the same two files (same sizes) — identity to be
checked by LFS oid; if identical to the Nov-2023 dump, Wikipedia is already OLMo's wiki data and corpus specificity
must be tested on DCLM / Dolmino non-wiki samples.

14:15 Literature: Zhao et al. 2408.15417 and Zhao & Thrampoulidis 2505.08348 read (HTML extracts); comparison with
Karkada in notes/phase5_ntp_vs_karkada.md. Key distinction: symmetric PMI (word–word, magnitudes) vs directional
context→next-token support/conditional (NTP-UFM; dominant component depends only on the support pattern). For a
15-key vocabulary the support pattern is degenerate at Wikipedia scale, so the directional *conditional* (smoothed
log C) is used as the NTP-type operator; this is stated as an approximation.
OLMo-Mix-1124 wiki and Dolmino-Mix-1124 wiki are byte-identical (same LFS oids): Stage 2 reuses the Stage-1 wiki files.

14:18 Wikipedia extraction families (phase5/scan_conditional.py; 9,168 key docs): ordered pairs A_win40 47,247;
B_any 10,775 (modulation 722, chord 2,038, signature 8,552, enharmonic 145, relation 252); C_patterns 317; D_doc 19,162.
§12 operators side by side (phase5/operators.py; results/phase5/operators.txt): symmetric PMI — ECI 0.04, controlled
circle|line +0.42 / line|circle +0.39; helper-word factorization (V=3000) over the 15 spellings — ECI 0.74, raw line RSA
+0.43 / circle +0.03 (the 15-spelling factorization is line-like, unlike its 12-key version); directional conditional
rows (A_win40, JS between rows) — raw line +0.56 / circle +0.25, controlled circle +0.55 / line +0.12, ECI 0.57.
What predicts what (Spearman over 105 pairs): behaviour (family E) is predicted by COND A_win40 at +0.80 (7B), +0.80
(1B), +0.79 (Qwen); by HELP at +0.65/+0.68/+0.64; by SYM at +0.54/+0.39/+0.40. Prompt-final 7B geometry: COND +0.61,
HELP +0.61, SYM +0.46. Key-name span-mean geometry: nothing above +0.15 (HELP +0.37 for Qwen). D_doc predicts nothing.
=> the directional conditional operator is the best predictor of behaviour and of the predicting-state geometry; the
symmetric operator is the worst; the key-name geometry is predicted by no corpus operator. Whether COND's advantage
survives the line/spelling backbone is the residual test (running).

14:22 First fingerprint pass (OLMo-1B, Wikipedia, families C/D): theory-only LOO Spearman +0.86; corpus-only
+0.5–0.7; theory+corpus +0.85–0.87 → ΔCV −0.02…+0.003, permutation p ≥ 0.28: full corpus rows add nothing beyond the
theory/nuisance features (MEDIUM claim fails so far). Residual r +0.04…+0.22, p 0.09–0.46. Twin-difference cosines
large (Cb|B +0.65…+0.85, p ≤ .01) but their target-permutation null does not remove the line's own prediction of the
asymmetry direction; a line-controlled version (residualizing ΔQ and ΔC on s_j, |s_j−s_a|−|s_j−s_b|, glyph class of j)
was added before reading the other models. Rerun queued.

14:39 WIKIPEDIA FINGERPRINT (results/phase5/fingerprint_wikipedia_v2.txt, _neutral.txt; 4 models × 3 families × extraction families)
Spelled view (15×15): theory-only LOO Spearman 0.78–0.95; corpus-only 0.5–0.75; ΔCV(both − theory) between −0.026
and +0.011, permutation p ≥ 0.10 everywhere → full corpus rows add nothing beyond circle/line/spelling/frequency
features. Residual r (both residualized on the features): OLMo-7B +0.24…+0.47, p ≤ .02 in 7/9 cells (E×D_doc +0.47,
p .001); Qwen +0.28 (C×A, p .048); OLMo-1B +0.28 (E×A, p .03); Gemma ≈ 0. Line-controlled twin-difference cosines are
significant in scattered cells (OLMo-1B C×A Cb|B +0.74 p .01; Qwen C×B Gb|F# +0.89 p .00; 7B E×A Cb|B +0.71, Db|C# +0.72,
p .02) but not consistently across extraction families or context families.
Neutralized view (15×12, target aliases merged): residual r larger and broad — E_modulation×D_doc +0.60 (1B), +0.59
(Gemma), +0.50 (Qwen), +0.60 (7B), all p ≤ .002; ×A_win40 +0.47/+0.47/+0.38/+0.47 (p ≤ .011). ΔCV: OLMo-7B C_harmonic
+0.087 (A, p < .001), +0.075 (D_doc, p < .001), +0.042 (B_any, p .007); E +0.047 (A, p < .001), +0.028 (B, p < .001);
Gemma E +0.020/+0.022 (p .007); all other cells ≈ 0 or negative.
Reading so far: the spelled rows are dominated by the line/spelling backbone (COARSE GEOMETRY at the MEDIUM level);
the residual, idiosyncratic corpus structure does correspond to residual model behaviour in the neutralized view for
all four models and adds held-out predictive value for OLMo-7B (and weakly Gemma) — a partial STRONG-level result
pending corpus-specificity and checkpoint tests.

14:40 Robustness (results/phase5/neutral_robustness.txt): with a rich nuisance model (base features + quadratics
+ cos(2πk·circle/12) harmonics k=1..3 + 15 source-row dummies) the neutral-view residual correspondence survives for
E_modulation in all four models: A_win40 +0.41/+0.40/+0.26/+0.40 (p .000/.002/.055/.004), D_doc +0.53/+0.52/+0.39/+0.53
(p ≤ .01) for OLMo-1B/Gemma/Qwen/7B; C_harmonic cells drop to +0.05…+0.31 (mostly n.s.). The residual signal is
specific to next-key (modulation) behaviour, the closest task analogue of the corpus statistic.

17:50 Infrastructure: the huggingface_hub checkpoint downloads died twice with Xet/CAS backend errors ("File
reconstruction error: CAS Client Error"); the remaining three checkpoints (1993B, 4001B, Stage-2 51B) are being fetched
with resumable curl against the classic resolve/ endpoint with per-file size verification (phase5/curl_ckpts.sh),
~8 MB/s. Six checkpoints (1B–1007B) were downloaded and scored before the failure; downstream chains unchanged.

18:35 CHECKPOINT TRAJECTORY complete (results/phase5/ckpt_trajectory.txt; OLMo-2-0425-1B, families E/C, Wikipedia
conditionals, neutral-view residual r with the rich nuisance model, 500-relabeling p):
E_modulation line|circle: −0.24 (1B tok) → +0.19 (21B) → +0.27 (49B, 105B, 294B) → +0.34 (1T) → +0.33 (2T, 4T) → +0.44
(Stage-2 ingredient-3 end) → +0.47 (released soup). circle|line ≈ 0 throughout (−0.25 at 49B is the only excursion).
Neutral residual r vs D_doc: +0.15 → +0.14 → +0.12 → +0.27 (105B, p .08) → +0.40 (294B, p .03) → +0.26 (1T, p .24)
→ +0.34 (2T, p .05) → +0.44 (4T, p .004) → +0.46 (S2, p .01) → +0.53 (final, p < .001); vs A_win40: ≈0 → +0.27 (S2,
p .06) → +0.41 (final, p .002). Twin asymmetry 4.3 → 2.1–3.4 → 2.6; row entropy 1.96 → 2.2.
Reading: the open coordinate is present by 21B tokens and grows in two steps (early stage 1; stage 2/soup); the residual
corpus alignment emerges later and noisily (first p < .05 at 294B, dip at 1T, stable from 4T onward, strongest in the
released soup). The released 'main' model is a soup of three stage-2 ingredients, hence ≠ the ingredient-3 endpoint.

18:41 Added the pre-registered held-out KL scorer (PHASE5_DESIGN §2 promised Spearman *and* softmax-KL; only Spearman had
been implemented) plus a pooled within-row R² gain (theory+corpus vs theory, row-centred log-probs) to phase5/fingerprint.py.
Re-running the Wikipedia spelled and neutral views as *_v3 (rng streams differ from v2 because the extraction list is
shorter; v2 files kept). Post-hoc metric addition, declared as such: motivated by the gap between residual r ≈ 0.5–0.6
and ΔCV ≈ 0 in the neutral view (a within-row rank metric over 11 targets is insensitive to a few re-weighted entries).
18:41 phase5/ckpt_twins.py → results/phase5/ckpt_twins.txt: twin difference vectors across checkpoints. ΔQ directions
stabilize early (cos with the final model ≥ 0.86 from 49B tokens for E, Cb|B ≥ 0.9 from 21B); line-controlled alignment
with Wikipedia ΔC is scattered: E Gb|F# × D_doc becomes significant only at 4T (+0.88, p < .001), S2 (+0.86) and the
soup (+0.63, p .03); Db|C# × A_win40 +0.60–0.81 at 49B/294B/4T/S2 (p ≤ .05) but n.s. at 105B/1T/2T; Cb|B never
significant for E. Not a consistent fingerprint; recorded as counter-evidence to the STRONG-level twin reading.
18:41 Corpus downloads: OLMo-Mix wiki-0000 at ~1–8 MB/s (HF throttling varies); chain is a single pass with
[ -s ] guards, so a truncated file would not be re-fetched — sizes to be verified against HEAD content-length before
trusting the scans. Neutral-view cross-corpus fingerprints chained after crosscorpus.done (→ crosscorpus_neutral.done).

18:44 v3 Wikipedia re-run with the KL and within-row R² scorers (results/phase5/fingerprint_wikipedia_v3[_neutral].txt).
NEUTRAL view, E_modulation: theory-only held-out KL 0.13 nats/row; adding the corpus row reduces it by
ΔKL +0.037/+0.029/+0.016/+0.036 (A_win40; 1B/Gemma/Qwen/7B) and +0.045/+0.036/+0.023/+0.039 (D_doc), all p < .001 —
a 12–35 % reduction; within-row residual variance explained ΔR² +0.29/+0.30/+0.19/+0.33 (A) and +0.36/+0.38/+0.24/+0.38
(D), all p ≤ .007. C_harmonic: ΔR² +0.14/+0.03/+0.15/+0.15 (A; p .003/.27/.013/<.001), ΔKL significant for 1B, 7B,
Qwen (A) but not Gemma. D_chord: only 7B (ΔKL +0.06–0.07, p ≤ .003) and Qwen (A_win40 ΔR² +0.14, p .003).
The rank-based ΔCV that was ≈ 0 everywhere is a metric insensitivity (11 targets per held-out row), not absence of
held-out gain. SPELLED view: E × A_win40 ΔR² +0.10 (1B, p .02), +0.22 (7B, p < .001), Gemma/Qwen ≈ +0.02 (n.s.);
E × D_doc ΔR² +0.14 (1B), +0.12 (Gemma, p .03), +0.30 (7B); ΔKL tiny because the theory KL is already 0.05–0.07 (the
spelled rows are line-dominated). Robustness run with --rich theory features (quadratics, cosine harmonics) started.
18:44 Released OLMo-2-0425-1B 'main' weights are NOT the stage2-ingredient3-step23852 revision: all 179 tensors differ
(embedding max |Δ| 3.3 vs mean |w| 0.19). HF lists three stage-2 ingredient branches, each ending at step 23852
(51B); the model card says the 1B used ingredient 3 without merging. Unresolved; the two endpoints are reported as
separate points. Ingredients 1 and 2 will be scored as stage-2 replicates when bandwidth frees up.
18:44 Temporal alignment with cumulative data: not reconstructible from published artifacts — the 1B/7B/13B were trained
with the old OLMo trainer, only the 32B has official OLMo-core scripts, and no data-order indices are published for
the 1B; the only usable temporal intervention is Stage 1 (OLMo-Mix, 94.9 % DCLM, 0.09 % wiki) → Stage 2 (Dolmino 50B
mix: 47 % DCLM-HQ, 21 % math, 17 % FLAN, 7 % wiki, 6 % pes2o, 2.5 % StackExchange).

18:47 --rich robustness (quadratics + cosine harmonics in the LOO theory model): neutralized E_modulation held-out ΔKL
survives in all four models (+0.023/+0.019/+0.007/+0.018 A_win40; +0.034/+0.026/+0.013/+0.022 D_doc; p ≤ .017; theory
KL 0.11–0.12), ΔR² +0.17/+0.18/+0.06/+0.18 (A), +0.26/+0.29/+0.11/+0.24 (D). C_harmonic gains vanish (n.s. everywhere).
Spelled E × D_doc ΔR² +0.10/+0.20/+0.03/+0.20 (1B/Gemma/Qwen/7B; Qwen n.s.). Recorded in PHASE5_RESULTS §3.

19:13 Corpora downloaded and size-verified (results/phase5/corpora_verify.txt, 10.1 GB). Scans (results/phase5/scan_*.log):
OLMo-Mix wiki: 6,857 key docs, A_win40 8,540 pairs, D_doc 6,152 — the Dolma wiki (provenance en_simple_wiki_v0) has
documents 3× shorter than the 20231101 parquet dump (mean 2.2k vs 6.7k chars) and 5.5× fewer window pairs; it is a
different Wikipedia object, not a re-scan of the same text. OLMo-Mix DCLM (9 shards, 1.75 GB zstd ≈ 0.02 % of DCLM):
386 key docs, A 633 / D 467 pairs; Dolmino DCLM (2 shards): 611 docs, A 1,320 / D 845; Dolmino FLAN: 49 docs, 6 pairs
(unusable). All far sparser than Wikipedia (47k / 19k), so cross-corpus comparisons are made against size-matched
(binomially thinned) Wikipedia baselines (phase5/thin_wikipedia.py). Post-hoc expansion: 60 more DCLM shards (~12 GB,
gs00–09 local-0 files 1–6) queued after the ingredient downloads → cond_olmomix_dclm_big.npz (+ thinned baseline).

19:20 thin_wikipedia.py crashed on the 0-d 'ndocs' array (thinned baselines from the first chain were never written);
fixed, baselines regenerated (A_win40 47,247 → 8,648 / 655 / 1,385 for olmomix_wiki / olmomix_dclm / dolmino_dclm) and
their spelled + neutral fingerprints relaunched. Stage-2 ingredient-1 endpoint downloaded; ingredient 2 in progress.

19:26 CROSS-CORPUS (results/phase5/crosscorpus_compare_neutral.txt / _spelled.txt; figures/phase5/crosscorpus*.png).
Generic gain from every corpus in every model, sparse DCLM samples included; real sparse corpora beat size-matched
Wikipedia for OLMo and non-OLMo models alike. DiD (OLMo − others) × (corpus − thinned Wikipedia): OLMo-Mix wiki C×A
+0.007 (p .01) and E×D +0.008 (p .04), driven by the 7B; DCLM and Dolmino DiD ≈ 0 or negative. Verdict set:
CONDITIONAL STATISTICS PREDICT BEHAVIOUR — not a TRAINING-DATA FINGERPRINT; partial TEMPORAL ACQUISITION evidence.

19:35 Stage-2 ingredient 1/2 endpoints scored (results/phase5/behav_stage2-ingredient*.log); trajectory and twin tables
re-run with all three ingredients. Replicated: line|circle +0.33 (4T) → +0.42/+0.42/+0.44; residual D_doc +0.44 →
+0.48/+0.45/+0.46; A_win40 +0.26 → +0.28/+0.27/+0.27. Stage 2 (75× Wikipedia share) strengthens the line, not the
Wikipedia-specific residual. phase5/soup_check.py + cosine check: released 'main' is orthogonal to all branch
checkpoints (cos ≈ 0.00; singular values ≈ 2.2× larger; identical config) while ingredients are cos 0.96–1.0 to the
stage-1 endpoint — the released model is a different training run, not a soup and not the ingredient-3 endpoint.
Earlier wording 'soup' in figures/logs is superseded. DCLM expansion: global-shard_00 does not exist (6 requests
returned 15-byte error bodies, removed by the size check); the remaining 54 shards continue.

20:15 DCLM expansion done (54 shards; global-shard_00 absent): cond_olmomix_dclm_big.npz 2,884 key docs, A_win40 3,843.
Fingerprints + size-matched baseline run; DiD ≈ 0 in every cell (neutral −0.004…−0.000, p ≥ .45; spelled p ≥ .33).
The 633-pair 'real corpus beats thinned Wikipedia' genre effect is not uniform at 3,843 pairs (document above, window
below). Recorded in PHASE5_RESULTS §4. Paper drafting started in paper/ (Kirin style, tectonic/XeTeX; figures from
paper/make_figs.py; refs in paper/refs.bib).

20:42 PAPER REVIEW CORRECTION. The adversarial review of paper/main.tex found that PHASE5_RESULTS §4 and the draft
reported only 4 of the 9 pre-registered DiD cells per corpus (harmonic/modulation × window/document); the omitted cells
contain 5 significant OLMo-favouring effects in the neutral view (incl. D_chord×D_doc on both DCLM corpora: Dolmino
+0.049 p .00; 54-shard +0.023 p .01) and 5 in the spelled view. Verdict wording changed from 'not a fingerprint' to
'fingerprint not established; specificity inconclusive'. Also corrected: 12–35 % → 14–34 % (arithmetic); Fig. 7(a)
plotted the code-controlled line|circle (not comparable across code conditions per PHASE3_RESULTS §5) → now the
uncontrolled series; KL0 base vs rich distinguished; spelled-view KL scorer is null/negative in 5/8 modulation cells
(only ΔR² shows a gain) now stated; 7B prompt-final circle is in all six families (not 'relational'); Phase-IV aligned-
code p at r=.003 is .16 (n.s.); minor p-values and ranges. LITERATURE_AUDIT.md gained a paper-stage section.

21:49 SECOND PAPER REVIEW: DiD denominator corrected to 72 (10 significant, 34 negative); per-model decomposition
corrected (6/10 cells separate both OLMo models from both others; 4 are 7B-only — the previous 'each carried by one
model, Gemma gains as much' was wrong and understated the signal); abstract/glance 'none in the modulation family on
DCLM' restricted to the neutral view (spelled DCLM9 E×A +0.005 p .01 exists, fails to replicate at 3,843 pairs);
scope box no longer asserts absence of a fingerprint; aliasing '29 %' was a float-tie artefact — exact: 24.7 % > ½,
6.7 % = ½, 31.4 % ≥ ½ (phase2/aliasing.py made tie-aware; summary.txt regenerated); spelled-view KL confirms the gain
in 3/8 cells; twin cells 17/108 significant; Phase-IV thresholds 0.8k–7.7k, floor 0.001–0.007, ratio 3–24×.

22:03 SOL-PERSPECTIVE REVIEW integrated: central-claim box corrected (the directional conditional declines the enharmonic
identification and is fifths-smooth; it is the behaviour that is the line, and the conditional predicts it as a
regressor — controlled partials COND circle|line +0.55 > SYM +0.42); abstract restructured (six paragraphs, headline
number qualified to the neutralized modulation view); six context families defined in §2; DiD independence noted
(10 significant cells = 7 distinct combinations); aliasing statistic added to Result 2; HELP controlled partials
reported; document-conditional conjecture labelled untested; Singh & Chopra 2026, Prieto et al. 2026, Kandpal et al.
2023 cited; subtitle changed; figure layouts fixed; Fig 8(b) title no longer says 'corpus-specific'.

MANUSCRIPT HARDENING PASS (2026-08-29 22:00 – 08-30). Findings, all logged in MANUSCRIPT_AUDIT.md:
- P-VALUE AUDIT: every Phase-V permutation p was b/B with B=300 (held-out nulls), 2000 (residual, twins), 500 (checkpoints);
  no +1 correction; "p<.001" in the paper for ΔKL was printed from 0.000 at B=300 (floor 0.0033) — NOT justified.
  phase5/fingerprint.py rewritten: p=(b+1)/(B+1), B=5000 (Wikipedia) / 2000 (other corpora), parallel over cells, shared
  npz materialized (a lazily loaded NpzFile crashed under fork). Flagship neutral E cells: b=0 of 5000 (p ≤ 2e-4).
  ckpt_fingerprint.py: 2000 relabelings + finite-sample p (294B p .037, 4T .005, S2 .006/.009/.017).
- MATCHED OPERATORS (same 40-word counts N): symmetrized conditional (N+Nᵀ rows) predicts behaviour +0.82/+0.78/+0.80/+0.81
  (1B/Gemma/Qwen/7B) ≈ directional +0.80/+0.78/+0.79/+0.80; reverse conditional the same; PMI from the same counts
  +0.12/+0.16/+0.10/+0.24; ECI: conditionals 0.56–0.60, matched PMI 0.05. Held-out (neutral E): sym +0.039/+0.030/+0.016/
  +0.040 ≈ directional; reverse +0.029/+0.027/+0.013/+0.030; matched PMI +0.021/+0.014/+0.012/+0.029. DIRECTIONALITY
  PER SE CONTRIBUTES NOTHING; the contrast is conditional-row vs association construction. Central claim reframed.
- TARGET PRIOR (training-row target fixed effect in the theory model): E ΔKL rises to +0.041/+0.035/+0.025/+0.047 (A),
  +0.047/+0.042/+0.031/+0.049 (D), all p ≤ 2e-4; rich+prior +0.027/+0.022/+0.011/+0.025. Gain is source-specific.
- DOC-CLUSTER BOOTSTRAP (9,168 docs, B=300; scan_conditional --perdoc): flagship E ΔKL 95% CIs exclude 0 in all 8 cells.
- SHARD-CLUSTER BOOTSTRAP (Phase I, scripts/corpus_cluster_boot.py): fifths partial +0.62 [0.44,0.68] canonical, +0.66
  [0.49,0.71] merged; merged cl +0.58 [0.39,0.64], lc −0.07 [−0.20,0.07], P(cl>lc)=1.00; canonical P=0.63 (Poisson said 0.87).
- LENGTH-NORMALIZED SCORER: line survives in all 16 cells (E: +0.44→+0.37, +0.54→+0.43, +0.52→+0.50, +0.56→+0.45); sig
  templates 63/64 → 58/64 (results/phase5/scorer_robustness.txt).
- TEMPLATES: flagship gain significant in every leave-one-template-out aggregate; single templates 1,2,4 significant for
  all models; template 3 n.s. for Qwen/7B.
- Gemma added to the operator comparison. Bibliography verified against arXiv/publisher metadata (Korchinski D.J.,
  Nava Andres; Sadek Najla; Hu Zhimin/Niu Lanhao; Zhao & Thrampoulidis title; Feucht author list; Fu et al. COLM 2026).

V4 CORRECTION PASS (2026-08-30 14:00 → 2026-09-02). Review-driven, post-hoc feature correction; full record in
V4_CORRECTION_REPORT.md, notes/OPUS5_V4_VERIFICATION.md, notes/HANDOFF_2026-08-30.md.
- 08-30 14:00 Opus 5 review of paper-v1 (notes/review_opus5_v4.md): B1 — phase5/fingerprint.py's `circ` was the
  chromatic cyclic distance (pc = 7S mod 12, then cyclic distance), not the fifths-circle distance (Spearman −0.19
  between the two over the 165 aggregated pairs); B2 — target-aggregated line feature used the class-mean coordinate
  (−1, 0, +1 for the three merged classes); B3 — template clause misstated. Both confirmed before editing.
- 08-30 14:10 first v4 rerun launched (nearest-spelling + centroid features, global z-scoring, hash()-derived seeds);
  completed, then REJECTED at the 21:57 audit (global scaling; centroid retained; salted hash seeds). Its outputs were
  quarantined outside the repository, not deleted.
- 08-30 22:00 – 08-31 13:40 contract frozen (set-valued merged-class line features: min/max |s_t−s_i|, signed
  endpoints; true d5 + separate chromatic; fold-local tokenizer residual and scaling; SHA-256 seeds from master seed
  20260830; exact duplicate columns dropped); phase5/theory_features.py, fingerprint.py, ckpt_fingerprint.py,
  ckpt_twins.py, thin_wikipedia.py, crosscorpus_compare_v4.py, compare_v3_v4.py, validate_v4.py, v4_provenance.py
  rewritten; tests/test_phase5_v4.py (19 tests). Independent verifier: pass; independent reviewer NOT_READY twice
  (provenance, output-path safety, twin seed metadata) → repaired → READY on commit a56ad3e (08-31 21:01).
- 08-31 19:23 UTC full chain phase5/rerun_v4.sh (20 CPU workers; Wikipedia both views × base/rich × target prior,
  B=5,000; doc-cluster bootstraps base+rich, 300; matched sym/rev/pmi; templates t0–3, lo0–3; four corpora × two views
  + five thinning seeds each, B=2,000; checkpoints + twins) → 09-01 01:22 UTC V4_COMPUTE_AND_COMPARISON_COMPLETE.
  results/phase5/v3_v4_comparison.csv (57,348 rows), crosscorpus_compare_v4_{aggregated,spelled}.{json,txt},
  thinning_seed_variance.csv, ckpt_trajectory_v4.txt, ckpt_twins_v4.txt.
- RESULT: flagship rich-window ΔKL +0.0035/+0.0061/−0.0002/+0.0033 (p .034/.0004/.402/.0026), document
  +0.0039/+0.0031/−0.0003/−0.0003; only Gemma's cluster interval excludes 0; Qwen null everywhere; harmonic/chord
  families null; cross-corpus heterogeneous; DiD 8/72 split 4/4; checkpoint residual has no significant stage-1 point.
  The 14–34 % headline, cross-model universality, "gain in every corpus" and the 294B/4T acquisition threshold are
  withdrawn (PHASE5_RESULTS.md §10).
- 09-01 – 09-02 paper rewritten from the v4 artifacts (paper-v2): abstract, glance, central claim, contributions,
  Results 4/5/6/9, methods, discussion, limitations, Appendix A (B1/B2/standardization/RNG rows), C.2/C.3/C.5/C.7,
  D.6; make_figs.py reads v4 files (Figs 5, 7, C.1). paper-v1 (fb3e3e8) left immutable.
