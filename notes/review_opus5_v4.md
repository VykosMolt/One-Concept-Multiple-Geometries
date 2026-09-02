# Adversarial review — *One Concept, Multiple Geometries* (tag `paper-v1`, HEAD `fb3e3e8`)

Reviewer: independent Opus 5 pass, 2026-08-30. Read-only on the repository except for this file.
Everything below was checked against the raw results files (`results/**`), the analysis code
(`phase5/*.py`, `phase2/keys15.py`, `scripts/corpus_cluster_boot.py`) and the compiled PDF.
Where I re-ran anything I did so on CPU with `.venv-cpu/bin/python`; my re-implementation of the
flagship leave-one-row-out regression reproduces the stored value
`olmo2_1b|E_modulation|A_win40 dkl = 0.036767766048443076` bit-for-bit, so the sensitivity numbers
below are computed with the paper's own pipeline, not an approximation of it.

---

## 0. Headline

The paper is unusually honest about its own history and most of its numbers check out. But three
things do not survive contact with the code and the files:

1. **The "circle" that the central held-out test controls for is not the circle of fifths.**
   `phase5/fingerprint.py:47,51` computes the *chromatic* cyclic distance and calls it `circ`.
   Over the 105 spelled pairs the coded feature correlates with the true circle-of-fifths distance
   at Spearman **−0.11** (−0.19 in the target-aggregated view). The abstract, the central-claim box
   and the contributions all say the gain is "beyond explicit **circle**, line, orthography and
   frequency baselines". As implemented, it is not.
2. **The "central number" (14–34 %) is an artefact of one arbitrary, undisclosed encoding choice.**
   Replacing "line distance to the merged class's *mean* line position" with "line distance to the
   *nearest* spelling of the class" — a strictly more defensible encoding of a log-sum-exp-merged
   class — cuts ΔKL by 65–87 % and turns the reduction into **2–16 %**, with Qwen × window becoming
   non-significant.
3. **Several reported ranges do not match the files they cite**, including the template-robustness
   clause (three errors in one parenthesis), "14 of 16" behavioural cells (the file says 15 of 16,
   and 63/64 in the same sentence implies 15 of 16), and the "+12 top-1 in 8–9 of 15 keys" statistic
   (it is 8–9 of *60* source × template cells).

The qualitative conclusion — that a local conditional-row operator adds replicated, positive held-out
predictive value beyond theory geometry — does survive my checks. The *magnitudes* and one of the
four models do not.

---

## 1. Numerical audit

`✓` = matches the raw file; `✗` = mismatch. Paper values are from `paper/main.tex` (line number given).

### 1.1 Corpus statistics (§3, Result 1, Figure 1–2)

| # | Claim (tex line) | Paper | File | |
|---|---|---|---|---|
| 1 | Wikipedia documents (L128) | 6.41 M | `results/corpus/wiki/report.json` `all/ndocs` = 6,407,814 | ✓ |
| 2 | Words (L128) | ≈3.1 B | `all/nwords` = 3,118,919,923 | ✓ |
| 3 | Months circulant fraction (L136) | 97 % | `all/months/circ_frac_offdiag` = 0.9678 | ✓ |
| 4 | Months paired spectrum (L136) | 0.50, 0.26, 0.12, 0.08, 0.03, 0.01 | `profile_abs_lambda` = [0.499, 0.256, 0.118, 0.081, 0.033, 0.012] | ✓ |
| 5 | Keys `M*` off-diagonal (L138) | 1.97–2.00 | computed from `all/major_canon/M`: 1.9744–1.9994 | ✓ |
| 6 | PMI kernel, fifths order (L138) | 8.64, 7.89, 7.59, 7.16, 7.07, 7.13, 6.48 | `major_canon@pmi/kappa` re-indexed by 7d′ mod 12: 8.637, 7.888, 7.592, 7.158, 7.071, 7.126, 6.475 | ✓ |
| 7 | Naive Fourier share (L138) | P₅ 0.50 vs P₁ 0.17 | `major_canon@pmi/profile_abs_lambda` = 0.505 / 0.172 | ✓ |
| 8 | Fifths partial, canonical (L150) | +0.62 [+0.44, +0.68] | `cluster_boot.txt` +0.623 [+0.442, +0.677] | ✓ |
| 9 | Fifths partial, merged (L150) | +0.66 [+0.49, +0.71] | +0.661 [+0.485, +0.708] | ✓ |
| 10 | Merged c\|l, l\|c (L150) | +0.58 [+0.39,+0.64]; −0.07 [−0.20,+0.07]; P=1.00 | +0.580 [+0.385,+0.643]; −0.070 [−0.195,+0.073]; 1.000 | ✓ |
| 11 | Canonical c\|l, l\|c (L150) | +0.40 [+0.05,+0.55]; +0.27 [−0.08,+0.58]; P=0.63 | +0.400 [+0.052,+0.551]; +0.269 [−0.077,+0.582]; 0.632 | ✓ |
| 12 | ECI₁₅ (L153) | 0.04 [0.03, 0.44] | +0.038 [+0.025, +0.440] | ✓ |
| 13 | White-key ordering (L150) | 78th of 5040, p=0.016 | `seam.txt` rank 78, p = 0.0155 | ✓ |
| 14 | Seam starvation (L150) | D♭\|F♯ 17 vs D♭\|G♭ 160 | `seam.txt` Db-F#=17; `pmi15.json` C[Db,Gb]=160 | ✓ |
| 15 | Twin PMI (L153) | 9.9–11.0 vs median 7.3 | `pmi15.txt` [9.92, 11.02, 9.78], median 7.3 | ✓ |
| 16 | 15-key partials (L153) | c\|l +0.42, l\|c +0.38 | `pmi15.json` stats: 0.4195 / 0.3794 | ✓ |
| 17 | Helper factorization inputs (L150) | V=3000, 14,235 docs | `keydocs_V3000.npz`: C (3000,3000), ndocs = 14235 | ✓ |
| 18 | Seam co-mentions (L153) | 41 | `enharmonic_pairs_context.json` = 41 entries | ✓ |
| 19 | "half of them containing 'enharmonic'" (L153) | half | 20 of 41 flagged `enharmonic: true` | ✓ |
| 20 | **"most of the rest slash notation" (L153)** | most of 21 | **3 of the 21 non-enharmonic segments contain a slash**; the rest are harp-notation prose and "changing the key from D♭ to C♯" narration. `PHASE2_RESULTS.md:125` said "slash notation **or harp-notation prose**"; the paper dropped the second half and thereby made it false | **✗** |
| 21 | Key-signature cue class (L153) | +0.68 / 0.00 | `results/phase2/corpus/conditional15.txt` line 6: +0.68±0.03 / +0.00±0.06 | ✓ |
| 22 | **Withdrawn Poisson bootstrap (L150)** | "gave ±0.02/±0.03 and P=0.87 **for the canonical family**" | `seam.txt`: canonical is ±0.05/±0.05, P=0.874; **±0.02/±0.03 are the merged-family SDs** (P=1.000). The two families' numbers are spliced together | **✗** |

### 1.2 Key-name geometry / aliasing (§4, Result 2, Table C.1)

| # | Claim (tex line) | Paper | File | |
|---|---|---|---|---|
| 23 | Accidental-indicator energy (L165) | 79.6 % in k=5/7 | `results/phase2/aliasing/summary.txt`: 0.796 | ✓ |
| 24 | Boxcar widths (L172) | 18/37/55/70/80/83 % | 0.182/0.373/0.553/0.700/0.796/0.829 | ✓ |
| 25 | Partition statistics (L165, L172) | 24.7 % > ½, 31.4 % ≥ ½, 6.6 % > 0.7, 6.7 % exactly ½ | 0.247 / 0.314 / 0.066 / 0.067 | ✓ |
| 26 | Root-letter identity (L172) | 34 % in the chromatic pair | 0.339 | ✓ |
| 27 | **Black-key Gram RSA, standard spelling (L165)** | "falls from **+0.70**…+0.86" | `results/decouple/*.txt`: the reported layers span **+0.56** (7B L2) to +0.86; Gemma L26 is +0.68 | **✗** |
| 28 | Respelled black-key RSA (L165) | −0.14…+0.12 | min −0.14 (7B L32), max +0.12 (1B L2) | ✓ |
| 29 | Respelled glyph RSA (L165) | +0.5/+0.4/+0.7/+0.3 | 0.51 / 0.43 / 0.67 / 0.28 | ✓ |
| 30 | Best-layer fifths partials + layers (Table C.1) | +0.21 (L13) / +0.29 (L23) / +0.23 (L14) / +0.34 (L30) | `results/nulls_multictx.txt`: +0.206@L13, +0.290@L23, +0.233@L14, +0.342@L30 | ✓ |
| 31 | Max-over-layer p, 7B (Table C.1) | 0.045 / 0.017 | 0.045 / 0.017 | ✓ |
| 32 | **Same p in text (L172)** | "p = 0.046 free / 0.018 block-preserving; 1,000 relabelings, (b+1)/(B+1)" | The table prints b/B, the text the (b+1)/(B+1) conversion of the same b. Two different numbers for one quantity, unexplained; and **Appendix D.6 (L541) says the Phase-II nulls are computed as b/B**, contradicting the parenthetical | **✗** |
| 33 | **"the other three are at p = 0.09–0.25" (L172)** | 0.09–0.25 | `nulls_multictx.txt`: 0.091–0.214 (max is Qwen block 0.193 / 1B block 0.214). 0.25 is not attained | **✗** |
| 34 | White-key exact-null p (Table C.1 note) | 0.037 / 0.003 / 0.025 / <0.001 | 0.037 / 0.003 / 0.025 / 0.000 | ✓ |
| 35 | **7B prompt-final c\|l (L172)** | +0.21…+0.36, p = .006–.03 | `results/phase2/geometry/olmo2_7b_symbol.json`: +0.209…+0.362 ✓, but p_max_free = **.004**…0.030 (D_chord .004) | **✗ (p lower bound)** |
| 36 | Qwen/7B last-token line (L172) | +0.26…+0.37, "significant in all six context families" | Qwen `__last` l\|c best = +0.263…+0.365 ✓; but significance holds only under the **free** null (p .004–.034). Under the **glyph-preserving** null Qwen is n.s. in all six (.062–.252) and 7B in 3 of 6. The paper does not say which null | **✗ (unstated null)** |
| 37 | **Subspace variance shares (L172)** | 0.53–0.73 and 0.46–0.66 | `results/phase2/respell/*_decomp.txt`, 120 `last`-position rows: spelling **0.53–0.76**, semantic **0.46–0.73** (sum > 1 in 120/120) | **✗ (upper bounds)** |
| 38 | **Context effects (L172)** | "+0.04…+0.07 band-averaged" | `results/phase2/contrast/summary_symbol.txt` `[final] cl relational−other`: +0.035 / **−0.026** / +0.071 / +0.054. `PHASE2_RESULTS.md:89` says "in three models" and `:176` says "Gemma shows none"; the paper dropped that qualifier | **✗** |

### 1.3 Behaviour (§5, Result 3, Table C.6)

| # | Claim (tex line) | Paper | File | |
|---|---|---|---|---|
| 39 | l\|c per model (L177) | +0.27…+0.44 / +0.32…+0.54 / +0.31…+0.52 / +0.14…+0.56 | `results/phase5/scorer_robustness.txt` totals: 0.27–0.44, 0.32–0.54, 0.31–0.52, 0.14–0.56 | ✓ |
| 40 | c\|l (L177) | −0.14…+0.11 (small models), +0.13…+0.28 (7B) | same file: −0.14…+0.11; +0.13…+0.28 | ✓ |
| 41 | **"4/4 in 14 of 16 cells" (L87, L177)** | 14 of 16 | `scorer_robustness.txt` and `results/phase2/behavior/summary.txt`: **15 of 16** cells are 4/4 (only 7B × B enharmonic is 3/4). Internally inconsistent with "63/64 significant templates" in the same sentence, which forces 15 of 16. `PHASE2_RESULTS.md:135` carries the same error | **✗** |
| 42 | 63/64 → 58/64 (L177) | 63/64, 58/64 | 15·4+3 = 63; 13+16+15+14 = 58 | ✓ |
| 43 | **"c\|l moves by at most 0.10" (L177)** | ≤0.10 | Table C.6 / `scorer_robustness.txt`: Gemma × D chord moves −0.14 → −0.03 = **0.11** | **✗** |
| 44 | ECI falls 0.05–0.25 (L177) | 0.05–0.25 | computed from C.6: min 0.05 (Gemma E), max 0.25 (1B D) | ✓ |
| 45 | Length-normalized modulation l\|c (L177) | 0.37 / 0.43 / 0.50 / 0.45 | same file | ✓ |
| 46 | Phase-I merged-target l\|c (L177) | +0.69→+0.46, +0.74→+0.52, +0.79→+0.66, +0.74→+0.56 | `results/predictive/{olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b}.txt`, `modulates_to` rows | ✓ |
| 47 | Merged-scorer c\|l (L177) | +0.14…+0.52, positive in every model | `results/phase2/behavior/summary.txt` `merged` rows: +0.14…+0.52 | ✓ |
| 48 | 7B enharmonic ECI (L180) | 0.11 total / 0.05 normalized; Qwen 0.51, 1B 0.70, Gemma 0.83 | `summary.txt` B_enharmonic rows: 0.11/0.05, 0.51, 0.70, 0.83 | ✓ |
| 49 | **"top-1 interval +12 in 8–9 of 15 keys" (L180)** | 8–9 of 15 keys | Recomputed from `results/phase2/behavior/olmo2_7b.json`: +12 is top-1 in **8 of 60** source × template cells (total scorer) and **9 of 60** (length-normalized); counting ±12 gives 12 and 15 of 60. Only 6 of the 15 sources have a twin at all, so the ceiling is 24 cells, not 15. Under the total scorer the *modal* top-1 interval is +1 (11/60), not +12 | **✗** |
| 50 | Outside-family ECI (L180) | 0.67–0.90 | `summary.txt` totals excluding family B: 0.67–0.90 | ✓ |
| 51 | Few-shot dominant accuracy (L180) | 0.08 / 0.33 / 0.75 / 0.92 | `results/behavior/fewshot_three_scorers.txt` raw | ✓ |
| 52 | PC1 tracks signed accidental count (L177) | \|ρ\| 0.70–0.80 | `RESULTS.md:96-100` table: 0.78, 0.73–0.78, 0.76–0.80, 0.70–0.78 (markdown only; no raw file) | ~ |

### 1.4 Operators (§6, Result 4)

All of `results/phase5/operators.txt` was checked line by line against L196.

| # | Claim | Paper | File | |
|---|---|---|---|---|
| 53 | Operator ECIs | SYM 0.04, 40-word PMI 0.05, directional 0.57, sym 0.56, reverse 0.60, helper 0.74 | 0.04 / 0.05 / 0.57 / 0.56 / 0.60 / 0.74 | ✓ |
| 54 | Operator controlled partials | c\|l +0.42/+0.62/+0.55/+0.55/+0.56/+0.21; l\|c +0.39/+0.14/+0.12/+0.18/+0.18/−0.07 | identical | ✓ |
| 55 | Behaviour Spearman, directional | +0.80/+0.78/+0.79/+0.80 | +0.80/+0.78/+0.79/+0.80 | ✓ |
| 56 | …symmetrized / reverse / helper / Karkada PMI / 40-word PMI | as printed | all match | ✓ |
| 57 | 7B prompt-final predictions | cond +0.60–0.61, helper +0.61, Karkada +0.46, 40-word +0.31 | identical | ✓ |
| 58 | Span-mean predictions | nothing above +0.15 (helper +0.37 for Qwen) | max +0.15 (SYM, 7B); Qwen helper +0.37 | ✓ |
| 59 | Document-conditional similarity | nothing above +0.14; −0.07…+0.03 for behaviour | +0.14 (prompt-final); behaviour −0.07…+0.03 | ✓ |
| 60 | Matched-operator ΔKL | sym +0.039/+0.030/+0.016/+0.040; dir +0.037/+0.029/+0.016/+0.036; rev +0.029/+0.027/+0.013/+0.030; PMI +0.021/+0.014/+0.012/+0.029 | `fingerprint/matched_{sym,rev,pmi}_neutral.json` | ✓ |
| 61 | **"all p ≤ .001" (L196)** | ≤.001 | max is **0.0016** (reverse × Qwen); sym × Qwen 0.0014, PMI × Gemma 0.0012. `MANUSCRIPT_AUDIT.md:29` carries the same slip | **✗** |
| 62 | **"retains half to three quarters" (L196)** | ½–¾ | PMI/symmetrized = 0.54, **0.46**, 0.76, 0.73; PMI/directional = 0.57, 0.48, 0.74, **0.81** | **✗ (loose)** |

### 1.5 Held-out prediction (§7, Result 5, Tables C.2 and C.7)

| # | Claim | Paper | File (`fingerprint/wikipedia_v3_neutral*.json`) | |
|---|---|---|---|---|
| 63 | ΔKL window | +0.037/+0.029/+0.016/+0.036 | 0.03677 / 0.02921 / 0.01639 / 0.03556 | ✓ |
| 64 | ΔKL document | +0.045/+0.036/+0.023/+0.039 | 0.04509 / 0.03591 / 0.02330 / 0.03931 | ✓ |
| 65 | Theory-only KL / "14–34 %" | 0.117–0.134; 14–34 % | 0.1167–0.1340; 14.05 %–33.7 % | ✓ |
| 66 | p ≤ 8×10⁻⁴; b=0 in six of eight | | max p = 0.0008; 0.0002 in 6 cells | ✓ |
| 67 | ΔR² window / document | +0.29/+0.30/+0.19/+0.33; +0.36/+0.38/+0.24/+0.38 | 0.286/0.298/0.185/0.329; 0.357/0.382/0.236/0.375 | ✓ |
| 68 | Document-cluster CIs (7 of 8) | as printed | `docboot` q025/q975 match to 3 dp | ✓ |
| 69 | **8th CI, 7B × document** | [+0.008, +0.040] | q025 = **0.00746** → rounds to +0.007 | **✗ (rounding)** |
| 70 | Rich ΔKL + p | +0.023/+0.019/+0.007/+0.018 (.0006/.0004/.013/.0002); +0.034/+0.026/+0.013/+0.022 (.0004/.0002/.003/.0008) | exact match | ✓ |
| 71 | Rich KL₀ 0.105–0.123, "6–28 %" | | 0.1052–0.1231; 6.4 %–27.6 % | ✓ |
| 72 | Target-prior ΔKL (base and rich) | +0.041/+0.035/+0.025/+0.047; +0.047/+0.042/+0.031/+0.049; +0.027/+0.022/+0.011/+0.025; +0.035/+0.029/+0.018/+0.029 | `_tp.json`, `_rich_tp.json`: exact | ✓ |
| 73 | Residual correspondence, document | +0.60/+0.59/+0.50/+0.60 (p ≤ 8e-4); rich +0.53/+0.52/+0.39/+0.51 | 0.5989/0.5881/0.5000/0.5982; 0.5294/0.5183/0.3908/0.5050 | ✓ |
| 74 | Spelled view: theory LOO Spearman 0.78–0.95 | | `wikipedia_v3.json` `loo.theory`: 0.7829–0.9543 | ✓ |
| 75 | Spelled ΔR² and ΔKL cells | +0.10/+0.22/≈+0.02; +0.14/+0.12/+0.05/+0.30; ΔKL +0.006/+0.012/+0.007, 7B×doc −0.002 | 0.102/0.220/0.022/0.023; 0.136/0.124/0.052/0.296; 0.0056/0.0117/0.0074/−0.0015 | ✓ |
| 76 | Spelled theory KL 0.05–0.07 | | 0.0523–0.0708 | ✓ |
| 77 | **Template robustness (L209)** | "ΔKL +0.026…+0.051 window, +0.030…+0.052 document, p ≤ .0005" for every leave-one-template-out aggregate | `wikipedia_v3_neutral_lo{0..3}.json`: window **+0.0145…+0.0465**, document **+0.0203…+0.0519**, max **p = 0.0040** (lo1 × Qwen × window). All three quoted figures are wrong; Table C.7 itself is correct | **✗✗✗** |
| 78 | Table C.7 (all 8 rows × 8 cells) | as printed | `wikipedia_v3_neutral_{t0..t3,lo0..lo3}.json`: every entry matches, and the `ns` marks match p ≥ .05 | ✓ |
| 79 | **Abstract "b = 0 of 5,000 relabelings in the flagship cells" (L68)** | reads as all | true in **6 of 8**; Result 5 says so correctly | **✗** |
| 80 | **Audit note "ΔCV ≈0 in the same cells (between −0.026 and +0.011)" (L225)** | −0.026…+0.011 | That is the **spelled**-view range. In the target-aggregated flagship cells ΔCV ranges **−0.026 to +0.047** (7B × window = +0.0467, p = .0002; Gemma × window/document +0.020/+0.022). The following clause discloses them, so the parenthetical contradicts its own sentence | **✗** |
| 81 | ΔCV significant cells | 7B five cells +0.028…+0.087; Gemma two at +0.020/+0.022, all p ≤ .008 | 7B: 0.0285, 0.0424, 0.0467, 0.0745, 0.0867 (p ≤ .0078); Gemma 0.0200/0.0224 (p .0064/.0038) | ✓ |
| 82 | Twin difference cosines | C♭\|B +0.65…+0.92; G♭\|F♯ +0.41…+0.86; D♭\|C♯ +0.02…+0.54 | `fingerprint_wikipedia_v3.txt`, 36 cells: 0.65–0.92 / 0.41–0.86 / 0.02–0.54 | ✓ |
| 83 | "seventeen of the 108 cells" | 17 | 17 at p ≤ 0.05 (15 at p < 0.05; the file rounds p to 2 dp, so the boundary is doing real work) | ✓(borderline) |

### 1.6 Cross-corpus (§8, Result 6, Table C.3)

| # | Claim | Paper | File | |
|---|---|---|---|---|
| 84 | Corpus sizes | OLMo-Mix wiki 6,857 docs / 8,540 pairs; DCLM 9-sh 386/633; 54-sh 2,884/3,843; Dolmino 611/1,320; FLAN 49/6 | `results/phase5/scan_*.log`: all exact | ✓ |
| 85 | Wikipedia extraction counts | 47,247 (A), 10,775 (B), 19,162 (D), 317 (C), 9,168 docs | `scan_wikipedia_perdoc.log`: identical | ✓ |
| 86 | 633-pair ΔKL | +0.037/+0.029/+0.030/+0.047, p ≤ .019 | `olmomix_dclm_neutral.json`: 0.0365/0.0290/0.0301/0.0471, max p 0.01849 | ✓ |
| 87 | 3,843-pair ΔR² and ΔKL | +0.18/+0.13/+0.15/+0.19 (p ≤ .0015); +0.021/+0.010/+0.017/+0.024, 7B p = .005 | `olmomix_dclm_big_neutral.json`: 0.1822/0.1275/0.1525/0.1904 (p .001/.001/.0015/.0005); 0.0211/0.0097/0.0171/0.0235, 7B p 0.0050 | ✓ |
| 88 | DCLM theory KL "inflated to 0.17–0.21" | | 0.1728–0.2098 | ✓ |
| 89 | 633-pair beats thinned Wikipedia | +0.024/+0.025/+0.025/+0.030 | `crosscorpus_compare_neutral.txt`: +0.0244/+0.0246/+0.0252/+0.0301 | ✓ |
| 90 | 3,843-pair mixed | doc +0.008…+0.011; window below for 3 of 4 by 0.007–0.016; Qwen +0.002 | `crosscorpus_compare_big_neutral.txt`: +0.0079…+0.0114; −0.0069/−0.0077/−0.0163; +0.0023 | ✓ |
| 91 | Table C.3, all 36 DiD entries | as printed | `crosscorpus_compare_{neutral,big_neutral}.txt`: all 36 values and p's match to the printed precision | ✓ |
| 92 | Spelled-view significant cells | +0.012 (.01), +0.012 (.01), +0.004 (.02), +0.005 (.01), +0.017 (.02); 54-sh all p ≥ .33 | +0.0115, +0.0116, +0.0037, +0.0046, +0.0166; 54-sh min p 0.33 | ✓ |
| 93 | DiD tally | 10 of 72 at p < .05, all positive; 36 positive / 34 negative / 2 zero; 0.004–0.05 nats | Parsed all 72 DiD lines: 10 sig, all positive, 36/34/2, range 0.0037–0.0491 | ✓ |
| 94 | Modulation DiD on DCLM | −0.014…+0.010, p ≥ .09 | −0.0138…+0.0095, min p 0.09 | ✓ |
| 95 | 9-sh spelled modulation × window fails to replicate at 3,843 | p = .76 | `crosscorpus_compare_big_spelled.txt` DiD p 0.76 | ✓ |

### 1.7 Synthetic (§9, Results 7–8, Table C.4)

| # | Claim | Paper | File | |
|---|---|---|---|---|
| 96 | Table C.4, all 20 cells | as printed | `results/phase3/analysis_main.txt` + `summary_main.json` | ✓ |
| 97 | Code-controlled l\|c footnote | 0.113 / 0.421 / 0.787 / 0.995 | `line_given_c` column: 0.1135 / 0.4207 / 0.7865 / 0.995 | ✓ |
| 98 | Uncontrolled l\|c vs oracle | 0.436 vs 0.422 against 0.439 | `analysis_uncontrolled.txt`: +0.436 / +0.422, oracle +0.439 | ✓ |
| 99 | Twin-target asymmetry | +0.013, 5/5, p = .044 | mean diff +0.013, 5/5 positive, t p = 0.044 (sign test p = 0.062) | ✓ |
| 100 | Early KL | 0.078 vs 0.276 at step 100 | trajectory table, step 100 | ✓ |
| 101 | Dose-response Spearman | +0.85 line, −0.68 KL | `dose_analysis.txt`: +0.85, −0.68 | ✓ |
| 102 | Representation shift | +0.08…+0.10, 5/5, p = .001 mid-layer | hidden_last +0.081 (t .014), hidden_mid +0.097 (t .001) | ✓ |
| 103 | Absolute partials / null q95 | −0.15 vs −0.23; +0.17…+0.21 | −0.15/−0.23 at last layer; q95 0.17–0.21 | ✓ |
| 104 | Positive control | 0.995 recovered; spurious c\|l +0.32 | line×permuted behaviour l\|c 0.995; hidden_mid c\|l +0.32 | ✓ |
| 105 | 91 runs | 91 | `results/phase4/runs`: 60 primary + 18 control + 12 extended + 1 smoke = 91 (the smoke run is counted) | ✓ |
| 106 | Global KL floor / rare-row ratio | 0.001–0.007; 3–24× | `phase4/analysis.txt`: 0.0010–0.0071; 3.3×–24.3× | ✓ |
| 107 | Power law | 1.18 N^−0.74; partials −0.84 / −0.19 | `exposure_collapse.txt`: 1.18·N^(−0.74); −0.843 / −0.186 | ✓ |
| 108 | Aligned-code effect | p = .02 / .035, n.s. at .16; prefactor 0.94 vs 1.5 | −0.0005 (p .020), −0.0014 (p .035), −0.0017 (p .157); 0.94 / 1.5 (exponents −0.72 vs −0.76, not a common exponent as stated) | ~ |
| 109 | Unique-state control | 0.0064 at 2.6k vs 0.0105 at 2.3k | 0.0064 @2585 is the r=.01 unique control; **0.0105 @2270 is the r=.003 rare alias**, a different arm. The comparison is exposure-matched but the sentence reads as within-arm; also 0.0064 < 0.0105 means the unique state is learned *better*, not "about as well" | ~ |
| 110 | Timing thresholds | rep ≈250–2,400; beh ≈0.8k–7.7k | file: rep 222–2425, beh 759–7698 | ~ |
| 111 | Natural ratios | C♭:B 0.024, C♯:D♭ 0.22, G♭:F♯ 0.41 | `PHASE4_RESULTS.md:26` — these are **rare shares** min/(min+max), not ratios: 121/438 = 0.28, 162/232 = 0.70. The colon notation is wrong for two of three | ~ |
| 112 | C♭/B rows differ | latent JS 0.228, 81st percentile | `PHASE4_RESULTS.md:9` only — no raw results file | ~ |

### 1.8 Checkpoints (§10, Result 9, Table C.5)

| # | Claim | Paper | File (`results/phase5/ckpt_trajectory.txt`) | |
|---|---|---|---|---|
| 113 | l\|c trajectory | −0.24 → +0.19 → +0.27 → +0.34 → +0.33 → +0.42/+0.42/+0.44 | −0.24, +0.19, +0.27, +0.27, +0.27, +0.34, +0.33, +0.33; s2i1 +0.42, s2i2 +0.42, s2i3 +0.44 | ✓ |
| 114 | Residual, document | +0.15 → +0.27 (.10) → +0.40 (.037) → +0.26 (.23) → +0.34 (.050) → +0.44 (.005) → +0.48/+0.45/+0.46 (≤.017) | +0.15 (.338), +0.27 (.095), +0.40 (.037), +0.26 (.233), +0.34 (.050), +0.44 (.005), +0.48/+0.45/+0.46 (.009/.017/.006) | ✓ |
| 115 | Table C.5, all 11 rows × 4 columns | as printed | every value matches (window p's .55/.49/.31/.10/.10/.33/.21/.061 ↔ .560/.496/.305/.104/.101/.332/.214/.061) | ✓ |
| 116 | Twin asymmetry | 4.3 → 2.6 | 4.34 → 2.63 | ✓ |
| 117 | Stage-2 deltas | line +0.09…+0.11 (3/3); residual +0.01…+0.04 | +0.09/+0.09/+0.11; +0.04/+0.01/+0.02 | ✓ |
| 118 | **"c\|l ≈ 0 throughout" (Result 9)** | ≈0 | −0.25 at 49B (the Figure 7 caption says so; the Result box does not) | **✗ (internal)** |
| 119 | Released-model provenance (L272) | cosine ≈0.00, singular values ≈2.2×, ingredients 0.96–1.0 | `PHASE5_RESULTS.md:157-158` / `PHASE5_LOG.md:135-136` only — **`phase5/soup_check.py` writes no file under `results/`**, so this paragraph does not satisfy the paper's own "every number traces to a results file" | **✗ (traceability)** |

### 1.9 Figures

| # | Claim | Paper | File | |
|---|---|---|---|---|
| 120 | Figure 1 caption | ρ = 0.25 (15-key), 0.68 (12-key) | computed from `phase2/keys15.py` geometries: 0.2494 and 0.6806 | ✓ |

**Score: 120 items checked, 20 mismatches**, of which one (#77) is a triple error in a single
robustness clause and two (#41, #49) are repeated in the "Results at a glance" box or used to
support a headline claim.

---

## 2. BLOCKING findings

### B1. The held-out regression never controls for the circle of fifths
`phase5/fingerprint.py:47` (target-aggregated) and `:51` (spelled) compute

```python
pc   = (7 * S) % 12                                   # S = signed line coordinate → pc = semitone pitch class
circ = np.minimum((pc[I] - J) % 12, (J - pc[I]) % 12) # cyclic distance in SEMITONES
```

`phase2/keys15.py` itself distinguishes the two geometries and names them correctly —
`circle_fifths = circ12(S_i, S_j)` and `chromatic = circ12(PC_i, PC_j)`. The feature used in the
flagship regression is `chromatic`, not `circle_fifths`. From source C, the coded "circle distance"
to G is **5** and to D♭ is **1** — the exact inverse of the circle of fifths. Over the 105 spelled
pairs the two correlate at Spearman −0.11; over the 165 target-aggregated pairs, −0.19.

The same feature is used in `phase5/ckpt_fingerprint.py:16-18`, so §10's residual correspondence
inherits it, and it propagates through `phase5/crosscorpus_compare.py` (which reads the fingerprint
JSONs) into §8. The *rich* model's harmonics `cos(2πk·d_c/12)` are therefore harmonics of the
chromatic distance. `scripts/corpus_cluster_boot.py` and `phase5/operators.py` are **correct** —
only the held-out pipeline is affected.

Why this is blocking rather than cosmetic: the abstract (L68), the central-claim box (L118), the
contributions (L120) and the Discussion (L266) all state that the corpus gain survives "beyond
explicit **circle**, line, orthography and frequency baselines". As implemented, the explicit
circle-of-fifths baseline is absent. Appendix D.6 (L541) also mis-describes the code.

Empirically the damage is bounded but real. Adding the true circle-of-fifths distance to the
target-aggregated modulation × window model:

| model | ΔKL as published | + true fifths circle | + fifths circle **and** nearest-spelling line |
|---|---|---|---|
| OLMo-1B | +0.0368 | +0.0309 | +0.0138 (p = .001) |
| Gemma | +0.0295 | +0.0268 | +0.0086 (p = .005) |
| Qwen | +0.0164 | +0.0166 | **+0.0028 (p = .053)** |
| OLMo-7B | +0.0356 | +0.0303 | +0.0130 (p = .002) |

### B2. "14–34 %" is not robust to the merged-class line encoding
Appendix D.6 defines the target-aggregated line feature as "line distance to the class's **mean**
line position". For the three enharmonic classes {C♭(−7), B(+5)}, {G♭(−6), F♯(+6)}, {D♭(−5), C♯(+7)}
the class mean is −1, 0 and +1 — coordinates that correspond to no key and that no line-ordered
model would produce. Because the target-aggregated response is a log-sum-exp over the two spellings,
it is dominated by whichever spelling the model prefers, so the *nearest* spelling's line distance is
the natural first-order predictor. Swapping mean → nearest (nothing else changed, same pipeline,
same nulls, B = 400–1000):

| cell | published ΔKL (p) | nearest-spelling ΔKL (p) | change |
|---|---|---|---|
| 1B × window | +0.0368 (.0004) | +0.0128 (.005) | −65 % |
| Gemma × window | +0.0292 (.0002) | +0.0077 (.005) | −74 % |
| Qwen × window | +0.0164 (.0008) | **+0.0021 (.100, n.s.)** | −87 % |
| 7B × window | +0.0356 (.0002) | +0.0105 (.008) | −71 % |
| 1B × document | +0.0451 (.0002) | +0.0173 (.003) | −62 % |
| Gemma × document | +0.0359 (.0002) | +0.0108 (.003) | −70 % |
| Qwen × document | +0.0233 (.0002) | +0.0057 (.025) | −76 % |
| 7B × document | +0.0393 (.0002) | +0.0122 (.008) | −69 % |

Theory-only KL₀ falls from 0.117–0.134 to 0.091–0.107, so the **"14–34 % reduction" becomes 2–16 %**
(and 3–13 % once the circle bug of B1 is also fixed). The rich model behaves the same way
(+0.0149/+0.0109/+0.0019ⁿˢ/+0.0119) and so does the target-prior variant
(+0.0137/+0.0086/+0.0037/+0.0142, vs the published +0.041/+0.035/+0.025/+0.047).

The paper runs robustness checks on the scorer, the templates, the nulls, the bootstrap, the target
prior and the rich model — but none on the construction of the theory features themselves, and the
whole positive claim lives in the view where that construction is most fragile (the spelled view's
gain is already ≤ +0.012 nats and significant in only 3 of 8 cells, and the paper says "only the
target-aggregated view is claimed"). A single, more defensible encoding removes two thirds to seven
eighths of the headline effect and one of four models. The direction and replication survive; the
stated magnitude does not.

### B3. The template-robustness clause misstates all three of its numbers
L209: "the flagship gain is significant in every leave-one-template-out aggregate for all four models
(ΔKL +0.026…+0.051 window, +0.030…+0.052 document, p ≤ .0005)". From
`results/phase5/fingerprint/wikipedia_v3_neutral_lo{0,1,2,3}.json`:

* window: +0.0145 … +0.0465 (Qwen is 0.0145–0.0200 throughout)
* document: +0.0203 … +0.0519
* max p = 0.0040 (lo1 × Qwen × window), not ≤ .0005

The *conclusion* ("significant in every leave-one-template-out aggregate") is true. Every published
figure supporting it is wrong, and each error is in the flattering direction. Table C.7, which the
sentence cites, contains the correct values.

---

## 3. MAJOR findings

### M1. The abstract's headline operator contrast is the untested statistic
The abstract (L68) leads with "aligns with those distributions far better (Spearman ≈ 0.80) than
PMI-type association operators … (0.1–0.5)". Those Spearman correlations
(`results/phase5/operators.txt`) carry **no null, no bootstrap and no confidence interval anywhere in
the paper** — over 105 mutually dependent pairwise distances, at hand-picked layers (7B L18 prompt-final,
7B L24 / Qwen L27 span-mean; the layers appear only in the results file, never in the paper). The
one place where the same comparison *is* tested — the matched held-out regressors of Result 4 — gives
a 2:1 ratio (+0.021 vs +0.039), not 7:1, and the paper says so ("the matched operators rank the same
way but less sharply"). An abstract should lead with the tested number.

### M2. The line does *not* survive the 15-key enharmonic-merged scorer, and the paper does not say so
The glance box (L87) asserts "the line survives enharmonic-merged scoring at roughly two thirds
strength". That is supported only by the Phase-I **12-key merged-target** analysis (+0.69→+0.46 etc.).
Under the **15-key enharmonic-merged scorer**, `results/phase2/behavior/summary.txt` `merged` rows give
line\|circle = **−0.08 / −0.23 / −0.21** (1B, B/C/D), −0.01 / −0.14 / −0.20 (Gemma),
−0.02 / −0.09 / −0.17 (Qwen), −0.06 / −0.03 / −0.12 (7B) — uniformly ≈0 or negative, and significant
in 0–2 of 4 templates in most cells. Result 3 (L177) reports only the favourable half of that table
(circle\|line turns positive) and never the line partial. The glance-box sentence does not distinguish
"merged target" from "merged scorer" and will be read as covering both.

Relatedly, the merged-scorer result is close to definitional: pooling the twins is what creates the
neutral pitch class, so ECI → 0.02–0.10 and circle\|line > 0 follow by construction. "Once twins are
pooled the periodic component reappears" reads as an empirical finding; it is largely an identity.

### M3. The corpus fifths partial had not converged, and the paper omits it
`results/corpus/wiki/convergence.json`: the controlled fifths partial rises monotonically
+0.27 (0.11 B words) → +0.24 → +0.35 → +0.43 → +0.45 (1.16 B) → **+0.63 (3.12 B)** with no plateau.
`RESULTS.md:47` states this explicitly ("had not plateaued"). The paper reports the months control as
"converged by 0.1B words" (L136) and then presents +0.62/+0.66 as a stable property of English
Wikipedia, with a shard-cluster CI that captures resampling noise but not this systematic trend. The
caveat is material for Result 1's framing ("robust periodic fifths structure") and for §8, where the
same statistic is computed on corpora three to four orders of magnitude smaller. It should be in the
paper, not only in the internal notes.

### M4. Two different control sets are printed under one symbol
`circle|line` and `line|circle` are defined once, in the glance box and §2.3, as "each controlling
for the other and for the orthographic and commonness features". In practice:

* `scripts/corpus_cluster_boot.py` (Result 1, 12-key): controls are the black-key block, log
  commonness and the other geometry — **no letter identity, no alphabet distance, no edit distance**.
* `phase5/operators.py` (Result 4, 15-key): `CTRL = [glyph_class, edit_distance, same_letter,
  alphabet, commonness]` plus the other geometry.
* `phase2/*` behaviour partials: a third set.

The paper compares these numbers across sections (Result 1's merged c\|l = +0.58 against Result 4's
SYM c\|l = +0.42, for instance) without noting that the residualization differs. At minimum this
needs a per-section statement of the control set.

### M5. The cross-corpus baseline is a single random thinning draw
`phase5/thin_wikipedia.py` produces the "size-matched Wikipedia" by one binomial thinning with
`seed = 0` (the seed is an optional third argument that the pipeline does not vary). For the 633-pair
comparison that is a 0.013 thinning of 47,247 pairs; for the 3,843-pair comparison, 0.081. The
resulting matrix is then treated as a fixed baseline in every DiD, and its own sampling variance
never enters the Wilcoxon. Result 6's "10 of 72 significant, all positive" and the panel-(b)
observation that "the window conditional falls below thinned Wikipedia for three of four models" are
both conditioned on one draw. Averaging ΔKL over, say, 20 thinning seeds is cheap and would settle it.

### M6. "+12 top-1 in 8–9 of 15 keys" (item #49)
Recomputed above: 8 and 9 of **60** source × template cells. The denominator inflates the apparent
rate by ~4×, and it is the evidence offered for "Enharmonic identity … is known behaviourally by the
7B model" — a claim that also carries the §5 conclusion that the two spaces differ at the seam. The
ECI evidence (0.11/0.05 vs a null of 0.5) is genuinely strong and sufficient; the top-1 statistic
should be restated (or dropped, since under the total scorer the modal top-1 interval in the
enharmonic family is +1, not +12).

---

## 4. Statistical audit

**Nulls.** I re-ran the flagship relabeling null (400 draws, 1B × E_modulation × A_win40) and the
null ΔKL distribution is centred at **−0.0007 with sd 0.0031** — well calibrated, not the negatively
biased distribution I expected from permuting a corpus column whose marginals no longer match the
frequency features. The observed +0.0368 is ~12 null-sd out. This is a point in the paper's favour
and it should be stated: the null is not anti-conservative.

**The joint key-relabeling null is the right null** for "is the corpus matrix's key labelling
arbitrary". It does not, however, test "does *any* smooth 15×15 matrix with the corpus's marginal
profile help", which is closer to what "beyond theory geometry" claims. A stronger null would
resample the corpus counts from a fitted theory-only model.

**Scorer invariance (a strength that is not claimed).** Because `loo` log-softmax-normalizes the
prediction within each held-out row, and because ΔR² and ΔCV are also within-row, all three scorers
are invariant to any source-level additive shift of `log C`. The corpus column's row normalizer —
which depends on self-repetition mass and on the source key's frequency — therefore cannot drive the
result. That is worth a sentence in D.6; it pre-empts an obvious reviewer objection.

**Multiplicity.** The audit note (L223) is candid and correct: 36 cells per view, no cell
pre-specified, modulation chosen post hoc, Bonferroni survived only by the base-model flagship cells.
Given B2, the Bonferroni argument no longer holds under the nearest-spelling encoding (p .003–.10 vs
36 cells), which strengthens the case for leaning on replication rather than thresholds — as the
paper already says it does.

**Bootstraps.** The document-cluster bootstrap (9,168 clusters, B = 300) is the right instrument and
its sd (0.0052 for 1B × window) exceeds the permutation null's spread (0.0031), as it should. The
shard-cluster bootstrap has only **41 clusters**, and its percentile intervals are visibly skewed
(+0.62 with CI [+0.44, +0.68]; ECI 0.04 with CI [0.03, 0.44]). With 41 clusters a percentile interval
is not well calibrated; a BCa interval or an explicit statement of the limitation would be honest.
The withdrawal of the Poisson bootstrap is correct and well documented (Appendix A), but the sentence
that describes it splices the merged family's SDs onto the canonical family's P (item #22).

**Ridge.** D.6 says "penalty λ = 1 on the standardized coefficients". The corpus column is **not**
standardized (`np.column_stack([Fx, cv])` with `cv = log C`, sd 0.43–1.13 depending on corpus and
extraction). I checked the magnitude: with ~154 training pairs, Σx² ≫ λ, so the shrinkage factor
differs by < 1 % across the observed scales and this changes nothing. But it means (a) D.6 is wrong,
and (b) λ = 1 is effectively no regularization at all — the estimator is OLS, which the text should
say rather than implying a tuned-free ridge.

**Leave-one-source-row-out.** Correct as described; the held-out row's response is never used, and
the z-scoring leakage is disclosed and negligible. The design is over 15 rows, which the paper
acknowledges as the binding power constraint.

**Twin analysis.** The line-controlled residualization uses five columns (intercept, S[keep], a
line-distance contrast |S−S_a|−|S−S_b|, flat indicator, sharp indicator); the paper describes only
"the target's line position and glyph class". The count "seventeen of 108" holds at p ≤ 0.05 with
p rounded to two decimals in `fingerprint_wikipedia_v3.txt`; at p < 0.05 it is 15. Since the paper
compares 17 against "the five expected by chance", the inclusive boundary matters and should be
stated.

**Synthetic n = 5.** Result 7's key inferences (twin asymmetry p = .044, representation shift
p = .001) are one-sample t-tests on five paired seeds. The sign test cannot go below 0.0625 with
n = 5, and `analysis_main.txt` reports exactly that for every one of these contrasts. Quoting only
the t p-value from five points is optimistic; the paper should quote both, as its own results file does.

---

## 5. Methods / code consistency

| Appendix D says | Code does | |
|---|---|---|
| D.6: theory features include "circle distance" | chromatic cyclic distance (`fingerprint.py:47,51`) | **✗ (B1)** |
| D.6: "penalty λ = 1 on the standardized coefficients" | the corpus column is not standardized | ✗ |
| D.6: nulls "B = 500 for the Phase-II geometry nulls and 1,000 for the Phase-II behaviour nulls (…computed as b/B)" | §4 (L172) quotes the key-name geometry p as "1,000 relabelings, (b+1)/(B+1)"; the Phase-I corpus nulls (B = 2,000) are not in D.6's inventory at all | ✗ |
| §10 (L263): checkpoints scored "under the rich theory model" | `ckpt_fingerprint.py:18` adds **15 source-row dummies** on top of the rich features. `PHASE5_RESULTS.md:64` calls this the "nuisance model (quadratics, cosine harmonics …, source-row dummies)". D.6's rich model has no row dummies, so §10's residual is not the §7 rich residual | ✗ |
| §6 (L192): matched PMI = log[(N+Nᵀ)_ij T / (r_i r_j)] | `operators.py` adds +0.5 smoothing to the numerator only | ~ (cosmetic) |
| D.3: family A "the scan stops at the first later mention beyond 40 words" | `scan_conditional.py`: `if nw > 40: break` | ✓ |
| D.3: family D "once per document for every pair of keys whose first mentions occur in that order" | `seen_first` loop | ✓ |
| D.3: family C "four relational regular expressions" | `PAT_C` has 4, but the fourth is `X major and Y major` — a conjunction, not a relation | ~ |
| D.2: sentence-initial matches excluded | `is_sentence_initial` applied; **not disclosed** that this preferentially deletes first mentions and therefore biases the ordered/document extraction | ~ |
| D.6: document-cluster bootstrap rebuilds counts and frequency features, B = 300 | matches | ✓ |
| D.6: joint key relabeling of the corpus matrix | `M[np.ix_(p, p)]` | ✓ |
| §5, Result 3: "Adding the model's own column marginal as a control raises the predictive partials" | traces to `RESULTS.md:89` (Phase-I 12-key matrices) only — no raw results file, and it is a statement about a different design from the surrounding 15-key sentences | ~ |

**Stale artefacts.** `results/phase5/fingerprint_wikipedia_v3.txt` (18:50) and
`results/phase5/fingerprint/wikipedia_v3.json` (22:38) disagree on p-values for the same cells
(e.g. 1B × E × A_win40 ΔKL p = 0.007 vs 0.0092) because the txt predates the B = 5,000 re-run. The
paper quotes the JSON, correctly, but the twin-difference numbers it quotes (L228) come from the
**older txt**, which is the only place they exist. A reader checking the repository will find two
files that disagree with no note saying which is current.

---

## 6. Logic and framing

**Does the reframed central claim follow?** Yes, and this is the paper's strongest move. The matched
operators (symmetrized and reversed conditionals from the same N) do as well as the directional one on
every observable, and the PMI from the same counts does much worse; the ECI split (conditionals
0.56–0.60 vs PMIs 0.04–0.05) is large and consistent. "Conditional-row vs association construction,
not directionality" is exactly what those numbers say. Appendix A records the retraction honestly.

**Is anything still written as if directionality mattered?** Two places. The abstract (L68) says
"An ordered key-mention conditional p(j|i) from a 40-word window aligns with those distributions far
better … *directionality itself contributes nothing*" — the ordering of clauses still foregrounds the
directional object, and "ordered" is doing rhetorical work the results deny. The Objects paragraph
(L205) and D.3 define everything in ordered terms. Consider leading with the symmetrized conditional,
which is the best performer on every metric (+0.82/+0.78/+0.80/+0.81; ΔKL +0.039/+0.030/+0.016/+0.040).

**Is the synthetic evidence over-used?** Mostly no. Result 7's framing ("insufficient for the
converged line") is exactly what the factorial shows, and the paper is careful that the +0.08…+0.10
representational shift is a *paired difference* with both absolute partials negative. Two overreaches:
(i) "which is why §7 could treat the line as data structure" (L261) — two falsified mechanisms do not
license treating the line as data structure, and the paper says as much one clause earlier; the
sentence should end at "the two most economical ones are"; (ii) Result 8's "An equally rare unique
state is learned about as well as the rare alias at matched exposure" compares two different rarity
arms and, in the numbers quoted, the unique state is learned *better* (0.0064 vs 0.0105) — which
strengthens the conclusion but is not what "about as well" says.

**Is §10 appropriately hedged?** Yes — "one run, one family with a clean signal, an unexplained dip
at 1T, and no reconstructible data order … suggestive, not a law" is the right register, and the
released-model provenance finding is reported against interest. Two nits: Result 9's "circle\|line ≈ 0
throughout" is contradicted by the −0.25 excursion that the figure caption acknowledges; and the
residual correspondence uses a nuisance model with source-row dummies that D.6 does not define.

**Is the "inconclusive" verdict on specificity right?** Yes, and it is the most disciplined section in
the paper. Ten significant cells out of 72 with ~3.6 expected, all positive, reducing to seven
distinct combinations, none replicating in the family that carries the main result, and nothing on the
corpus that is 95 % of OLMo's stage-1 tokens — "inconclusive" is the correct call and the reasoning is
laid out fairly. The only gap is M5 (single thinning draw).

**Scope of "one concept, multiple geometries".** The Discussion is careful ("one domain does not
establish a law"). Good.

---

## 7. Presentation

* **Figure 5(a,b)** is the paper's most information-dense panel and is legible, but almost every bar
  carries an asterisk, so the `*: p < .05` marking conveys nothing. Consider marking the exceptions.
* **Figure C.1** tick labels give the *window* pair mass ("8.5k", "633", "3.8k", "1.3k") on all three
  panels, including panel (a), which is modulation × **document** (where OLMo-Mix wiki has 6,152 pairs,
  not 8,540). Mislabeled.
* **Figure C.1** legend has one grey hatched entry for four thinned-Wikipedia bars; which hatched bar
  belongs to which model is positional only.
* **Figure 7** plots the three stage-2 seeds and the released model on the "stage-1 tokens (B)" log
  axis at x ≈ 6×10³ and 10⁴, positions that correspond to token counts they do not have. The caption
  explains it and tiny grey "S2"/"rel." labels mark the region, but the axis is still asserting
  something false. A broken axis or a separate strip would be cleaner.
* **Figure 4** caption does not give the layers (7B L18 prompt-final, 7B L24 / Qwen L27 span-mean),
  which appear only in `results/phase5/operators.txt`. Since these are best-layer-ish selections with
  no max-over-layers null — the correction the paper insists on everywhere else — the layers and their
  provenance belong in the caption.
* **Figures 5(c,d)** use a second, differently coloured two-entry legend inside panel (c) while the
  four-entry legend below serves (a,b). Minor inconsistency.
* **Notation**: `c|l` / `l|c` denote three different control sets (M4). ECI is used for four different
  objects (corpus PMI, operators, behavioural matrices, synthetic latent states) — fine, but the
  glance-box gloss only covers the first.
* **Table C.1** prints b/B p-values while the §4 text prints (b+1)/(B+1) of the same b; neither says so.

**References** (`paper/refs.bib`). I verified every pre-2026 entry against publication metadata:
Levy & Goldberg (NeurIPS 27, 2177–2185), Temperley (Music Analysis 19(3):289–319), Moss et al.
(JMM 17(2):173–197, DOI correct), Krumhansl & Kessler (Psych. Rev. 89(4):334–368), Chew (MIT thesis
2000), Quinn (PNM 44(2):114–158), Amiot (Springer CMS 2016), Chuan et al. (NCA 32:1023–1036, DOI
correct), Huang et al. (IUI 2016:241–250, DOI correct), Gurnee & Tegmark (ICLR 2024, 2310.02207),
Nanda et al. (ICLR 2023, 2301.05217), Zhong et al. (NeurIPS 36, 2306.17844), Kandpal et al. (ICML
2023, 2211.08411), Zhao et al. (2408.15417), OLMo 2 (2501.00656), Gemma 2 (2408.00118), Qwen2.5
(2412.15115), DCLM (2406.11794) — **all correct**. The 2026 entries are past my verification horizon
and I take no position on them. One formatting defect: `sadek2026circle` has
`booktitle = {Proceedings of Machine Learning Research}`, which is a series, not a venue — the actual
conference is missing. Nothing looks fabricated.

---

## 8. What I tried to break and could not

* The flagship pipeline is exactly reproducible: my independent re-implementation of `loo` returns
  `0.036767766048443076`, identical to the stored JSON to the last digit.
* The permutation null is well calibrated (mean −0.0007, sd 0.0031).
* All three held-out scorers are invariant to source-level shifts of the corpus column, so the row
  normalizer cannot be driving the gain.
* The target-prior control genuinely does what it claims, and the gain grows under it in every model
  and under every feature encoding I tried.
* Every one of the 36 target-aggregated DiD cells, the 36 spelled ones, all 12 rows of Table C.3, all
  16 rows of Table C.6, all 8 rows of Table C.7, all 11 rows of Table C.5 and all 8 rows of Table C.2
  reproduce exactly from the raw files.
* The Figure 1 caption's ρ = 0.25 / 0.68 is correct, which — given B1 — shows the *paper* knows which
  circle it means even where the code does not.

---

## Verdict

**MAJOR REVISION** — the operator-dependence result and the direction of the held-out gain survive, but the central number's stated magnitude does not, the regression does not control for the circle of fifths it claims to, and several reported ranges disagree with the files they cite.

### The five most important things to fix, ranked

1. **Fix `circ` in `phase5/fingerprint.py` (and `ckpt_fingerprint.py`) to be the circle-of-fifths
   distance `circ12(S_i, S_j)`, then re-run §7, §8 and §10 and re-state every number.** Until then the
   claim "beyond explicit circle … baselines" (abstract, central claim, contributions, discussion) is
   false as written. Expect ΔKL to drop ~15 %.
2. **Report the sensitivity of the flagship gain to the merged-class line encoding, and re-state the
   headline as a range across encodings.** Under "line distance to the nearest spelling" the reduction
   is 2–16 %, not 14–34 %, and Qwen × window is non-significant. Either justify the class-mean encoding
   against the alternative or lead with the conservative number. This is the difference between "the
   corpus conditionals add a third of the residual KL" and "they add a tenth of it in three of four
   models".
3. **Correct the template-robustness clause (L209) to +0.015…+0.047 window, +0.020…+0.052 document,
   p ≤ .004**, and correct the other 19 numerical mismatches listed in §1 — in particular "14 of 16"
   → 15 of 16 (L87, L177, and `PHASE2_RESULTS.md:135`), "+12 in 8–9 of 15 keys" → 8–9 of 60 cells
   (L180), the spliced Poisson-bootstrap SDs (L150), "p = 0.09–0.25" → 0.09–0.21 (L172), "all p ≤ .001"
   → ≤ .002 (L196), the ΔCV parenthetical (L225), and the abstract's "b = 0 … in the flagship cells"
   → in six of eight.
4. **Report the negative result under the 15-key enharmonic-merged scorer** (line\|circle −0.23…−0.01,
   mostly non-significant) alongside the positive circle\|line, and rewrite the glance-box bullet so it
   does not read as if the line survives merged *scoring*. While there, note that the merged-scorer
   circle result is largely definitional.
5. **Add the corpus-size non-convergence caveat to §3** (fifths partial +0.27 → +0.63 with no plateau,
   `results/corpus/wiki/convergence.json`), **average the cross-corpus thinning over multiple seeds**,
   and **reconcile Appendix D.6 with the code** (unstandardized corpus column; λ = 1 is effectively
   OLS; the §10 nuisance model's source-row dummies; the missing Phase-I null inventory).
