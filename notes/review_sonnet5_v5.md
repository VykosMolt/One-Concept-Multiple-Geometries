# Adversarial review — "One Concept, Multiple Geometries," paper-v2 (tag `paper-v2`, commit `24754d7`)

Reviewer: Claude Sonnet 5, independent pass per `notes/review_brief_v5.md`. Scope: `paper/main.tex` (full text,
lines 1–559), `paper/main.pdf` (24 pages, rendered and visually inspected), the Phase-V code
(`phase5/theory_features.py`, `phase5/fingerprint.py`, `phase5/operators.py`, `phase5/crosscorpus_compare_v4.py`),
results artifacts (`results/phase5/fingerprint/*_v4*.json`, `crosscorpus_compare_v4_*.json`,
`thinning_seed_variance.csv`, `results/corpus/wiki/cluster_boot.txt`, `results/phase5/operators.txt`), and the audit
trail (`V4_CORRECTION_REPORT.md`, `MANUSCRIPT_AUDIT.md`, `PHASE5_RESULTS.md`, `PHASE2_RESULTS.md`, `PHASE2_LOG.md`).
I did not re-run any analysis chain or GPU job; all numeric checks below are read-only recomputations from committed
artifacts, run in seconds.

## 0. What I verified numerically (30+ spot checks, zero transcription errors found)

Every number I traced matched its source file to the reported precision. This is worth stating plainly because it
is unusual: across Results 1, 4, 7, 8, Appendix C.2/C.3/C.5, and the Appendix B combinatorics, I found **no
arithmetic or transcription error**. Specifically verified (source in parentheses):

- Result 1: shard-cluster bootstrap CIs for canonical/merged fifths, `circle|line`, `line|circle`, ECI, and
  `P(circle>line)` — all eight numbers match `results/corpus/wiki/cluster_boot.txt` exactly.
- Result 4 / Figure 4: every ECI and controlled-partial value for all eight operators, and all 6×4 Spearman
  prediction values (behaviour and geometry columns), match `results/phase5/operators.txt` exactly.
- Result 7 / Table C.2: `dkl`, `dkl_p`, theory-only KL for base and rich, target-prior `dkl`/`p`, and all eight
  document-cluster bootstrap CIs (base and rich, window and document, all four models) match
  `results/phase5/fingerprint/wikipedia_v4_neutral{,_rich,_rich_tp,_docboot,_rich_docboot}.json` to 4 decimal
  places, including every reduction percentage (5.9/13.0/−0.4/7.3%, recomputed independently from the raw KL
  values).
- Result 8 / Appendix C.3: recomputed the full 144-cell direct-significance table and the full 72-cell DiD table
  from `results/phase5/crosscorpus_compare_v4_{aggregated,spelled}.json`; reproduced "8/0/6/0" and "7/3/8/2"
  significant-cell counts per corpus exactly, and reproduced "8 of 72 … four favouring OLMo and four opposing it
  (33 positive and 39 negative contrasts)" exactly, including the identity of all eight significant cells.
- Appendix C.3: reproduced the maximum per-cell thinning-seed SD (0.0061 aggregated / 0.0063 spelled) exactly from
  `results/phase5/thinning_seed_variance.csv`.
- Appendix B: independently recomputed the contiguous-block Fourier shares (18/37/55/70/80/83% for w=1..6, and
  79.6% for the accidental indicator) from first principles — exact match. On the "4,094 nontrivial binary
  partitions" statistic I first got a divergent number (27.0%/31.4% vs. the paper's 24.7%/31.4%) using a naive
  epsilon-tolerance re-implementation; tracing this down, I found the discrepancy is a well-known float-tie
  artifact that the authors already caught and fixed in an earlier round (`PHASE2_LOG.md`: "aliasing '29%' was a
  float-tie artefact… `phase2/aliasing.py` made tie-aware"), and running `results/phase2/aliasing/summary.txt`
  reproduces the paper's 24.7%/6.7%/31.4%/6.6% exactly. This is a point in the paper's favour, not a finding: the
  repository shows its own prior correction of exactly this class of error.
- Citation graph: every `\cite` key in `main.tex` resolves 1:1 to a `refs.bib` entry and vice versa (32 keys,
  no orphans either direction). I could not verify the external truth of 2026-dated references (arXiv IDs,
  venues) against ground truth — that is outside what I can check — but found no internal inconsistency,
  duplicate, or obviously fabricated-looking entry.

This level of fidelity is genuinely strong evidence that the "430 mechanical checks" claim in the brief is not
decorative. It also means my adversarial effort below is concentrated on statistics, logic, and framing rather than
number-chasing, per the brief's instruction.

---

## 1. Claim audit

**Abstract / glance box / central claim.** Internally consistent with the body; I did not find a claim in these
three places that isn't also stated, with the same hedges, in the corresponding Result box. The headline held-out
number (5.9/13.0/−0.4/7.3%) is quoted identically in all four places (abstract, glance box, central claim box,
Result 7) with the same "significant in three, cluster-bootstrap support only in Gemma" hedge attached every time.
That is good discipline — most manuscripts let the qualifier erode as it migrates from Results to Abstract, and
this one doesn't.

**Is §8 stated honestly?** Yes, and more so than most papers I have reviewed. Qwen's null is repeated, not
softened, at every occurrence (8 times in the main text alone). The document-cluster bootstrap section
(Result 7) explicitly states that only Gemma's interval excludes zero under the rich model, immediately after
reporting three significant relabeling p-values — the paper puts the weaker test right next to the stronger one
instead of leading with the flattering number. Result 8's "directionless" verdict (4 favouring / 4 opposing) is
reported with the exact split, not summarized as "inconclusive" in a way that could hide the balance. Result 9
withdraws its own earlier (v3) reading explicitly ("the earlier reading … is withdrawn").

**Is the headline (operator-dependent geometry + orthographic alias) earned by §3–§6?** Largely yes for the
descriptive claim. Result 1 (PMI is periodic, bootstrapped) and Result 2 (the fifths harmonic in hidden states is
an accidental-glyph artifact, respelling-confirmed) are both causal or quasi-causal interventions, not just
correlational RSA, and are the strongest parts of the paper. Result 3 (behaviour follows the line) is a large,
consistent, multiply-controlled effect (15/16 model×family cells significant, robust to length normalization).
Result 4 (conditional-row vs. association) is where I have the most reservations — see §2 below; the matched-operator
control is genuinely good work (it correctly identified and killed a prior "symmetric vs. directional" framing that
turned out to be wrong), but the remaining contrast rests on an ECI/JS statistic that gets less scrutiny than
its PMI counterpart in Result 1 (§2, Finding M2).

**Verdict on this axis:** no BLOCKING overclaims found. The paper's most exposed rhetorical moment is the
abstract's closing clause "the instrument lesson generalizes beyond music," which sits nine lines away from the
Limitations paragraph's "generalization is untested" (main.tex L284 vs. the abstract). These are reconcilable —
the "instrument lesson" is a domain-general *mathematical* fact about cyclic-group aliasing (proven in Appendix B
for any 12-cycle, not just keys) and a methodological principle, not an empirical claim that the specific
operator-dependent-geometry finding will replicate elsewhere — but the two sentences are close enough, and the
first is prominent enough (last sentence of the abstract, echoed as a subsection header "Methodological
implications"), that a hostile reader could reasonably accuse the paper of quietly having it both ways. **MINOR.**

## 2. Statistical audit

**Finding M1 (MAJOR) — the ridge penalty is de facto absent for the coefficient that matters, and de facto total
for the "rich" terms, and neither is disclosed.** Appendix D.6 states "Ridge regression uses an unpenalized
intercept and fixed λ=1" and nothing else about its behaviour. I fit the actual fold design
(`phase5/theory_features.make_fold_feature_set`, aggregated view, real Wikipedia frequencies from
`results/phase5/cond_wikipedia.npz`) and computed the eigenspectrum of the standardized training design (154
observations, 19 features base / 25 rich):

```
base:  eigenvalues of X'X range 1.75 .. 663;  shrinkage λ/(λ+eig) = 0.36 (weakest direction) down to 0.002 (strongest)
rich:  eigenvalues of X'X range 0.006 .. 1004; shrinkage λ/(λ+eig) = 0.994 (weakest direction) down to 0.001 (strongest)
corpus predictor coefficient variance ratio (ridge/OLS): 0.92 (base), 0.90 (rich)
```

Two consequences follow that the paper never states:

1. For the **corpus coefficient that produces the headline ΔKL**, ridge with λ=1 shrinks its estimated precision by
   only ~8–10% relative to unpenalized OLS on this design (n=154, p=19–25, standardized). The word "ridge" implies
   meaningful regularization; on the dimension the paper reports, it is close to OLS with a cosmetic penalty. This
   matters for interpretation (an OLS-flavoured LOO estimate is more exposed to the small-n overfitting the paper
   is otherwise careful about) and for reproducibility (a reader who assumes λ=1 does real shrinkage work here would
   be wrong).
2. For the **"rich" model's own added terms** — `circle_fifths_sq`, `line_max_abs_sq`, the two interaction terms,
   and the two cosine terms — the weakest identified direction has eigenvalue ≈0.006, giving shrinkage ≈99.4%. That
   direction is almost entirely determined by the penalty, not the data. This means the "rich" vs. "base" theory
   contrast that Table C.2, Figure 6, and the abstract lean on throughout §8 is not comparing two meaningfully
   different feature sets on this dimension — most of what "rich" adds beyond "base" is regularized to near-zero
   before it can do anything. The paper's own text ("the rich variant adds prespecified quadratic, interaction and
   two independent fifth-harmonic terms," D.6) presents these as substantive additions; the fitted coefficients on
   at least one of them are largely fixed by λ, not by evidence.

Neither the eigenspectrum, the shrinkage factors, nor a λ-sensitivity check (e.g., λ∈{0.1,1,10}) appears anywhere
in the paper or the results JSONs I inspected (`loo.fold_scalers` records the mean/scale, not the shrinkage). This
is exactly the "is the ridge with fixed λ=1 effectively OLS, and does the paper say so" question posed in the
brief, and the answer is: partially yes (for the load-bearing coefficient), and no, it is not disclosed.

**Finding M2 (MAJOR) — the row-conditional operators' headline ECI/JS statistic (Result 4, Figure 4) carries no
uncertainty quantification, unlike its PMI counterpart, and rests on badly imbalanced counts.** Result 1's PMI ECI
(0.04, 15-key) gets a full shard-cluster bootstrap: `[0.03, 0.44]`, with the paper explicitly attributing the wide
upper tail to "the 38 mentions of C♭ major" (main.tex L151). Result 4's conditional-row ECI (0.57, directional;
0.56, symmetrized; 0.60, reverse — the exact numbers that anchor "conditional-row operators keep spellings apart")
gets **no** bootstrap or resampling check anywhere in the paper, the results files, or `phase5/operators.py`. I
pulled the actual window-conditional row totals from `results/phase5/cond_wikipedia.npz` (`A_win40`):

```
Cb:  48 total co-occurrence events     Gb: 128     Db: 284
B:  1506 total co-occurrence events    F#: 156     C#:  66
```

The C♭/B pair — the twin driving most of the ECI, per the paper's own attribution of Result 1's uncertainty — has
a 31:1 count imbalance in the very statistic (`js` in `phase5/operators.py:38-41`, add-0.5 smoothing only, no
other correction) used to build the conditional-row distances. Jensen–Shannon divergence between two empirical
multinomials estimated from very different, and in one case tiny (48-event), samples is a well-known
positively-biased statistic: two rows that are truly identical will still show non-trivial JS divergence purely
from sampling noise when one has 48 observations spread over up to 14 targets. The paper does not test whether
subsampling B's row to C♭'s effective sample size collapses the apparent separation, and does not report a
bootstrap CI on the conditional ECI the way it does for the PMI ECI three sections earlier. This is not a claim
that the effect is spurious — Result 6's own diagnostic ("Wikipedia's next-key rows for C♭ and B are not the same
predictive state, latent JS 0.228, 81st percentile of all pairs") is suggestive that some of the separation is
real — but the paper applies materially more statistical rigor to the operator that supports the weaker,
more-hedged part of its story (Result 1's periodicity) than to the operator underwriting its central,
least-hedged claim (Result 4's "conditional-row operators keep spellings apart… while remaining fifths-smooth,"
which is promoted, unqualified, into the Central Claim box).

**Finding M3 (MAJOR) — the cross-corpus "fingerprint" specificity test has an unstated identifiability problem.**
The DiD test in §8.2 (Result 8) compares (OLMo models − Gemma/Qwen) × (corpus − matched-Wikipedia) and interprets
a null/directionless result as "no training-data fingerprint." This inference requires that Gemma-2 and
Qwen2.5 did *not* train on data resembling OLMo-Mix/DCLM/Wikipedia — but Gemma's and Qwen's training corpora are
undisclosed (both are closed-data models), and all four models almost certainly saw large amounts of Wikipedia and
Wikipedia-adjacent web text regardless of vendor. A null DiD is equally consistent with (a) no fingerprint exists,
and (b) a fingerprint exists but the "control" models (Gemma, Qwen) share enough of the tested corpora with the
"treatment" models (OLMo) that the contrast has no power. The Limitations paragraph's "the corpora are proxies
even for OLMo" gestures at data-representativeness in general but never states this specific identifiability
problem for the DiD design. Since the paper's own conclusion here is a null result, this doesn't overturn
anything — but the paper's stated interpretation ("neither generalizes… nor fingerprints," L260) should be
qualified as "this test could not detect a fingerprint even if one existed, given closed and likely-overlapping
training corpora," which is a different and weaker claim than "we tested for a fingerprint and did not find one."

**Finding M4 (MAJOR, related to M1) — "row-conditional vs. association" is partly a normalization-convention
difference, and the paper does not run a null model for that convention.** The matched-operator control (Result 4)
correctly shows directionality is not the active ingredient (symmetrized ≈ directional ≈ reverse, all ECI
0.56–0.60). What is not tested is whether *row-stochastic normalization itself* — dividing each row by its own
sum, as opposed to PMI's normalization by both marginals — mechanically produces larger per-row JS distances than
PMI produces Spearman-distance separations, on a corpus with no real key-specific structure at all (e.g., a
count matrix built from a null process with the true row/column marginals but no cell-level structure). Without
that ablation, "conditional-row operators are a fundamentally different, more distinction-preserving construction"
and "row-normalization mechanically inflates apparent distinctiveness relative to marginal normalization" remain
observationally similar, and the paper asserts the former without ruling out the latter.

**What is done well, statistically.** The relabeling null (`perm_keys`, jointly permuting corpus rows/columns while
holding theory features and behaviour fixed) is the *right* null for the specific claim being tested — "does the
true key-identity correspondence between corpus and model beat a random one, after conditioning on theory
features" — and the paper is self-aware that this conditions on the observed corpus (Limitations: "the pointwise
relabeling null conditions on the observed corpus, while document-cluster intervals … are much less decisive").
The fold-local preprocessing (tokenizer residuals and scalers fit only on the 14 training keys, response never
touching preprocessing) is implemented correctly — I read `theory_features.make_fold_feature_set`,
`fit_scaler`/`transform_scaler`, and `_prepare_loo` line by line and found no leakage. The B=5,000/2,000
resolution and `(b+1)/(B+1)` estimator are used consistently (I did not find a stray `b/B` anywhere in the
reported v4 numbers). The document-cluster bootstrap correctly rebuilds corpus matrices, frequencies, and raw
theory features inside every resample (`compute_cell`, `boot_dkl`/`docboot` blocks) rather than only refitting the
final regression. The training-row target-prior control is a sound way to test whether the corpus gain is a
disguised popularity prior. None of M1–M4 above reflects sloppiness in the engineering; they are places where a
methodological choice that *was* carefully executed was not matched with the same disclosure/uncertainty
treatment the paper gives elsewhere.

## 3. Feature-set audit

Appendix D.6's v4 feature set (true fifths-circle distance, separate chromatic distance, set-valued line
min/max/signed-endpoint features for merged classes, orthographic and frequency controls, fold-fitted tokenizer
residuals) is a substantial improvement over v3 and, given the constraints of the domain, is a reasonable
"scientifically strongest defensible" baseline for the specific comparison being run. I did not find an
obviously missing, cheap-to-add geometric or orthographic control (I checked for: relative-major/minor
confounds — not applicable, major-only 15-key design; parallel major/minor — not applicable, same reason; a
"famous-work salience" confound, e.g. well-known pieces skewing which keys get discussed together — plausible but
expensive to operationalize and arguably subsumed by log-frequency; a direct "shared substring/prefix" tokenizer
artifact beyond the fold-fitted residual — already handled).

One conceptual point the paper does not engage: the corpus conditional and the model's held-out behaviour could
covary not because the model's internal geometry "aligns with" the corpus's local co-occurrence structure, but
because both are downstream of the same small set of **textbook facts about key relationships** (dominant,
subdominant, relative minor, common modulation targets) that are stated in similar phrasing across many corpora,
including but not limited to Wikipedia — i.e., a memorization-of-facts story rather than a geometry-of-representations
story. The paper's own cross-corpus null (Result 8) is compatible with, and doesn't distinguish between, this
alternative and the paper's preferred "no fingerprint, no universal generalization" reading, but the alternative
is never named. This does not change the paper's stated (already-hedged) conclusion; it would strengthen the
Discussion to name and set aside the memorization alternative explicitly, the way the paper already does for the
lexical-code and frozen-transient alternatives in §7. **MAJOR** as a framing gap (it bears on how much
interpretive weight "held-out gain beyond theory" should carry), though it does not invalidate any reported number.

## 4. Logic and framing

**Does "operator-dependent geometry" follow from PMI vs. conditional rows, or is it partly definitional?** The
naive version of this worry (PMI is symmetric by construction, conditionals are directional by construction, so
of course they differ) is directly and convincingly pre-empted by the matched-operator control — this is good
science and I want to say so plainly, since a lazier version of this paper would have left the "symmetric vs.
directional" framing standing (as paper-v1 did, per `MANUSCRIPT_AUDIT.md` Q2) and it took a real, non-trivial
follow-up experiment to discover that directionality contributes nothing. The residual, subtler version of the
worry — row-normalization vs. doubly-marginal normalization as a construction difference, and small-sample JS
bias — is not addressed (Findings M2, M4 above).

**Does the matched-operator control really separate construction from directionality?** Yes, on its own terms: the
three matched operators are built from the identical N40 count matrix, holding window/weighting/scale fixed,
varying only symmetry. This is a clean design and the result (symmetrized ≈ directional ≈ reverse, PMI
different) is convincing as far as it goes.

**Is the orthographic-alias result (§4) as general as the paper says?** The core finding (accidental-glyph
indicator explains most of the P5 share; respelling moves the block with the glyph) is well-supported and the
respelling intervention is genuinely causal. The paper is appropriately modest about scope here — "only OLMo-2-7B
keeps a controlled fifths signal," "the apparent scale curve is three noise-level points and one significant one"
— and Appendix B's combinatorial argument (24.7–31.4% of all binary partitions alias into a paired mode) correctly
generalizes the *mechanism* beyond this one feature. I have no substantive complaint here.

**Is the synthetic evidence (§7) used for more than it can bear?** No — if anything the paper undersells it
appropriately: "output-code geometry alone… and sparse-alias learning are insufficient for the natural line" is a
narrow, correctly-scoped negative claim (rules out two specific mechanisms) rather than a positive claim about
what does explain the line, and the paper says so explicitly ("other lexical or learning explanations are not
thereby excluded").

**Is "One concept, multiple geometries" a finding or a truism, and does the paper make that case?** It is a
finding, not a truism, and the paper does make the case — the interesting content is not "different statistics
can look different" (trivially true) but the specific, surprising instrument failure (§4: a naive Fourier probe
mistakes an orthographic feature for a genuine tonal harmonic) plus the specific alignment result (§6: it's
construction, not directionality, that predicts behaviour). Those are non-obvious, falsifiable, and were arrived
at by killing several more obvious hypotheses first (Appendix A's ledger is, among other things, evidence that the
authors tried to find the boring explanation and had to be argued out of it by their own controls).

## 5. Structure and length

The restructuring (lead with §3–§6, demote Phase V to §8) is the right call given the v4 correction shrank the
Phase-V effect substantially; the paper would have been actively misleading if it still led with what is now a
small, model-dependent residual. The roadmap box and glance box are unusually well-executed reader aids for a
paper this dense.

24 pages is defensible *given the self-audit thesis* (a corrected preprint that wants to show its work) but is
long for most venues. Concretely:

- Appendix A (the retracted-claims ledger, ~40 rows) is valuable for transparency but reads as an internal
  engineering changelog rather than material a reviewer needs in the main submission; it would serve the paper
  better linked from the text and hosted as supplementary/repository material, with only the 4–5 most
  consequential retractions (B1, B2, the fingerprint retraction, the "14–34%→5.9/13.0/−0.4/7.3%" correction) kept
  in-line.
- Appendix C.7 (template robustness, a dense 8-row table of per-template ΔKL) and much of C.3's raw DiD table
  could be condensed to summary statistics with the full table left in the results files (already the case for
  the underlying JSON).
- Page 19 (Appendix C.1/C.2) has roughly half a page of unused white space below two short tables — a minor
  layout inefficiency, not a content issue.

**MINOR** collectively; I would not block on length, but I'd ask for a trim pass before venue submission
(see Top 5, below).

## 6. Presentation

Figures are clean, consistently styled, correctly captioned, and — notably — show null and negative results as
readily as positive ones (Figure 6's harmonic-relation panel is visibly mixed/negative; Figure 4's document-
conditional bars are near zero; Figure C.1 shows negative bars in every panel). This is a real strength; many
papers with figures this polished use the polish to make weak results look stronger, and this one does not.

- §6 (Result 4): "Predicting key-name span-mean geometry: nothing exceeds +0.15 except helper/Qwen at +0.37" is
  technically true only if read as "no *positive* correlation exceeds +0.15," but the same table
  (`results/phase5/operators.txt`) has COND `B_any`/7B at −0.20 and COND `D_doc`/Qwen at −0.34, both larger in
  magnitude than +0.15 and in the opposite direction. A reader skimming the prose (not the underlying table) would
  come away thinking key-name geometry is uniformly unpredicted by any operator, when in fact one operator anti-
  predicts it fairly strongly. **MINOR**, but an easy fix: state the range in absolute value or note the sign.
- Gemma-2-2B is loaded from `unsloth/gemma-2-2b`, "a bf16 safetensors mirror… used because the original repository
  is gated" (D.5), with no explicit fidelity check reported against the official weights — a real gap given the
  paper otherwise reports an explicit bf16/fp32 sensitivity check for OLMo-2-1B ("changed Fourier shares by
  ≤0.002"). Community mirrors of gated models are usually faithful re-uploads, but "usually" is not demonstrated
  here. **MINOR.**
- Notation (`\lc`, `\cl` for line|circle / circle|line partials) is used consistently everywhere I checked; I
  did not find a swapped instance.
- No fabricated-looking citations found; all keys resolve; I could not verify external facts for 2026-dated
  references given my own knowledge cutoff, which I disclose as a limitation of this review rather than a finding
  against the paper.

---

## Verdict

**MAJOR REVISION.**

Not because any reported number is wrong — I found none — and not because the central, most-hedged claims are
overclaimed; they are unusually well-calibrated for a paper of this scope, and the self-correction apparatus
(Appendix A, `V4_CORRECTION_REPORT.md`) is genuinely exemplary. The reason for Major rather than Minor is that I
found four/five substantive, checkable statistical-methodology gaps (M1–M4, feature-set framing) that bear
directly on how much weight the reader should put on the paper's own headline correction — exactly the class of
issue this paper has already twice rewritten itself over (B1: wrong circle predictor; B2: indefensible merged-line
centroid). A paper that holds itself to "every number traces to a results file" and "the ridge with fixed λ=1" gets
a stated methodological detail should also disclose that the same λ=1 is doing ~99% shrinkage on the "rich"
model's added terms and ~10% shrinkage on the coefficient the headline number is built from, and should give the
conditional-row ECI in Result 4 the same bootstrap treatment the PMI ECI gets in Result 1 three pages earlier. None
of these fixes require new data collection — they are checks and disclosures on data and code the repository
already has.

### Ranked top five fixes

1. **Disclose ridge shrinkage.** Report the shrinkage factor (or a λ-sensitivity curve) for the corpus predictor
   and for each "rich" term, at least in Appendix D.6; reframe "rich" vs. "base" comparisons with an explicit note
   that most of the rich model's added terms are heavily penalized toward zero on this design (Finding M1).
2. **Bootstrap the conditional-row ECI.** Give Result 4's ECI/JS statistics (Figure 4, the central-claim-anchoring
   "conditional-row operators keep spellings apart" number) the same shard/document-cluster bootstrap rigor
   Result 1 gives the PMI ECI, and directly test sensitivity to the C♭/B (48-vs-1506) count imbalance, e.g. by
   subsampling B's row to C♭'s count and reporting how much ECI moves (Finding M2).
3. **Name the corpus-overlap confound in the fingerprint test.** State explicitly in §8.2 that Gemma/Qwen's
   training data is undisclosed and may overlap with the tested corpora, so the DiD null cannot distinguish "no
   fingerprint" from "no power to detect one" (Finding M3).
4. **Fix the "nothing exceeds +0.15" sentence** to acknowledge the larger-magnitude negative correlations in the
   same table (§6).
5. **Trim the appendices for venue length**: move most of Appendix A's retraction ledger and Appendix C.7's
   per-template table to supplementary/repository material, keeping only the handful of consequential retractions
   in-line; this alone would bring the paper closer to a standard page budget without losing any auditable content
   (it all stays in the repository regardless).

### Counts

- BLOCKING: 0
- MAJOR: 5 (ridge/OLS non-disclosure — M1; conditional-row ECI uncertainty gap — M2; fingerprint-test corpus-overlap
  confound — M3; row-normalization-vs-association not ablated — M4; memorization-vs-geometry alternative not named
  in the feature-set framing)
- MINOR: 9 (abstract/limitations generalization-wording tension; "nothing exceeds +0.15" imprecision; Gemma mirror
  fidelity undisclosed; appendix length/layout; Figure 4 rightmost panel visually understating D_doc's negative
  correlation; matched-operator ΔKL numbers lack a document-cluster bootstrap; target-aggregated/spelled-view
  disagreement not flagged at first introduction in §2.2; unverifiable 2026 citations noted as a review limitation,
  not a finding; page-19 whitespace)

### One-paragraph assessment

This is an unusually rigorous, unusually honest empirical paper that has already survived one real correction
cycle (the v1→v2 rewrite after a wrong circle predictor and an indefensible merged-line centroid were caught) and
shows it in the text rather than hiding it — the Result boxes report Qwen's nulls and the cluster-bootstrap's
weaker verdicts right alongside the positive numbers, and I could not find a single miscomputed or mistranscribed
figure across thirty-plus independent checks spanning the abstract, nine Result boxes, and four appendix tables.
Its strongest material — the orthographic-alias finding (§4) and the construction-not-directionality result (§6)
— is genuinely good, causally-tested interpretability work that would be worth a strong workshop or a methods-
track slot at a mid-to-top ML venue as is. What keeps this at Major Revision rather than Accept is that the paper's
own standard for itself (explicit uncertainty on every headline number, explicit disclosure of every estimator
choice) is not quite met for the two statistics that anchor its narrower, more novel claims — the ridge behaviour
underlying the corrected held-out gain, and the ECI/JS statistic underlying "conditional-row operators keep
spellings apart." Both gaps are fixable from data already in the repository without new experiments, which is
exactly the situation this paper's own audit process is built to handle. Venue fit: a specialized interpretability
or cognitive-science-of-language-models venue (or a strong workshop) rather than a flagship ML conference's main
track — the domain is narrow by the paper's own admission ("one unusually structured semantic domain;
generalization is untested"), and the paper's real contribution is methodological (an instrument-failure case
study and a construction-vs-directionality result) rather than a result whose magnitude alone would carry a
top-tier venue.
