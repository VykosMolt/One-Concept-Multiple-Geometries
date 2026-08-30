# PHASE4_LOG.md — chronological (machine clock)

Sat Aug 29 12:04:39 PM CEST 2026 — Phase IV start. Design frozen in PHASE4_DESIGN.md. Natural calibration from Phase-II Wikipedia counts:
Cb:B 38:1544 (r = 0.024), C#:Db 121:438 (r = 0.22), Gb:F# 162:232 (r = 0.41); rare-alias absolute counts 38–162 per
3.1B words; explicit enharmonic co-mention windows 7/18/9. Sweep r ∈ {0.5, 0.1, 0.03, 0.01, 0.003, 0.001}.
Differences from Phase III (deliberate): classes sampled uniformly (1/12 each) instead of the 15 observed labels
uniformly; data sampled online each step (exposure = cumulative draws) instead of a fixed 300k set; cosine LR floors
at 10% so extended runs keep learning. Targets: Phase-III CIRCLE law unchanged.

12:05 Smoke (alias r=0.01, aligned, seed 0, 2000 steps, 21 s): 1,245 rare exposures → latent twin JS 0.004,
rare-row KL 0.017 vs common 0.002, global 0.005; Q twin distance 0.35 of the mean non-twin distance. Sweep launched
(90 runs, 3 workers): primary 6 r × 2 codes × 5 seeds @12k; unique-state control 3 r × 2 × 3 @12k; extended r ∈
{0.003, 0.001} × 2 × 3 @48k.
Literature (targeted, not re-read in full): grokking/delayed generalization (Nanda et al. 2023; 2026 capacity/phase-
transition papers; GrokTransfer ICLR 2025 — data embedding decides whether generalization is delayed) and long-tail
fact learning (Kandpal et al. 2023 'LLMs struggle to learn long-tail knowledge'; Chang et al. NeurIPS 2024 'How do LLMs
acquire factual knowledge'; 2025 frequency–recall linear relations). None studies learning an *equivalence* between
two surface forms with the same latent state; the closest analogue is entity-alias / paraphrase consistency. We only use
"grokking" if the curves show generalization long after the training loss plateaus.

13:59 SWEEP RESULTS (91 runs; results/phase4/analysis.txt, exposure_collapse.txt; figures/phase4/curves.png)
Tolerance (3 × q95 of balanced-run final latent JS) = 0.00025. Balanced runs reach it after ≈ 1.2–3.1×10⁵ rare exposures.
Sparse runs never reach it within budget (r ≤ 0.01), but the error is a smooth power law in cumulative rare-alias
exposures: latent JS ≈ 1.18 · exposure^(−0.74) (Spearman −0.86, n = 1254 checkpoints); partial Spearman with exposure
given step −0.84, with step given exposure −0.19 → sample-limited, not frozen. Exposure-matched medians coincide across
r (e.g. 1600–3200 exposures: 0.0034/0.0032/0.0024/0.0024/0.0025 for r = .1/.03/.01/.003/.001, aligned).
Code effect: aligned code LOWERS latent JS at every sparse r (final r=.01: 0.0006 vs 0.0011, p .02; r=.001: 0.0049 vs
0.0063, p .035; prefactor 0.94 vs 1.5) — the opposite of "code delays equivalence".
Global vs rare: at 12k steps global KL 0.002–0.007 with rare-row KL 3–10× the common-row KL — a real but small
unresolved residual that shrinks as exposure^−0.7.
Representation: Q twin distance falls below 0.3 of the non-twin distance at ≈ 250–2,000 exposures, before behavioural
JS < 0.002 (1.5k–7.7k exposures) — representation leads behaviour.
Unique-state control: an equally rare unique state has row KL comparable to the rare alias at matched exposure
(aligned: 0.0064 at 2.6k vs 0.0105 at 2.3k; permuted 0.0129 vs 0.0158): the model does not exploit the equivalence.
Bridge to natural data: Wikipedia's conditional next-key rows for Cb and B are NOT equivalent (latent JS 0.228, 81st
percentile of all pairs; Cb → B/Cb .37, Gb/F# .22, Eb .11; B → D .15, E .14, C .14, F .13); Gb/F# 0.050 (22nd pct),
Db/C# 0.068 (32nd pct). The rarest natural alias does not share its twin's predictive state in the data.
