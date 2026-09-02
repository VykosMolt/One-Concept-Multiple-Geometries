# pitch_fourier — do language co-occurrence statistics predict the Fourier structure of LLM pitch-class / key representations?

Autonomous multi-night research run, 2026-08-28/29. Start with `RESULTS.md` (Phase I verdict),
then `RESEARCH_LOG.md` (chronological, including what was decided before/after seeing data),
`MATH.md` (definitions and verified identities), `LITERATURE_AUDIT.md`, `NEXT_STEPS.md`.

## Phases (each has a *_DESIGN / *_LOG / *_RESULTS triple; results files start with the verdict)
- Phase I (`RESULTS.md`, tag `phase1-final`): corpus periodic fifths structure real; key-name geometry orthographic.
- Phase II (`PHASE2_RESULTS.md`, `phase2/`): 15-key enharmonic design; MIXED / MODEL-SPECIFIC.
- Phase III (`PHASE3_RESULTS.md`, `synthetic/`): output code causes a transient lexical bias, not converged behaviour.
- Phase IV (`PHASE4_RESULTS.md`, `synthetic/phase4*.py`): sparse-alias equivalence is sample-limited only.
- Phase V (`PHASE5_RESULTS.md`, `phase5/`): training-data fingerprint test. v4 correction (2026-09-02,
  `V4_CORRECTION_REPORT.md`): the held-out gain beyond a corrected tonal/orthographic baseline is small and
  model-dependent (3 of 4 models by relabeling, Qwen null); cross-corpus gains heterogeneous; fingerprint not
  supported; checkpoint residual trajectory weak. Paper-v1's 14–34 % headline used a chromatic "circle" feature.

## Layout
- `pf/fourier.py` — concept-axis DFT, paired energies, circulant projection, |M| prediction, nulls.
- `pf/extract.py`, `pf/families.py` — hidden-state extraction; concept families with *fixed* orderings.
- `corpus/concepts.py`, `corpus/scan_wiki.py`, `corpus/merge.py`, `corpus/analyze.py` — Karkada-style
  windowed co-occurrence (L=16, f(d)=L+1−d) on Wikipedia 20231101.en; M* and PMI variants.
- `scripts/` — drivers: `extract_all.py`, `spectra.py`, `corpus_report.py`, `compare.py`, plus
  exploratory experiments (each documented in RESEARCH_LOG.md).
- `tests/test_synthetic.py` — instrumentation tests (run: `python -m tests.test_synthetic`).
- `results/`, `figures/` — outputs (JSON + PNG). `data/` and `models/` are not committed.

## Reproduce
```
uv venv --python 3.12 .venv && uv pip install --index-url https://download.pytorch.org/whl/cu128 torch && uv pip install transformers accelerate safetensors numpy scipy matplotlib pyarrow scikit-learn pandas
python -m tests.test_synthetic
# corpus: download wikimedia/wikipedia 20231101.en parquet shards into data/wiki, then
python -m corpus.scan_wiki data/wiki results/corpus_wiki 16 && python -m corpus.merge results/corpus_wiki results/corpus_wiki/merged.json
python scripts/corpus_report.py results/corpus_wiki/merged.json wiki all,nocof,cof
# model
python scripts/extract_all.py models/OLMo-2-0425-1B olmo2_1b
python scripts/spectra.py olmo2_1b
python scripts/compare.py results/corpus/wiki/report.json all olmo2_1b '{"months":"months","weekdays":"weekdays","major_canon":"major_canon","minor_canon":"minor_canon"}'
# confounds, multi-context, decomposition, behaviour, predictive matrices, circle-vs-line
python scripts/confounds.py olmo2_1b major_canon anchor
python scripts/corpus_confounds.py results/corpus/wiki/report.json all major_canon,minor_canon
scripts/run_model_pipeline.sh models/OLMo-2-0425-1B olmo2_1b fp32      # multicontext + analyze + decompose + fewshot + predictive
python -m corpus.scan_vocab data/wiki results/corpus_merged/keydocs_V3000.npz 3000 20 && python scripts/theory_embedding.py results/corpus_merged/keydocs_V3000.npz pmi 300
python scripts/convergence.py; python scripts/circle_vs_line.py olmo2_1b,gemma2_2b,qwen25_3b
python scripts/predict_position.py models/OLMo-2-0425-1B olmo2_1b fp32; python scripts/fig_predict_position.py olmo2_1b,gemma2_2b,qwen25_3b
python scripts/figures.py olmo2_1b,gemma2_2b; python scripts/fig_summary.py olmo2_1b,gemma2_2b,qwen25_3b; python scripts/fig_circle_line.py olmo2_1b,gemma2_2b
```
Key figures: `figures/summary/fig1_corpus_kernels.png` (PMI kernels, semitone vs fifths order),
`fig2_corpus_matrices.png`, `fig3_<model>_P1P5.png` (layerwise P1/P5 raw vs black-projected),
`fig4_partial_fifths_summary.png` (corpus vs theory embedding vs models), `fig5_circle_vs_line_matrices.png`,
`fig6_predicting_position.png` (the fifths kernel appears at the state that predicts a key; circle mid-network → line at output).
Models used: allenai/OLMo-2-0425-1B (fp32), unsloth/gemma-2-2b mirror of google/gemma-2-2b (bf16),
Qwen/Qwen2.5-3B (bf16), allenai/OLMo-2-1124-7B (bf16, CPU offload) — weights not committed (`models/`).

## Public mirror

The paper links `github.com/VykosMolt/One-Concept-Multiple-Geometries`, a mirror of this repository built by
`scripts/build_mirror.sh` (code, docs, results files, figures, paper; model weights, corpora, hidden-state
tensors and synthetic checkpoints excluded and listed with SHA-256 digests in `MIRROR_MANIFEST.md`).
The corrected manuscript is tag `paper-v2.2` (round-5 reviews integrated, plain-language rewrite, synthetic and Phase-V sections moved to Appendices E–F); `paper-v2`/`paper-v2.1` are earlier corrected drafts and `paper-v1` (fb3e3e8) the immutable pre-correction snapshot.
