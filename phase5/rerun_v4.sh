#!/usr/bin/env bash
set -euo pipefail

# Complete review-round-4 recompute.  This script writes only v4-labelled
# outputs; the tracked v3 seed-0 thinning inputs and all v3 JSON files remain
# untouched.  It uses the CPU environment because Phase-V statistics do not
# require an accelerator.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
P="${PYTHON:-.venv-cpu/bin/python}"
if [[ ! -x "$P" ]]; then
  echo "missing Python interpreter: $P" >&2
  exit 1
fi
W="results/phase5/cond_wikipedia.npz"
PD="results/phase5/cond_wikipedia_perdoc.npz"
MODELS="olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b"
JOBS="${JOBS:-20}"
LOG="results/phase5/rerun_v4.log"
MANIFEST="results/phase5/v4_provenance_manifest.json"
COMPLETION="results/phase5/V4_COMPUTE_AND_COMPARISON_COMPLETE.marker"
QUARANTINE="${ROOT}_v4_quarantine"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# Each cell is an independent process.  The underlying linear systems are
# tiny, so nested BLAS thread pools only oversubscribe the 24-core host.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -e "$LOG" ]]; then
  mkdir -p -- "$QUARANTINE"
  PRE="$QUARANTINE/rerun_v4.preacceptance.${STAMP}.log"
  suffix=0
  while [[ -e "$PRE" ]]; do
    suffix=$((suffix + 1))
    PRE="$QUARANTINE/rerun_v4.preacceptance.${STAMP}.${suffix}.log"
  done
  mv -- "$LOG" "$PRE"
  echo "preserved previous log in external quarantine: $PRE"
fi
if [[ -e "$COMPLETION" ]]; then
  mkdir -p -- "$QUARANTINE"
  OLD_MARKER="$QUARANTINE/V4_COMPUTE_AND_COMPARISON_COMPLETE.preacceptance.${STAMP}.marker"
  marker_suffix=0
  while [[ -e "$OLD_MARKER" ]]; do
    marker_suffix=$((marker_suffix + 1))
    OLD_MARKER="$QUARANTINE/V4_COMPUTE_AND_COMPARISON_COMPLETE.preacceptance.${STAMP}.${marker_suffix}.marker"
  done
  mv -- "$COMPLETION" "$OLD_MARKER"
  echo "preserved previous completion marker in external quarantine: $OLD_MARKER"
fi
: > "$LOG"

# Bind the complete run to one exact source/input snapshot.  The helper is
# deliberately fail-closed: missing inputs, changed Git HEAD/status, or any
# changed digest aborts the run before the next stage can consume outputs.
"$P" -m phase5.v4_provenance --write="$MANIFEST" >> "$LOG" 2>&1

verify_snapshot() {
  "$P" -m phase5.v4_provenance --verify="$MANIFEST" >> "$LOG" 2>&1
}

verify_snapshot
"$P" -m phase5.v4_provenance --preflight >> "$LOG" 2>&1
verify_snapshot

[[ -s "$W" ]] || { echo "missing Wikipedia input: $W" >&2; exit 1; }

run_fp() {
  local expected="$1"
  shift
  local -a model_items family_items extraction_items
  IFS=',' read -r -a model_items <<< "$3"
  IFS=',' read -r -a family_items <<< "$4"
  IFS=',' read -r -a extraction_items <<< "$5"
  local expected_cells=$(( ${#model_items[@]} * ${#family_items[@]} * ${#extraction_items[@]} ))
  verify_snapshot
  {
    printf '== %s fingerprint %s\n' "$(date -u +%H:%M:%S)" "$*"
    "$P" -m phase5.fingerprint "$@" --jobs="$JOBS"
  } >> "$LOG" 2>&1
  verify_snapshot
  [[ -s "$expected" ]] || { echo "missing expected output: $expected" >&2; exit 1; }
  "$P" -m phase5.validate_v4 --expected-cells="$expected_cells" "$expected" >> "$LOG" 2>&1
}

run_thin() {
  local target="$1"
  local output="$2"
  local replicate="$3"
  verify_snapshot
  [[ -s "$target" ]] || { echo "missing thinning target: $target" >&2; exit 1; }
  {
    printf '== %s thinning %s replicate=%s\n' "$(date -u +%H:%M:%S)" "$target" "$replicate"
    "$P" -m phase5.thin_wikipedia "$target" "$output" "$replicate"
  } >> "$LOG" 2>&1
  verify_snapshot
  [[ -s "$output" ]] || { echo "missing expected thinning output: $output" >&2; exit 1; }
}

run_ckpt() {
  verify_snapshot
  {
    printf '== %s checkpoint fingerprint\n' "$(date -u +%H:%M:%S)"
    "$P" -m phase5.ckpt_fingerprint
  } >> "$LOG" 2>&1
  verify_snapshot
  {
    printf '== %s checkpoint twin differences\n' "$(date -u +%H:%M:%S)"
    "$P" -m phase5.ckpt_twins
  } >> "$LOG" 2>&1
  verify_snapshot
  [[ -s results/phase5/ckpt_fingerprint_v4.json ]] || { echo "missing checkpoint JSON" >&2; exit 1; }
  [[ -s results/phase5/ckpt_trajectory_v4.txt ]] || { echo "missing checkpoint trajectory text" >&2; exit 1; }
  [[ -s results/phase5/ckpt_twins_v4.json ]] || { echo "missing checkpoint twin JSON" >&2; exit 1; }
  [[ -s results/phase5/ckpt_twins_v4.txt ]] || { echo "missing checkpoint twin text" >&2; exit 1; }
  "$P" -m phase5.validate_v4 \
    --checkpoint=results/phase5/ckpt_fingerprint_v4.json \
    --checkpoint-twins=results/phase5/ckpt_twins_v4.json >> "$LOG" 2>&1
}

run_comparisons() {
  verify_snapshot
  {
    printf '== %s exhaustive v3-v4 comparison\n' "$(date -u +%H:%M:%S)"
    "$P" -m phase5.compare_v3_v4
  } >> "$LOG" 2>&1
  verify_snapshot
  for output in \
    results/phase5/v3_v4_comparison.csv \
    results/phase5/v3_v4_comparison.md \
    results/phase5/thinning_seed_variance.csv; do
    [[ -s "$output" ]] || { echo "missing comparison output: $output" >&2; exit 1; }
  done

  verify_snapshot
  {
    printf '== %s corrected cross-corpus comparison\n' "$(date -u +%H:%M:%S)"
    "$P" -m phase5.crosscorpus_compare_v4
  } >> "$LOG" 2>&1
  verify_snapshot
  for output in \
    results/phase5/crosscorpus_compare_v4_spelled.json \
    results/phase5/crosscorpus_compare_v4_spelled.txt \
    results/phase5/crosscorpus_compare_v4_aggregated.json \
    results/phase5/crosscorpus_compare_v4_aggregated.txt; do
    [[ -s "$output" ]] || { echo "missing cross-corpus comparison output: $output" >&2; exit 1; }
  done
}

# A. Wikipedia: complete spelled/aggregated grid, rich and training-row prior.
run_fp results/phase5/fingerprint/wikipedia_v4.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
  --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_neutral.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
  --neutral --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_neutral_rich.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --neutral --rich --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_rich.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --rich --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_neutral_tp.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
  --neutral --targetprior --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_neutral_rich_tp.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --neutral --rich --targetprior --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_rich_tp.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --rich --targetprior --nperm-res=5000 --nperm-loo=5000
run_fp results/phase5/fingerprint/wikipedia_v4_tp.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
  --targetprior --nperm-res=5000 --nperm-loo=5000

# B. Document-cluster bootstrap under both corrected primary baselines.  The
# base run supports the direct v3->v4 comparison; the rich run is the
# scientifically strongest baseline.  Both rebuild corpus matrices,
# frequencies, and raw features inside every document resample.
run_fp results/phase5/fingerprint/wikipedia_v4_neutral_docboot.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --neutral --nperm-res=1000 --nperm-loo=1000 --docboot="$PD:300" --tag=_docboot
"$P" -m phase5.validate_v4 --expected-cells=16 --expected-docboot=300 \
  results/phase5/fingerprint/wikipedia_v4_neutral_docboot.json >> "$LOG" 2>&1
run_fp results/phase5/fingerprint/wikipedia_v4_neutral_rich_docboot.json \
  "$W" wikipedia_v4 "$MODELS" C_harmonic,E_modulation A_win40,D_doc \
  --neutral --rich --nperm-res=1000 --nperm-loo=1000 --docboot="$PD:300" --tag=_docboot
"$P" -m phase5.validate_v4 --expected-cells=16 --expected-docboot=300 \
  results/phase5/fingerprint/wikipedia_v4_neutral_rich_docboot.json >> "$LOG" 2>&1

# C. Matched same-count operators, both views and training-row prior.
for op in sym rev pmi; do
  IN="results/phase5/cond_wikipedia_matched_${op}.npz"
  run_fp "results/phase5/fingerprint/matched_${op}_v4.json" \
    "$IN" "matched_${op}_v4" "$MODELS" C_harmonic,D_chord,E_modulation A_win40 \
    --nperm-res=5000 --nperm-loo=5000
  run_fp "results/phase5/fingerprint/matched_${op}_v4_neutral.json" \
    "$IN" "matched_${op}_v4" "$MODELS" C_harmonic,D_chord,E_modulation A_win40 \
    --neutral --nperm-res=5000 --nperm-loo=5000
  run_fp "results/phase5/fingerprint/matched_${op}_v4_neutral_tp.json" \
    "$IN" "matched_${op}_v4" "$MODELS" C_harmonic,E_modulation A_win40 \
    --neutral --targetprior --nperm-res=5000 --nperm-loo=5000
done

# D. Single-template and leave-one-template-out robustness.
for t in 0 1 2 3; do
  run_fp "results/phase5/fingerprint/wikipedia_v4_neutral_t${t}.json" \
    "$W" wikipedia_v4 "$MODELS" E_modulation A_win40,D_doc \
    --neutral --templates="$t" --nperm-res=2000 --nperm-loo=2000 --tag="_t${t}"
  remaining=()
  for other in 0 1 2 3; do
    [[ "$other" == "$t" ]] || remaining+=("$other")
  done
  lo="${remaining[0]},${remaining[1]},${remaining[2]}"
  run_fp "results/phase5/fingerprint/wikipedia_v4_neutral_lo${t}.json" \
    "$W" wikipedia_v4 "$MODELS" E_modulation A_win40,D_doc \
    --neutral --templates="$lo" --nperm-res=2000 --nperm-loo=2000 --tag="_lo${t}"
done

# E. Cross-corpus cells and five deterministic Wikipedia thinning replicates.
for corpus in olmomix_wiki olmomix_dclm dolmino_dclm olmomix_dclm_big; do
  TARGET="results/phase5/cond_${corpus}.npz"
  run_thin "$TARGET" "results/phase5/cond_wikipedia_thin_${corpus}_v4.npz" 0
  for s in 1 2 3 4; do
    run_thin "$TARGET" "results/phase5/cond_wikipedia_thin_${corpus}_v4_s${s}.npz" "$s"
  done
  run_fp "results/phase5/fingerprint/${corpus}_v4.json" \
    "$TARGET" "${corpus}_v4" "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
    --nperm-res=2000 --nperm-loo=2000
  run_fp "results/phase5/fingerprint/${corpus}_v4_neutral.json" \
    "$TARGET" "${corpus}_v4" "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
    --neutral --nperm-res=2000 --nperm-loo=2000
  for s in 0 1 2 3 4; do
    if [[ "$s" == 0 ]]; then
      THIN="results/phase5/cond_wikipedia_thin_${corpus}_v4.npz"
      NAME="wikipedia_thin_${corpus}_v4"
    else
      THIN="results/phase5/cond_wikipedia_thin_${corpus}_v4_s${s}.npz"
      NAME="wikipedia_thin_${corpus}_v4_s${s}"
    fi
    run_fp "results/phase5/fingerprint/${NAME}.json" \
      "$THIN" "$NAME" "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
      --nperm-res=2000 --nperm-loo=2000
    run_fp "results/phase5/fingerprint/${NAME}_neutral.json" \
      "$THIN" "$NAME" "$MODELS" C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc \
      --neutral --nperm-res=2000 --nperm-loo=2000
  done
done

# F. Checkpoint residual trajectory under the same rich projection.
run_ckpt

# Comparison is part of the reproducible compute handoff.  This marker is
# intentionally not a manuscript-completion claim; paper integration and
# review remain separate downstream gates.
run_comparisons
verify_snapshot
printf 'V4_COMPUTE_AND_COMPARISON_COMPLETE\n' > "$COMPLETION"
printf 'V4_COMPUTE_AND_COMPARISON_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
