#!/bin/bash
# Usage: run_model_pipeline.sh <model_path> <tag> <dtype> [device_map]
cd ~/Documents/Research/pitch_fourier
M=$1; T=$2; D=$3; DM=${4:-cuda}
P=.venv/bin/python
echo "=== $T start $(date +%H:%M)"
$P scripts/multicontext.py $M $T both $D $DM 2>&1 | grep -E "saved|done|Error|error" 
$P scripts/multictx_analyze.py $T results/corpus/wiki/report.json all > results/multictx/${T}_analysis.txt 2>&1
for fp in "major last" "major mean" "minor last"; do set -- $fp; $P scripts/decompose_rsa.py $T results/corpus/wiki/report.json $1 $2 2>&1 | grep -E "^##|^layer|^ +[0-9]+ " > results/multictx/${T}_decompose_$1_$2.txt; done
$P scripts/behavior_fewshot.py $M $T $D $DM 2>&1 | grep -E "^##|acc=" > results/behavior/${T}_fewshot.txt
$P scripts/predictive_matrix.py $M $T $D $DM 2>&1 | grep -E "^##|^[a-z_]+ +\|" > results/predictive/${T}.txt
$P scripts/predict_position.py $M $T $D $DM 2>&1 | grep -E "^##|^layer|^ +[0-9]+ \|" > results/predict_position/${T}.txt
echo "=== $T done $(date +%H:%M)"
