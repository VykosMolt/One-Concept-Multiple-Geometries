#!/bin/bash
# Append the OLMo-2-7B results to RESULTS.md once the pipeline finished (called automatically).
cd ~/Documents/Research/pitch_fourier
until grep -q "=== olmo2_7b done" results/olmo7b_pipeline.log 2>/dev/null; do sleep 120; done
P=.venv/bin/python
$P scripts/fig_summary.py olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b > /dev/null 2>&1
CL=$($P scripts/circle_vs_line.py olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b 2>/dev/null | grep -E "olmo2_7b predictive")
{
echo; echo "## OLMo-2-7B (bf16, CPU offload) — appended automatically $(date +%H:%M)"; echo
echo '```'; echo "# token geometry, major keys, last token: partial fifths per layer (ctrl block+commonness+letter+alphabet)"
grep -E "^ +[0-9]+  fifths" results/multictx/olmo2_7b_decompose_major_last.txt
echo; echo "# black-block RSA per layer (col 6) — see results/multictx/olmo2_7b_decompose_major_last.txt"
echo; cat results/behavior/olmo2_7b_fewshot.txt; echo; cat results/predictive/olmo2_7b.txt; echo; echo "$CL"; echo; echo "# predicting-position geometry"; cat results/predict_position/olmo2_7b.txt; echo '```'
} >> RESULTS.md
git add -A && git -c user.name=VykosMolt -c user.email=illjaesterhazy@gmail.com commit -qm "OLMo-2-7B results appended automatically"
