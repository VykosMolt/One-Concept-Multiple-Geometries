#!/bin/bash
# Usage: phase4_worker.sh <worker_id> <n_workers>; takes every n-th job from results/phase4/jobs.txt
cd ~/Documents/Research/pitch_fourier; W=$1; N=$2; i=0
while read -r job; do
  if [ $((i % N)) -eq $W ]; then
    set -- $job; d="results/phase4/runs/$6/$1_r$2_$3_s$4"
    if [ ! -s "$d/trajectory.json" ]; then .venv/bin/python -m synthetic.phase4 $job > "results/phase4/log_$6_$1_r$2_$3_s$4.txt" 2>&1; fi
  fi; i=$((i+1))
done < results/phase4/jobs.txt
echo WORKER_${W}_DONE >> results/phase4/workers.done
