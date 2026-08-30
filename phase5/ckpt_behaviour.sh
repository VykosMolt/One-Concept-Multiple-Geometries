#!/bin/bash
# Run the 15-key behaviour (families C, D, E) on each OLMo-2-1B checkpoint as it finishes downloading.
cd ~/Documents/Research/pitch_fourier
for r in stage1-step300-tokens1B stage1-step10000-tokens21B stage1-step23100-tokens49B stage1-step50000-tokens105B stage1-step140000-tokens294B stage1-step480000-tokens1007B stage1-step950000-tokens1993B stage1-step1907359-tokens4001B stage2-ingredient3-step23852-tokens51B; do
  until grep -q "DONE $r" results/phase5/ckpt_download.log 2>/dev/null; do sleep 60; done
  tag="olmo2_1b_${r}"
  if [ ! -s results/phase2/behavior/${tag}.json ]; then .venv/bin/python phase2/behavior15.py models/olmo2_1b_ckpts/$r $tag fp32 - C_harmonic,D_chord,E_modulation > results/phase5/behav_${r}.log 2>&1; fi
  echo "BEHAV_DONE $r" >> results/phase5/ckpt_behaviour.done
done
echo ALL_BEHAV_DONE >> results/phase5/ckpt_behaviour.done
