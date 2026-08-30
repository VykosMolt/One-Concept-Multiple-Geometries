#!/bin/bash
# Manuscript-hardening reruns: finite-sample p (b+1)/(B+1), 5000 relabelings, parallel over cells. Log: results/phase5/harden.log
cd ~/Documents/Research/pitch_fourier; P=.venv/bin/python; W=results/phase5/cond_wikipedia.npz; M4=olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b; L=results/phase5/harden.log
run() { echo "== $(date +%H:%M:%S) $*" >> $L; $P -m phase5.fingerprint "$@" --jobs=20 >> $L 2>&1; }
run $W wikipedia_v3 $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --neutral --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,E_modulation A_win40,D_doc --neutral --rich --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,E_modulation A_win40,D_doc --rich --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --neutral --targetprior --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,E_modulation A_win40,D_doc --neutral --rich --targetprior --nperm-res=5000 --nperm-loo=5000
run $W wikipedia_v3 $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --targetprior --nperm-res=5000 --nperm-loo=5000
for op in sym rev pmi; do
  run results/phase5/cond_wikipedia_matched_$op.npz matched_$op $M4 C_harmonic,D_chord,E_modulation A_win40 --neutral --nperm-res=5000 --nperm-loo=5000
  run results/phase5/cond_wikipedia_matched_$op.npz matched_$op $M4 C_harmonic,D_chord,E_modulation A_win40 --nperm-res=5000 --nperm-loo=5000
  run results/phase5/cond_wikipedia_matched_$op.npz matched_$op $M4 C_harmonic,E_modulation A_win40 --neutral --targetprior --nperm-res=5000 --nperm-loo=5000
done
for t in 0 1 2 3; do
  run $W wikipedia_v3 $M4 E_modulation A_win40,D_doc --neutral --templates=$t --nperm-res=2000 --nperm-loo=2000 --tag=_t$t
  lo=$(echo 0 1 2 3 | tr ' ' '\n' | grep -v "^$t$" | paste -sd,)
  run $W wikipedia_v3 $M4 E_modulation A_win40,D_doc --neutral --templates=$lo --nperm-res=2000 --nperm-loo=2000 --tag=_lo$t
done
for c in olmomix_wiki olmomix_dclm dolmino_dclm olmomix_dclm_big wikipedia_thin_olmomix_wiki wikipedia_thin_olmomix_dclm wikipedia_thin_dolmino_dclm wikipedia_thin_olmomix_dclm_big; do
  run results/phase5/cond_$c.npz $c $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --nperm-res=2000 --nperm-loo=2000
  run results/phase5/cond_$c.npz $c $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --neutral --nperm-res=2000 --nperm-loo=2000
done
echo HARDEN_DONE >> $L
