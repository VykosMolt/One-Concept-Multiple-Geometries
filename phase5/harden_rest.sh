#!/bin/bash
cd ~/Documents/Research/pitch_fourier; P=.venv/bin/python; M4=olmo2_1b,gemma2_2b,qwen25_3b,olmo2_7b; L=results/phase5/harden.log
run() { echo "== $(date +%H:%M:%S) $*" >> $L; $P -m phase5.fingerprint "$@" --jobs=20 >> $L 2>&1; }
run results/phase5/cond_olmomix_dclm.npz olmomix_dclm $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --neutral --nperm-res=2000 --nperm-loo=2000
for c in dolmino_dclm olmomix_dclm_big wikipedia_thin_olmomix_wiki wikipedia_thin_olmomix_dclm wikipedia_thin_dolmino_dclm wikipedia_thin_olmomix_dclm_big; do
  run results/phase5/cond_$c.npz $c $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --nperm-res=2000 --nperm-loo=2000
  run results/phase5/cond_$c.npz $c $M4 C_harmonic,D_chord,E_modulation A_win40,B_any,D_doc --neutral --nperm-res=2000 --nperm-loo=2000
done
echo HARDEN_DONE >> $L
