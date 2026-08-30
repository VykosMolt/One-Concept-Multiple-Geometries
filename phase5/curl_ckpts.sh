#!/bin/bash
cd ~/Documents/Research/pitch_fourier; H="Authorization: Bearer $HF_TOKEN"; LOG=results/phase5/ckpt_download.log
for r in stage1-step950000-tokens1993B stage1-step1907359-tokens4001B stage2-ingredient3-step23852-tokens51B; do
  d=models/olmo2_1b_ckpts/$r; mkdir -p $d; [ -s $d/.complete ] && continue
  tree=$(curl -s -H "$H" "https://huggingface.co/api/models/allenai/OLMo-2-0425-1B/tree/main?revision=$r")
  files=$(echo "$tree" | python3 -c "import json,sys; print(' '.join(x['path'] for x in json.load(sys.stdin) if x['path'].endswith(('.safetensors','.json','.txt'))))")
  rm -rf $d/.cache; ok=1
  for f in $files; do
    exp=$(echo "$tree" | python3 -c "import json,sys; d={x['path']:x.get('size',0) for x in json.load(sys.stdin)}; print(d.get('$f',0))")
    for attempt in $(seq 1 12); do
      got=$(stat -c %s $d/$f 2>/dev/null || echo 0); [ "$got" = "$exp" ] && break
      curl -sL -C - --speed-limit 50000 --speed-time 60 --retry 3 -H "$H" "https://huggingface.co/allenai/OLMo-2-0425-1B/resolve/$r/$f" -o $d/$f; sleep 10
    done
    got=$(stat -c %s $d/$f 2>/dev/null || echo 0); [ "$got" = "$exp" ] || { echo "size mismatch $r/$f $got vs $exp" >> $LOG; ok=0; }
  done
  [ $ok = 1 ] && { echo ok > $d/.complete; echo "DONE $r" >> $LOG; }
done
grep -q "DONE stage2-ingredient3-step23852-tokens51B" $LOG && echo ALL_CKPTS_DONE >> $LOG
