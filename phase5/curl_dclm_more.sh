#!/bin/bash
# fetch additional OLMo-Mix DCLM shards (global shards 00-09, local shard 0, files 1-6) with size verification; resumable
cd ~/Documents/Research/pitch_fourier; H="Authorization: Bearer $HF_TOKEN"; B=https://huggingface.co/datasets/allenai/olmo-mix-1124
P=data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data
LOG=results/phase5/dclm_more.log
for gs in 00 01 02 03 04 05 06 07 08 09; do
  tree=$(curl -s -H "$H" "https://huggingface.co/api/datasets/allenai/olmo-mix-1124/tree/main/$P/global-shard_${gs}_of_10/local-shard_0_of_10")
  for i in 1 2 3 4 5 6; do
    f=$(printf "shard_%08d_processed.jsonl.zstd" $i); out=data/olmo_mix/dclm/gs${gs}_$f
    exp=$(echo "$tree" | python3 -c "import json,sys; d={x['path'].split('/')[-1]:x.get('size',0) for x in json.load(sys.stdin)}; print(d.get('$f',0))")
    for attempt in 1 2 3 4 5 6; do
      got=$(stat -c %s $out 2>/dev/null || echo 0); [ "$got" = "$exp" ] && break
      curl -sL -C - --speed-limit 50000 --speed-time 60 --retry 3 -H "$H" "$B/resolve/main/$P/global-shard_${gs}_of_10/local-shard_0_of_10/$f" -o $out; sleep 5
    done
    got=$(stat -c %s $out 2>/dev/null || echo 0); [ "$got" = "$exp" ] && echo "OK $out $got" >> $LOG || { echo "BAD $out $got vs $exp" >> $LOG; rm -f $out; }
  done
done
echo DCLM_MORE_DONE >> $LOG
