#!/bin/bash
# Scan Wikipedia shards as they finish downloading. A shard i is "complete" when shard i+1 exists
# (sequential downloader) or when all 41 exist and the downloader has exited.
cd ~/Documents/Research/pitch_fourier
while true; do
  n=$(ls data/wiki/*.parquet 2>/dev/null | wc -l)
  dl_running=$(pgrep -f "datasets/wikimedia/wikipedia/resolve" | wc -l)
  for f in data/wiki/train-*.parquet; do
    b=$(basename $f .parquet); i=$((10#${b:6:5}))
    nxt=$(printf "data/wiki/train-%05d-of-00041.parquet" $((i+1)))
    if [ -e "$nxt" ] || { [ "$n" -eq 41 ] && [ "$dl_running" -eq 0 ]; }; then
      [ -e "data/wiki_done/$b.parquet" ] || ln -s "$PWD/$f" "data/wiki_done/$b.parquet"
    fi
  done
  .venv-cpu/bin/python -m corpus.scan_wiki data/wiki_done results/corpus_wiki 16 2>&1 | grep -v "^0 shards"
  done_n=$(ls results/corpus_wiki/*.json 2>/dev/null | wc -l)
  echo "$(date +%H:%M) scanned=$done_n downloaded=$n dl_running=$dl_running"
  [ "$done_n" -ge 41 ] && break
  sleep 120
done
echo ALL_SHARDS_SCANNED
