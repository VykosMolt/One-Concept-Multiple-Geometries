#!/usr/bin/env bash
# Build the public mirror of this repository (code, docs, results files, figures, paper) without the
# large binary artifacts (model weights, corpora, hidden-state tensors, synthetic checkpoints).
# Usage: scripts/build_mirror.sh <mirror-dir>   (then commit/push from <mirror-dir>)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"; DST="${1:?mirror dir}"
mkdir -p "$DST"
rsync -a --delete \
  --exclude '.git/' --exclude '.git_old_bloated/' --exclude 'data/' --exclude 'models/' \
  --exclude '.venv/' --exclude '.venv-cpu/' --exclude '__pycache__/' --exclude '*.pyc' --exclude 'nohup.out' \
  --exclude 'results/phase2/hidden/' --exclude 'results/phase2/respell/*.npz' --exclude 'results/hidden/' \
  --exclude '*.pt' --exclude 'results/multictx/*/*.npz' --exclude 'results/corpus_merged/*.npz' \
  --exclude 'results/predict_position/*_H.npz' --exclude 'HARDEN_DONE' --exclude 'STOP' \
  "$SRC/" "$DST/"
# Manifest of everything tracked in the working repository but absent from the mirror.
cd "$SRC"
{
  echo "# Mirror manifest"
  echo
  echo "Working-repository commit: \`$(git rev-parse HEAD)\` (tag \`$(git describe --tags --exact-match 2>/dev/null || echo untagged)\`)."
  echo
  echo "Files tracked in the working repository but excluded from this mirror (size in bytes, SHA-256). They are"
  echo "derived artifacts: hidden-state tensors extracted from the released models by the scripts in this mirror,"
  echo "synthetic-model checkpoints from Phases III-IV, and merged corpus count matrices; every number in the paper"
  echo "is computed from the text/JSON results files that ARE included."
  echo
  echo "| file | bytes | sha256 |"
  echo "|---|---|---|"
  git ls-files -z | while IFS= read -r -d '' f; do
    [ -e "$DST/$f" ] && continue
    printf '| `%s` | %s | %s |\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-64)"
  done
} > "$DST/MIRROR_MANIFEST.md"
echo "mirror built at $DST; excluded files listed in MIRROR_MANIFEST.md ($(grep -c '^| `' "$DST/MIRROR_MANIFEST.md") entries)"
du -sh "$DST"
