#!/usr/bin/env bash
# Download the sample videos listed in kaggle/samples.tsv (name <TAB> Google Drive file id) into
# $SAMPLES (default /tmp/samples). Files already present with the right size are kept.
#
# Usage from a notebook cell:  !bash /tmp/roadwatch/kaggle/fetch_samples.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLES="${SAMPLES:-/tmp/samples}"
mkdir -p "$SAMPLES"

while IFS=$'\t' read -r name file_id; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
  size=$(curl -sSL -r 0-0 -D - -o /dev/null "$url" | tr -d '\r' \
    | awk -F/ 'tolower($0) ~ /^content-range:/ {print $2}' | tail -1)
  if [[ -z "$size" ]]; then
    echo "$name: Drive did not return the file (not shared as 'anyone with the link', or quota exceeded)" >&2
    exit 1
  fi
  out="$SAMPLES/$name"
  if [[ -f "$out" && "$(stat -c %s "$out")" == "$size" ]]; then
    echo "$name: already downloaded"
    continue
  fi
  echo "$name: downloading $((size / 1000000)) MB"
  SECONDS=0
  curl -sSL --fail --retry 3 -o "$out.part" "$url"
  if [[ "$(stat -c %s "$out.part")" != "$size" ]]; then
    echo "$name: incomplete download ($(stat -c %s "$out.part") of $size bytes)" >&2
    exit 1
  fi
  mv "$out.part" "$out"
  echo "$name: done in ${SECONDS}s"
done < "$REPO/kaggle/samples.tsv"

ls -lh "$SAMPLES"
