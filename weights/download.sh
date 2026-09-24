#!/usr/bin/env bash
# Fetch the model weights into weights/ once, before the offline run (organizers' starter-kit convention).
#
# Files come from this repository's GitHub release "weights-v1". Each one is checked against
# weights/SHA256SUMS; files already present with the right checksum are kept. While the repository is
# private, export GITHUB_TOKEN (read access to the repository) first. scripts/fetch_weights.py rebuilds
# the same files from the official Ultralytics checkpoint.
#
# Usage:  bash weights/download.sh
set -euo pipefail

REPO="madaminovabdulaziz/roadwatch"
TAG="weights-v1"
DIR="$(cd "$(dirname "$0")" && pwd)"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() {  # fetch <asset name> <destination>
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    local asset_id
    asset_id=$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
      "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
      | python3 -c "import json, sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name'] == sys.argv[1]))" "$1")
    curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
      -o "$2" "https://api.github.com/repos/$REPO/releases/assets/$asset_id"
  else
    curl -fsSL -o "$2" "https://github.com/$REPO/releases/download/$TAG/$1"
  fi
}

while read -r expected name; do
  [[ -z "${name:-}" ]] && continue
  target="$DIR/$name"
  if [[ -f "$target" && "$(sha256 "$target")" == "$expected" ]]; then
    echo "$name: present"
    continue
  fi
  echo "$name: downloading"
  fetch "$name" "$target.part"
  actual="$(sha256 "$target.part")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$target.part"
    echo "$name: checksum mismatch (expected $expected, got $actual)" >&2
    exit 1
  fi
  mv "$target.part" "$target"
  echo "$name: ok"
done < "$DIR/SHA256SUMS"
