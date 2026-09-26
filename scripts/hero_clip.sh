#!/usr/bin/env bash
# Home-page background loop: a short stretch of the raw 4K sample, cut to a seamless loop (the last
# FADE seconds cross-fade into the first ones), H.264 without audio at 1080p and 720p, plus a poster.
# Usage: bash scripts/hero_clip.sh [source.MP4] [start_sec] [length_sec]
set -euo pipefail

SRC="${1:-samples/C3902.MP4}"
START="${2:-125}"
LEN="${3:-15}"
FADE=1.5
OUT=web/public/data/hero
mkdir -p "$OUT"

MAIN=$(echo "$LEN - $FADE" | bc)      # length of the body after the head is cut off
OFFSET=$(echo "$MAIN - $FADE" | bc)   # where the tail starts fading into the head

LOOP="[0:v]fps=30000/1001,split=2[a][b];\
[a]trim=start=$FADE,setpts=PTS-STARTPTS[body];\
[b]trim=end=$FADE,setpts=PTS-STARTPTS[head];\
[body][head]xfade=transition=fade:duration=$FADE:offset=$OFFSET,format=yuv420p"

for H in 1080 720; do
  ffmpeg -v error -y -ss "$START" -t "$LEN" -i "$SRC" -an \
    -filter_complex "$LOOP,scale=-2:$H:flags=lanczos" \
    -c:v libx264 -preset slow -crf 26 -profile:v high -pix_fmt yuv420p -movflags +faststart \
    "$OUT/hero-$H.mp4"
done
ffmpeg -v error -y -i "$OUT/hero-1080.mp4" -frames:v 1 -q:v 6 "$OUT/hero-poster.jpg"
ls -la "$OUT"
