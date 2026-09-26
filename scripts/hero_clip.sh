#!/usr/bin/env bash
# Home-page background loop: a short stretch of the raw 4K sample, cut to a seamless loop (the last
# FADE seconds cross-fade into the first ones), H.264 without audio at 720p (desktop) and 480p (phones),
# plus a poster. It plays under a dark scrim, so full HD bought nothing visible: 720p at crf 30 is ~1.5 MB
# against 5.1 MB for the old 1080p encode, and the loop starts that much sooner.
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

for SPEC in "720 30" "480 31"; do
  read -r H CRF <<< "$SPEC"
  ffmpeg -v error -y -ss "$START" -t "$LEN" -i "$SRC" -an \
    -filter_complex "$LOOP,scale=-2:$H:flags=lanczos" \
    -c:v libx264 -preset slow -crf "$CRF" -profile:v high -pix_fmt yuv420p -movflags +faststart \
    "$OUT/hero-$H.mp4"
done
rm -f "$OUT/hero-1080.mp4"
ffmpeg -v error -y -ss "$START" -i "$SRC" -frames:v 1 -vf "scale=1280:-2:flags=lanczos" -q:v 8 "$OUT/hero-poster.jpg"
ls -la "$OUT"
