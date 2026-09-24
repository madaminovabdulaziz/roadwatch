#!/usr/bin/env bash
# Clean-machine check on a Kaggle T4 notebook (RUNBOOK P0.4, SPEC §12.13 and §12.16).
#
#  1. fresh venv + plain `pip install -r requirements.txt`, exactly what the organizers run;
#  2. torch sees the T4 and runs FP16 on it;
#  3. the test suite passes on Linux;
#  4. the official harness on a synthetic clip in the sample format (4K H.264 High 4:2:2 10-bit,
#     140 Mbps, 29.97 fps). Our Part B does nothing yet, so its time is the harness's own decode cost.
#
# Usage from a notebook cell:  !bash /tmp/roadwatch/kaggle/check_env.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/tmp/rw-venv}"
WORK="${WORK:-/tmp/rw-check}"
PY="$VENV/bin/python"
cd "$REPO"

# Kaggle injects its own paths (their sitecustomize imports wrapt); a clean machine has none.
# Kaggle's python3 also lacks ensurepip, so the venv falls back to virtualenv below.
unset PYTHONPATH

echo "== machine"
python3 --version
echo "cpu cores: $(nproc)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free -g | head -2
df -h /tmp | tail -1

echo "== install: fresh venv, plain pip (organizers' path)"
python3 -m venv "$VENV" || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q "$VENV"; }
"$PY" -m pip install -q --upgrade pip
SECONDS=0
"$PY" -m pip install -q -r requirements.txt
echo "requirements.txt installed in ${SECONDS}s"
"$PY" -m pip install -q -r requirements-dev.txt

echo "== torch / CUDA"
"$PY" - <<'EOF'
import cv2
import numpy
import supervision
import torch

print("torch", torch.__version__, "| CUDA runtime", torch.version.cuda, "| available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("sm_75 kernels shipped:", "sm_75" in torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print("fp16 matmul on GPU ok:", bool(torch.isfinite((x @ x).float()).all()))
print("cv2", cv2.__version__, "| numpy", numpy.__version__, "| supervision", supervision.__version__)
EOF

echo "== tests"
"$PY" -m pytest -q

echo "== synthetic clip in the sample format"
mkdir -p "$WORK/clips" "$WORK/bin"
FFMPEG=ffmpeg
encode() {
  "$1" -v error -y -f lavfi -i "testsrc2=size=3840x2160:rate=30000/1001" -t 10 \
    -c:v libx264 -preset veryfast -profile:v high422 -pix_fmt yuv422p10le \
    -b:v 140M -maxrate 140M -bufsize 280M -g 15 -bf 2 -x264-params b-pyramid=none \
    "$WORK/clips/synthetic_4k422p10_10s.mp4"
}
if ! command -v ffmpeg >/dev/null || ! encode ffmpeg 2>/dev/null; then
  echo "system ffmpeg cannot encode 10-bit 4:2:2; using a static ffmpeg build"
  curl -sSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ -C "$WORK/bin" --strip-components=1
  FFMPEG="$WORK/bin/ffmpeg"
  encode "$FFMPEG"
fi
ls -lh "$WORK/clips"

echo "== official harness on the synthetic clip"
"$PY" run_submission.py --videos "$WORK/clips" --out "$WORK/predictions.json" --team roadwatch
"$PY" evaluate.py --pred "$WORK/predictions.json" --validate-only
"$PY" - "$WORK/predictions.json" <<'EOF'
import json
import sys

for video, log in json.load(open(sys.argv[1]))["log"].items():
    ratio = log["total_sec"] / log["duration"]
    print(f"{video}: harness total {log['total_sec']} s for {log['duration']} s of video "
          f"= {ratio:.2f}x duration (budget 3.00x; our Part B does no work yet)")
EOF
