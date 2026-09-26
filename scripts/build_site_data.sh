#!/usr/bin/env bash
# Regenerate everything the website shows from predictions_samples.json, in order. Run it after the
# final Kaggle run (scripts/make_predictions_samples.py --unpaced) has put that file in the repo root.
# The home page's runtime comes from runs/kaggle_t4_official.json, the same machine's official 3x run
# (pacing on), when it exists: the deliverable's own log is from the relaxed unpaced run.
#
#   metrics.json   evaluate.py on the submission run vs labels/dev_gt.json (scripts/eval_dev.py --pred)
#   report.json    the report page, numbers from the files above (scripts/write_report.py)
#   eda/           charts and overlays from the track caches (scripts/eda.py)
#   results/       annotated samples, gallery clips, failure cases (render_samples.py, render_failures.py)
#   dashboard/     where the events happen (scripts/event_heat.py)
#
# Needs the sample videos (default samples/) and their track caches (cache/tracks/). Rendering the
# full-length samples takes about 20 minutes on a laptop CPU.
#
# Usage: bash scripts/build_site_data.sh [samples/]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
VIDEOS="${1:-samples}"
DATA=web/public/data

TIMING=()
[[ -f runs/kaggle_t4_official.json ]] && TIMING=(--timing runs/kaggle_t4_official.json)

"$PY" evaluate.py --pred predictions_samples.json --validate-only
cp predictions_samples.json "$DATA/predictions_samples.json"
"$PY" scripts/eval_dev.py --pred predictions_samples.json | tail -n 3
"$PY" scripts/write_report.py ${TIMING[@]+"${TIMING[@]}"}
"$PY" scripts/eda.py "$VIDEOS"
rm -rf "$DATA/results"  # clips of classes the new run no longer has must not linger
"$PY" scripts/render_samples.py "$VIDEOS" --predictions predictions_samples.json ${TIMING[@]+"${TIMING[@]}"}
"$PY" scripts/render_failures.py "$VIDEOS" --predictions predictions_samples.json
"$PY" scripts/event_heat.py --predictions predictions_samples.json
du -sh "$DATA"/results/*/ "$DATA"/eda
