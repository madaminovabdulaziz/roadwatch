#!/usr/bin/env bash
# Create (once per Kaggle session) a venv with requirements.txt installed by plain pip, the organizers'
# install path, plus the dev tools. Re-running only installs what changed.
#
# Usage from a notebook cell:  !bash /tmp/roadwatch/kaggle/setup_venv.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/tmp/rw-venv}"
PY="$VENV/bin/python"

# Kaggle injects its own paths (their sitecustomize imports wrapt); a clean machine has none.
# Kaggle's python3 also lacks ensurepip, so the venv falls back to virtualenv.
unset PYTHONPATH

if [[ ! -x "$PY" ]]; then
  python3 -m venv "$VENV" || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q "$VENV"; }
  "$PY" -m pip install -q --upgrade pip
fi
SECONDS=0
"$PY" -m pip install -q -r "$REPO/requirements.txt"
echo "requirements.txt ready in ${SECONDS}s ($VENV)"
"$PY" -m pip install -q -r "$REPO/requirements-dev.txt"
