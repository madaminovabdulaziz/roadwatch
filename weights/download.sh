#!/usr/bin/env bash
# Fetch model weights into weights/ once, before the offline run (organizers' starter-kit convention).
# The release archive and its checksum are added in RUNBOOK P0.3/P0.4; until then there is nothing to fetch.
set -euo pipefail
echo "weights/download.sh: no weights archive published yet (RUNBOOK P0.3)" >&2
exit 1
