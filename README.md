# RoadWatch

Traffic event detection (Part A) and accident anticipation (Part B) for one fixed road camera. This is the
WIUT Hackathon 2026 computer-vision elimination task. `solution.py` is the organizers' interface; the logic
lives in the `roadwatch/` package.

> Status: project skeleton (RUNBOOK P0.1). The submission runs end to end and returns valid, empty
> predictions. Perception, rules and risk land in the next RUNBOOK steps.

## Run it (what the organizers run)

```bash
pip install -r requirements.txt          # Python 3.10-3.12, Linux + NVIDIA driver >= 525
bash weights/download.sh                 # fetch weights once, before the offline run (P0.3)
python run_submission.py --videos /data/test --out predictions.json --team roadwatch
python evaluate.py --pred predictions.json --validate-only
```

## Develop

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-dev.txt
.venv/bin/python -m pytest              # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

`requirements.txt` is a lock generated from `requirements.in` (see the command at the top of that file).

## Where things are

- `docs/SPEC.md`: technical spec and resolved decisions (§12). `docs/WEBSITE_SPEC.md`: website and demo.
- `RUNBOOK.md`: build order. `CLAUDE.md`: working rules.
- `docs/STARTER_KIT.md` and `docs/task_description.md`: the organizers' starter-kit README and task text.
- `run_submission.py`, `evaluate.py`, `examples/`: the organizers' files, unmodified.

## Models, datasets and licences

To be completed as components land (P0.4): YOLO11m COCO weights (Ultralytics, AGPL-3.0), ByteTrack via
`supervision` (MIT).
