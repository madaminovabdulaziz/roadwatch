# RoadWatch

Traffic event detection (Part A) and accident anticipation (Part B) for one fixed road camera. This is the
WIUT Hackathon 2026 computer-vision elimination task. `solution.py` is the organizers' interface; the logic
lives in the `roadwatch/` package.

> Status: packaging and perception are in place (RUNBOOK P0.1–P0.4). The submission runs end to end and
> returns valid predictions; all 14 event classes are still disabled in `configs/thresholds.yaml` until they
> pass the enable policy (`docs/SPEC.md` §7), so the event list is empty for now.

## Run it (what the organizers run)

Requirements: Linux, Python 3.10–3.12, an NVIDIA GPU with driver ≥ 525 (tested on a T4). No internet is
needed after the two setup lines.

```bash
pip install -r requirements.txt          # pinned lock; torch/torchvision are the CUDA 12.6 builds
bash weights/download.sh                 # fetch weights once, before the offline run (checksummed)
python run_submission.py --videos /data/test --out predictions.json --team roadwatch
python evaluate.py --pred predictions.json --validate-only
```

While the repository is private, `weights/download.sh` needs `GITHUB_TOKEN` (a token with read access to
the repository) in the environment.

### With Docker

```bash
bash weights/download.sh                 # or pass the token as a build secret, see Dockerfile
docker build -t roadwatch .
docker run --rm --gpus all --network none \
  -v /data/test:/data/videos:ro -v "$PWD/out":/data/out roadwatch
```

The container runs `run_submission.py` on `/data/videos`, writes `/data/out/predictions.json`, then runs
`evaluate.py --validate-only` on it. Dependencies and weights are baked into the image at build time, so
`--network none` works. Override `VIDEOS`, `OUT` or `TEAM` with `-e`.

### Reproduce `predictions_samples.json`

```bash
python run_submission.py --videos samples --out predictions_samples.json --team roadwatch
```

Two runs give the same file: seeds are fixed, cuDNN is deterministic and TF32 is off (`roadwatch/config.py`).

## Develop

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-dev.txt
.venv/bin/python -m pytest              # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

On Windows use `.venv\Scripts\python.exe`. `requirements.txt` is a lock generated from `requirements.in`
(see the command at the top of that file). Running on a Kaggle T4: `kaggle/README.md`.

## Where things are

- `docs/SPEC.md`: technical spec and resolved decisions (§12). `docs/WEBSITE_SPEC.md`: website and demo.
- `RUNBOOK.md`: build order. `CLAUDE.md`: working rules.
- `docs/STARTER_KIT.md` and `docs/task_description.md`: the organizers' starter-kit README and task text.
- `run_submission.py`, `evaluate.py`, `examples/`: the organizers' files, unmodified.

## Models and licences

| Model | Use | Source | Licence |
|---|---|---|---|
| YOLO11m (COCO-pretrained) | object detection, Part A and Part B | Ultralytics, exported to a raw-head TorchScript file by `scripts/fetch_weights.py` (SPEC §12.19) | AGPL-3.0 |

The weights are published as the GitHub release `weights-v1` of this repository and checked against
`weights/SHA256SUMS`. `scripts/fetch_weights.py` rebuilds them from the official Ultralytics checkpoint.
The `ultralytics` package is used only for that export, never at inference.

## Datasets and licences

We train nothing yet. The detector's weights were trained by Ultralytics on
[COCO](https://cocodataset.org) (annotations CC BY 4.0; images under their Flickr licences). Our own
development labels for the sample videos live in `labels/`.

## Attribution of reused code

- **ByteTrack** multi-object tracker, from [`supervision`](https://github.com/roboflow/supervision) 0.30.5
  (MIT), imported from `supervision.tracker.byte_tracker.core` (SPEC §12.21).
- **YOLO11** network architecture and weights: [Ultralytics](https://github.com/ultralytics/ultralytics)
  (AGPL-3.0). Box decoding and NMS are our own code on top of `torchvision.ops.batched_nms`.
- Video decoding: [PyAV](https://github.com/PyAV-Org/PyAV) (BSD-3-Clause) over FFmpeg (LGPL-2.1+), and
  OpenCV (Apache-2.0). Tensors: PyTorch and torchvision (BSD-3-Clause).

## Assumptions

- One fixed camera on a tripod; one hand-calibrated `configs/scene.json` serves every video (SPEC §12.17).
- Frame rate is read from each file (the samples are 29.97 fps, not 25); `t_sec = frame_idx / fps`.
- Metric scale for speeds comes from a homography on lane markings, assuming a lane width of 3.5 m and
  urban dashed markings of about 3 m line + 9 m gap (SPEC §4).
- Classes whose rule needs scene geometry the camera does not show are disabled, not guessed (SPEC §12.7).
