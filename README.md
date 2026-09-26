# RoadWatch

Traffic event detection (Part A) and accident anticipation (Part B) for one fixed road camera. This is the
WIUT Hackathon 2026 computer-vision elimination task. `solution.py` is the organizers' interface; the logic
lives in the `roadwatch/` package.

> Status: the scene is calibrated and 8 of the 14 classes are enabled, each after it passed the enable
> policy on our labelled samples (`docs/SPEC.md` §7, decisions §12.52–§12.55): accident, red_light,
> wrong_way, jaywalking, failure_to_yield, solid_line_crossing, stop_line, road_obstacle. The official
> `evaluate.py` scores the submission's run on a T4 against our labels (2 videos, 39 events) in
> `web/public/data/metrics.json`; Part B raised 2 alarms in 7.4 min of accident-free traffic. On a Kaggle T4
> with 4 CPU cores the official run takes about 2.7x the video length (budget 3x). The others stay off
> because we could not show on real footage that they help: near_miss, stopped_vehicle, illegal_u_turn,
> congestion, illegal_turn, fire_smoke (not implemented).

## Run it (what the organizers run)

Requirements: Linux, Python 3.10–3.12, an NVIDIA GPU with driver ≥ 525 (tested on a T4). No internet is
needed after the install line: the weights are in the repository (`weights/`, checked against
`weights/SHA256SUMS`).

```bash
pip install -r requirements.txt          # pinned lock; torch/torchvision are the CUDA 12.6 builds
python run_submission.py --videos /data/test --out predictions.json --team roadwatch
python evaluate.py --pred predictions.json --validate-only
```

`bash weights/download.sh` re-fetches the same weights from the GitHub release `weights-v1` if the files
were lost (it keeps files whose checksum already matches).

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
One condition: the machine must be fast enough to finish inside the 3x budget without pacing. On a
slower machine the pipeline thins its work to stay inside the budget, so its output depends on the
machine's speed. A Kaggle T4 with 4 CPU cores runs both samples at 2.7x with pacing. The committed file was
written by `python scripts/make_predictions_samples.py --unpaced` (pacing off, and the harness's own
`--time-factor` relaxed so the slower machine is not cut off), which is the output a fast enough machine
produces.

## Develop

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-dev.txt
.venv/bin/python -m pytest              # tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

On Windows use `.venv\Scripts\python.exe`. The development loop, in order:

```bash
python scripts/cache_tracks.py samples/            # detect + track once per video (also the lamp timeline)
python scripts/calibrate_scene.py --video samples/C3902.MP4   # click configs/scene.json (browser tool)
python scripts/learn_lane_flow.py                  # lane directions from the tracks
python scripts/labels_to_gt.py                     # labels/raw/*.csv -> labels/dev_gt.json
python scripts/eval_dev.py                         # per-class F1 + every FP/FN with timestamps
python scripts/risk_replay.py                      # Part B's alarms on the cached tracks, with their causes
python scripts/tune.py --classes wrong_way         # small grid search on cached tracks
python scripts/render_samples.py samples/ --preview-all   # look at what each rule would emit
python scripts/check_determinism.py samples/       # two harness runs, identical output + x duration
python scripts/make_predictions_samples.py --unpaced   # the predictions_samples.json deliverable
bash scripts/build_site_data.sh samples/           # everything the website shows, from that file
```

`requirements.txt` is a lock generated from `requirements.in`
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

- One fixed camera; one hand-calibrated `configs/scene.json`, registered to each recording's framing with
  SIFT on its median background (SPEC §12.17, §12.34).
- Frame rate is read from each file (the samples are 29.97 fps, not 25); `t_sec = frame_idx / fps`.
- Metric scale comes from the frame's own perspective: the vanishing points of the lane lines and of the
  stop line fix the road plane, and the one assumed length is a 3.5 m lane width (SPEC §12.46).
- Classes whose rule needs scene geometry the camera does not show are disabled, not guessed (SPEC §12.7).
