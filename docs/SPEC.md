# RoadWatch — Technical Spec (source of truth)

Plain rules. If code and this doc disagree, fix one of them the same day.

---

## 1. Scoring facts that drive design

- **Score A** = mean over classes C of (mean over IoU τ ∈ {0.3, 0.5, 0.7} of F1). C = classes in the test set **plus any class we predict**. A single false prediction of an absent class adds a class with F1 = 0. → Every class has an `enabled` flag and a conservative threshold.
- TP/FP/FN are pooled over all test videos, then F1 per class.
- Matching is greedy by temporal IoU. Boundaries must follow the annotators' start/end rules (§5). IoU 0.7 on a 5 s event means ~±0.7 s total error. → Boundary refinement at full frame rate (§6).
- Two same-class events at the same time = ONE segment (union).
- **Score B** = 0.4·AP (chance-normalised) + 0.4·F1_alarm + 0.2·(mTTA/10). Alarm = run of score ≥ 0.5 (runs < 2 s apart merge). Alarm matches an accident if it starts in [s−10, s). Frames inside accidents and within [s−5, e] of near-misses are ignored; alarms starting there are discarded (free).
- Model M = 0.7·A + 0.3·B (**M = A if the test set has no accidents**, evaluate.py). Elimination = 0.6·M + 0.25·Website + 0.15·Code.
- Hardware: 1× T4 16 GB, 8 CPU, 32 GB RAM, no internet, ≤ 3× video duration for A+B, ≤ 5 GB weights, Python ≥ 3.10.
- The 3× budget also pays for the harness decoding every 4K frame for Part B (§12.9–12.11). That decode alone measured 1.56× duration locally.

## 2. Repository layout

```
solution.py                 # thin: imports roadwatch, exposes CLASSES, detect_events, RiskEstimator
run_submission.py           # ORGANIZERS — do not edit
evaluate.py                 # ORGANIZERS — do not edit
examples/                   # ORGANIZERS — do not edit
roadwatch/
  __init__.py
  config.py                 # loads configs/thresholds.yaml + scene.json, seeds, device
  types.py                  # dataclasses: VideoMeta, FrameDetections, Segment; TrackTable columns
  video.py                  # FrameReader (sequential, stride), read_window(t0, t1, stride=1)
  perception/
    detector.py             # Detector: load once, predict(batch of frames) -> list[FrameDetections]
    tracker.py              # OnlineTracker wrapping supervision.ByteTrack
    run.py                  # run_perception(video_path, stride) -> TrackTable
  scene/
    scene.py                # Scene: polygons, lines, lanes, homography, helpers
    light.py                # SignalStateEstimator (HSV lamps, or flow fallback)
    calibration.py          # reference median, tool shapes <-> scene.json, validate_scene (§12.26)
    overlay.py              # draw_scene: scene layers on an image
    flow.py                 # image-space step velocities, lane directions from tracks, flow field (§12.28)
  features.py               # add_kinematics(TrackTable, scene) ; pair_ttc(...)
  events/
    base.py                 # EventRule protocol, VideoContext, is_runnable (registry: events/__init__.py)
    accident.py near_miss.py red_light.py wrong_way.py illegal_u_turn.py
    stopped_vehicle.py jaywalking.py failure_to_yield.py illegal_turn.py
    solid_line_crossing.py stop_line.py congestion.py road_obstacle.py fire_smoke.py
  postprocess.py
  pipeline.py               # detect_events orchestration + global model cache
  risk.py                   # RiskCore used by solution.RiskEstimator
  render.py                 # annotated video + overlays
configs/
  scene.json                # hand calibrated (see §4)
  thresholds.yaml           # every tunable + enabled flags + postprocess params
scripts/
  fetch_weights.py  cache_tracks.py  calibrate_scene.py (+ .html)  render_scene.py
  learn_lane_flow.py  plot_signal_timeline.py
  labels_to_gt.py   eval_dev.py      tune.py             bench.py
  render_samples.py eda.py           make_predictions_samples.py  check_determinism.py
labels/dev_gt.json          # our own annotations of samples, organizer GT format
cache/                      # gitignored: tracks parquet, backgrounds
weights/                    # gitignored except download.sh; weights ship as a release archive (§12.14)
tests/
demo/                       # FastAPI backend (see WEBSITE_SPEC)
web/                        # frontend (see WEBSITE_SPEC)
docs/STARTER_KIT.md  docs/task_description.md   # organizers' README and task text, for reference
Dockerfile  requirements.in  requirements.txt  requirements-dev.txt  requirements-export.txt  pyproject.toml
kaggle/                     # T4 notebook scripts: venv setup, sample download, clean-machine check
README.md  predictions_samples.json
```

## 3. Perception

**Detector.** Default: YOLO11m (Ultralytics), COCO weights, run without the `ultralytics` package (§12.12, §12.19), FP16, `imgsz=960`, detections from `conf_min=0.1` go to the tracker (§12.20), classes kept: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7), plus obstacle candidates: bird(14), cat(15), dog(16), horse(17), sheep(18), cow(19), backpack(24), suitcase(28). If bench shows > 0.6× duration for Part A alone, drop to YOLO11s. Licence: AGPL-3.0 (list in README). Load weights from `weights/` by local path only. Never call `.plot()`.

**Batching.** Predict in batches of 8–16 frames. Warm up once at load.

**Tracker.** ByteTrack from `supervision` (MIT, §12.21), one tracker per class group (vehicles, persons, two-wheelers, others) so IDs don't jump across types; NMS runs per group too (§12.20). `frame_rate = fps / stride`. Fixed camera → no motion compensation. Each track gets one class: the confidence-weighted majority of its detections.

**Sampling.** Part A: stride 3 (~10 fps at 29.97 fps), B-frames never decoded (§12.18). Part B: stride 3. Boundary refinement: stride 1 inside small windows.

**TrackTable** (pandas DataFrame, one row per track per processed frame):
`frame, t, track_id, cls, conf, x1, y1, x2, y2, fx, fy` where `(fx, fy)` = footprint = bottom-centre of box in pixels.
After `features.add_kinematics`: `X, Y` (metres), `vx, vy, speed, accel, heading_deg, yaw_rate, lane_id, on_road, in_crosswalk, in_intersection, in_no_uturn, zone ids`.

**Kinematics.** Footprint → metres via homography. Part A smooths with Savitzky–Golay (window ≈ 1.2 s, order 2); Part B uses causal EMA / constant-velocity Kalman. Speeds are unreliable for tracks shorter than 1 s → mark `kin_valid=False`.

**Position persistence.** Stationary objects cause ID switches. For stopped_vehicle / road_obstacle, a new track that appears within 2.0 m of where a stationary track vanished less than 3 s earlier is treated as the same object.

## 4. Scene config (`configs/scene.json`)

Hand-calibrated once on a reference frame with `scripts/calibrate_scene.py` (OpenCV click tool: pick a layer, click points, Enter to close, S to save). All coords in pixels of the native resolution.

```json
{
  "image_size": [1920, 1080],
  "reference_frame": "configs/reference.jpg",
  "homography": {"image_pts": [[x,y],[x,y],[x,y],[x,y]], "world_pts": [[0,0],[3.5,0],[3.5,20],[0,20]]},
  "carriageway": [[x,y], ...],
  "sidewalks":   [[[x,y], ...]],
  "parking_zones": [[[x,y], ...]],
  "intersection": [[x,y], ...],
  "directions": [{"id": "NB", "lanes": ["n1","n2"]}, {"id": "SB", "lanes": ["s1","s2"]}],
  "lanes": [
    {"id": "n1", "polygon": [[x,y],...], "direction": [dx,dy], "approach": "S", "signal": "L1",
     "allowed_exits": ["N_out","E_out"]}
  ],
  "exits": [{"id": "N_out", "polygon": [[x,y],...]}],
  "stop_lines": [{"id": "sl_S", "line": [[x,y],[x,y]], "lanes": ["n1","n2"], "signal": "L1"}],
  "crosswalks": [{"id": "cw_S", "polygon": [[x,y],...]}],
  "solid_lines": [{"id": "sol1", "polyline": [[x,y],...]}],
  "signals": [{"id": "L1", "red": [x,y,w,h], "yellow": [x,y,w,h], "green": [x,y,w,h]}],
  "no_u_turn_zones": [[[x,y], ...]],
  "u_turn_prohibited_everywhere": false
}
```

Rules:
- `direction` is an image-space unit vector; `scripts/learn_lane_flow.py` proposes it from sample tracks (median velocity per lane); a human confirms.
- `homography`: 4 points on lane markings with known real spacing (lane width 3.5 m, dashed line + gap ≈ 3 m + 9 m urban; state your assumption in README).
- If a layer is missing (e.g. no visible signal), rules that need it disable themselves and log it.

## 5. Event rules (one module each)

Common signature:
```python
def detect(tt: pd.DataFrame, scene: Scene, ctx: VideoContext, cfg: dict) -> list[Segment]
# Segment(start: float, end: float, label: str, score: float, track_ids: tuple, meta: dict)
```
`ctx` holds fps, duration, stride, signal-state timeline, background model, frame reader for refinement.
Thresholds below are defaults in `thresholds.yaml`. `score` ∈ [0,1] is used for the enable/threshold policy (§7).

| Class | Trigger | Start | End |
|---|---|---|---|
| **stopped_vehicle** | vehicle speed < 0.5 m/s continuously ≥ 10 s, footprint on carriageway, not in parking zone, NOT queued: (a) not within 30 m upstream of a stop line whose signal is red/yellow, and (b) no stopped vehicle ahead in same lane within 12 m | first time speed dropped < 0.5 m/s (backdated) | speed > 1.0 m/s for ≥ 1 s, or track gone (after persistence merge); end of video → duration |
| **wrong_way** | speed > 2 m/s, cos(velocity, lane direction) < −0.5 for ≥ 1.0 s | time footprint entered the lane where it is wrong (walk back) | leaves that lane to a correct one, or leaves frame |
| **jaywalking** | person footprint on carriageway, outside crosswalks & sidewalks, ≥ 1.0 s. Exclude riders: person box centre inside a 2-wheeler box expanded 20%, or IoU > 0.3 with a 2-wheeler. Exclude persons whose box is mostly inside a vehicle box | steps on road | leaves road. Union across people |
| **red_light** | signal of the vehicle's lane is RED (stable ≥ 0.5 s), vehicle front point crosses its stop line in lane direction, and crossing happens ≥ 0.3 s after red onset | crossing time (interpolate between frames) | footprint leaves intersection polygon, or leaves frame |
| **stop_line** | signal RED, vehicle stopped (speed < 0.5 m/s ≥ 1 s) with front point past stop line by > 0.3 m, footprint NOT in intersection | time it stopped | signal turns GREEN (union overlapping vehicles) |
| **failure_to_yield** | vehicle speed > 1.5 m/s with footprint in crosswalk polygon while ≥ 1 person footprint is in the same crosswalk or within 1 m of its edge moving toward it | vehicle enters crosswalk | vehicle leaves crosswalk |
| **solid_line_crossing** | both bottom box corners start on one side of a solid polyline and end on the other side; lateral move ≥ 1 m; not inside intersection | first bottom corner crosses | second corner crosses |
| **illegal_u_turn** | cumulative heading change ≥ 150° within 12 s, speed > 1 m/s at start, turn starts in a no-U-turn zone (or `u_turn_prohibited_everywhere`) | yaw rate first > 10°/s | heading stable (±15°) for 1 s |
| **illegal_turn** | movement (entry lane at stop line → exit polygon) not in `allowed_exits`; not a U-turn | leaves entry lane / yaw rate > 10°/s | enters exit polygon with stable heading |
| **congestion** | per direction, every 1 s: mean speed < 1.5 m/s AND ≥ 2 vehicles per lane in all lanes; sustained ≥ 20 s; and either lasts ≥ 60 s or persists ≥ 10 s into a GREEN phase (so normal red queues don't count) | queue stopped moving (backdated to when condition began) | mean speed > 3 m/s sustained 5 s |
| **accident** | pair (vehicle–vehicle, vehicle–person, vehicle–2-wheeler): footprints < 1.5 m apart AND boxes overlap, AND ≥ 1 kinematic shock within ±0.6 s: decel > 6 m/s², or heading change > 30° in < 1 s, or a person box aspect flips (falls). Plus confirmation: involved objects stay stopped ≥ 3 s outside any queue, or leave. Single-vehicle: speed > 8 m/s → < 1 m/s in < 1.5 s, off-lane or near road edge | first contact frame (refined at stride 1) | all involved speed < 0.3 m/s for 2 s, or leave frame |
| **near_miss** | pair with TTC < 1.5 s and closing speed > 3 m/s, then evasive action (decel > 4 m/s² or yaw rate > 25°/s), min separation stays > 1.5 m, no accident within 3 s | onset of evasive action (decel first > 2 m/s²) | separation increasing and TTC > 3 s |
| **road_obstacle** | (a) COCO animal/bag class on carriageway ≥ 1 s, or (b) static foreground blob (background = running median over 30 s) on carriageway, area ≥ 0.3 m², not overlapping any vehicle/person box (+20%), static ≥ 3 s | first appearance | disappears for ≥ 2 s |
| **fire_smoke** | **disabled by default.** Enable only if an open fire/smoke model is shipped, gives 0 detections on all samples, and fires on public fire clips | first detection | last detection + 1 s |

Occlusion is the #1 accident false-positive source in perspective views: overlapping boxes alone are NEVER enough. Always require a kinematic shock plus confirmation.

## 6. Post-processing (`postprocess.py`, in this order)

1. Drop segments with `score < cls.min_score`.
2. Per class, merge segments with gap < `merge_gap` (default 1.0 s; congestion 5 s; stopped_vehicle 2 s).
3. Drop segments shorter than `min_dur` (accident 0.8 s, jaywalking 1.0 s, red_light 0.5 s, stopped_vehicle 10 s, congestion 20 s, others 0.5 s).
4. **Boundary refinement**: for classes with contact/crossing boundaries (accident, red_light, solid_line_crossing, failure_to_yield, near_miss), re-read ±2 s around start and end at stride 1, re-run detection on those frames only, recompute the exact crossing/contact frame. Budget: skip if total refinement frames > 15% of video frames.
5. Same-class union (overlaps → one segment).
6. Clip to [0, duration]; if the event is ongoing at the last frame, end = duration. Drop start ≥ end. Cast to float, round to 3 decimals.

## 7. Class enable policy

- A class is emitted only if `enabled: true` in `thresholds.yaml` AND the scene has the geometry it needs.
- Dev-validated classes (present in our sample labels): enable if dev mean-F1 ≥ 0.25; tune thresholds on dev.
- Classes absent from samples: enable if the rule is pure geometry and passes its synthetic tests (wrong_way, red_light, stop_line, failure_to_yield, solid_line_crossing, illegal_u_turn, illegal_turn), with `min_score` high, and zero false positives on all sample videos.
- road_obstacle: enabled only with zero FPs on samples. fire_smoke: see §5.
- Final check before freeze: run on all samples; every emitted event of a class with no dev labels must be manually reviewed. If any is wrong, raise its threshold or disable the class.

## 8. Part B — Risk (`risk.py`)

`RiskEstimator` (in solution.py) owns a `RiskCore` with its own Detector (shared global weights object is fine) and its own OnlineTracker. Processes every 3rd frame; returns the last score on skipped frames. Never opens the video file, never touches `cache/`.

Features per processed frame (causal, EMA-smoothed kinematics):
- `ttc_min`: min over converging pairs of distance / closing speed (only pairs < 25 m apart, closing speed > 1 m/s)
- `closing_speed` of that pair
- `decel_max`: largest current deceleration among vehicles (m/s²)
- `wrong_way_active`, `red_light_active` (from causal signal state), `ped_on_road_with_vehicle_ttc` (min TTC vehicle→person on carriageway)

Score:
```
z = b + w1·g(ttc_min) + w2·clip(closing_speed/10) + w3·clip(decel_max/8)
      + w4·wrong_way_active + w5·red_light_active + w6·g(ped_ttc)
g(ttc) = exp(-ttc / 1.5)   (0 if none)
raw = sigmoid(z) ; score = EMA(raw, alpha=0.4) ; clamp to [0,1]
```
Start weights (hand-set, then tune on dev + public CCTV crash clips): b=−5.0, w1=5.0, w2=1.5, w3=2.0, w4=1.0, w5=0.8, w6=3.0. Targets: calm traffic < 0.05; TTC 1.0 s at 6 m/s closing → ~0.6; TTC 3 s → ~0.15.
Hysteresis: once ≥ 0.5, hold for ≥ 1 s (fewer fragmented alarms).
Always return a finite float; on any exception return the last score.

## 9. Testing

- `tests/test_geometry.py`: point-in-polygon, line crossing, homography round-trip.
- `tests/test_events_*.py`: synthetic TrackTables per class (a car driving backward along a lane → wrong_way with exact start/end; a car crossing stop line during red → red_light; etc.).
- `tests/test_postprocess.py`: merge, min-dur, union, clipping, floats.
- `tests/test_interface.py`: `detect_events` on a 10 s clip returns valid list; `RiskEstimator` returns floats in [0,1] for every frame.
- `tests/test_offline.py`: import solution and run with the network blocked (monkeypatch socket).
- Time-reversal trick: reversing a sample clip makes every vehicle wrong-way → sanity check for wrong_way.

## 10. Dev set and tuning

- Label every sample video with the §5 conventions in Label Studio (video timeline labels) or by hand in a CSV (`video,start,end,label`). `scripts/labels_to_gt.py` → `labels/dev_gt.json` in organizer format with `duration` and `fps`.
- `scripts/eval_dev.py`: runs pipeline from cached tracks, writes `predictions_dev.json`, calls `evaluate.py --gt labels/dev_gt.json`, prints per-class table.
- `scripts/tune.py`: small grid search over a few thresholds per class on cached tracks. Prefer round numbers; do not overfit tiny dev sets.

## 11. Runtime, determinism, packaging

- Models loaded once at module level (lazy global). Warm-up at load.
- `scripts/bench.py` reports per-video: decode, detect, track, rules, refine, Part B, total / duration. Target ≤ 1.0×.
- Safety: if Part A elapsed > 1.2× duration, skip refinement; if detection throughput is low, increase stride to 5.
- Determinism: seeds (random, numpy, torch), cudnn deterministic, TF32 off, sorted iteration everywhere (dict/set order), stable sort by (t, track_id). `scripts/check_determinism.py` runs twice and diffs.
- Packaging: `requirements.txt` with exact pinned versions verified on a Kaggle T4; `Dockerfile` (CUDA runtime base) that runs the same two commands; weights in `weights/` (Git LFS) or a single release archive + `scripts/fetch_weights.py`. Test with `docker run --network none`.
- README: setup, the two commands, datasets + licences, models + licences, attribution of reused code, assumptions (lane width, etc.), how to reproduce `predictions_samples.json`.

## 12. Decisions (ambiguities resolved — do not re-debate)

1. Normal red-signal queues are NOT congestion (rule in §5).
2. t_sec = frame_idx / fps. Duration = n_frames / fps.
3. Part B never reuses Part A data, even causally computed. It recomputes.
4. ~~Front point of a vehicle = footprint shifted by half box height along the lane direction.~~ **Superseded by §12.35**: that rule put the front 2.3–5.0 m ahead of the real bumper for vehicles driving toward the camera.
5. Motorcycle/bicycle riders are never jaywalkers.
6. Pedestrians on sidewalks/islands are never jaywalkers; if no sidewalk polygon exists, only carriageway counts.
7. When a rule needs missing scene geometry, it is disabled, not guessed.
8. Unknown camera (demo uploads): detected via reference-frame similarity (§ WEBSITE_SPEC); only camera-independent classes run.
9. **Harness facts (run_submission.py, verified line by line).** CLI: `python run_submission.py --videos <dir or .mp4> --out predictions.json --team <name>`; dev-only flags `--solution`, `--no-risk`, `--risk-stride N`, `--time-factor F`. evaluate.py: `--pred`, `--gt`, `--json`, `--per-video`, `--validate-only`. Per video, in order: `video_meta` (OpenCV `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`; duration = n_frames / fps) → budget clock starts → `detect_events(path)` → a **new** `RiskEstimator()` → `reset(meta)` → the harness decodes **every** frame with `cv2.VideoCapture.read()` (full-res BGR) and calls `step(frame, idx / fps)` → clock stops. It filters `CLASSES` through `evaluate.OFFICIAL_CLASSES` when evaluate.py is importable.
10. **Budget enforcement.** Budget = 3 × duration, measured from before `detect_events` to the end of the Part B loop, so it includes the harness's own decode. If Part A alone exceeds it, Part B is skipped. Over budget → the **whole video** scores empty (Part A events too). The Part B deadline is checked every 100 frames. Model loading and warm-up count against the first video's budget.
11. **Crash semantics.** An exception in `detect_events` → no events for that video. An exception escaping `step()` → the **whole risk curve** of that video is lost. Hence solution.py contains every failure (§2) and imports `roadwatch` lazily: an import error at module load would fail every video. The harness clips `end` to duration, rounds events to 3 decimals and risk to 4, and drops same-class overlaps (keeping the earlier start). evaluate.py accepts `end ≤ duration + 0.5`.
12. **No `ultralytics` package at inference.** It depends on non-headless `opencv-python` (needs libGL, clashes with the harness's `opencv-python-headless`). YOLO11m weights are exported once in a separate dev environment and run on plain torch with our own letterbox and NMS (torchvision). Export format (TorchScript or torch.export) is decided in RUNBOOK P0.3. `supervision` (ByteTrack) no longer depends on OpenCV and stays.
13. **Dependencies.** `requirements.in` lists direct deps; `requirements.txt` is the full lock (`uv pip compile --universal --python-version 3.10 --emit-index-url`). Torch 2.8.0 / torchvision 0.23.0 use the **CUDA 12.6** builds on Linux (driver ≥ 525, T4 sm_75 supported). The default PyPI builds of newer torch need CUDA 13 / driver ≥ 580, which is too risky on an unknown machine. Every pin supports Python 3.10–3.12 (cu126 wheels exist for cp310–cp312 only). Dev uses Python 3.11.
14. **Weights delivery.** The organizers' README expects `weights/download.sh`, run once before the offline run, for anything not in the repository. We ship weights as a release archive fetched by that script (built in P0.3/P0.4).
15. **Sample video format.** The samples come from a Sony a6700 (not CCTV): 3840×2160, H.264 High 4:2:2 10-bit (`yuv422p10le`), ~140 Mbps, **29.97 fps (30000/1001), not 25**. A keyframe every 15 frames and 2 non-reference B-frames between reference frames. `moov` sits at the end of the file. Never assume 25 fps: read fps from the file. The T4's NVDEC cannot decode 4:2:2 H.264, so decoding is CPU-only and is the largest runtime cost. The test set has the same resolution and frame rate. One sample is at dusk.
16. **Measured decode cost.** On the harness path (OpenCV `read()` of every 4K frame, our Part B doing nothing), a 10 s synthetic clip in the sample format took 15.6 s: **1.56× duration** on an 8-core Apple M1. That leaves about 1.4× for Part A plus our `step()` work. On a Kaggle T4 notebook (4 CPU cores, Python 3.12, driver 580) the same run took 13.2 s = **1.32× duration** (2026-09-24). The organizers' machine has 8 cores, so we size our own work against the Kaggle number: ≤ 1.0× on Kaggle for Part A + our Part B compute, leaving margin. Part A must not decode frames it will not use (skip non-reference frames and/or decode in parallel chunks at keyframes). Part A also gets its own deadline (`runtime.part_a_deadline_factor`).
17. **Scene assumptions to verify.** The camera sits on a tripod. Compare the first frame of every sample with `configs/reference.jpg` before trusting one `scene.json` for all videos. If the framing moved, register each video to the reference frame instead of re-clicking the scene.
18. **Part A decoding.** PyAV with FFmpeg frame threading. Non-reference B-frames are never decoded (`skip_frame=NONREF`; in the samples the I/P frames are exactly every 3rd frame, idx 2, 5, 8, …), and frames are converted straight to the detector's input size in the same swscale pass. A background thread decodes ahead of the GPU (`video.prefetch`). The frame index comes from each frame's pts and matched the harness's OpenCV sequential index on real footage (84/84 frames, `bench.py --verify`). On the M1, the first 6 s of C3902 took 0.58× to decode this way, against 0.99× for the harness's full decode. Parallel chunk decoding (`bench.py --workers`) is measured but not used yet.
19. **Detector weights = raw-head TorchScript.** `scripts/fetch_weights.py` (separate export venv, `requirements-export.txt`) fuses YOLO11m and traces it *without* box decoding. roadwatch decodes (anchor grid, DFL, xyxy) and runs NMS itself. The traced graph has no device/dtype/shape constants, so one file runs on CPU/CUDA, FP32/FP16, any batch and input size (verified bit-exact vs eager at 1×960, 8×960, 2×1280, 1×640). Parity with Ultralytics' own `predict` on 6 real frames: 294/294 boxes matched, min IoU 0.99998, max confidence difference 5e-6. Weights are published as GitHub release `weights-v1`; `weights/download.sh` fetches them and checks `weights/SHA256SUMS`.
20. **Detection threshold and NMS.** `conf_min=0.1` supersedes §3's 0.25. ByteTrack starts tracks only from boxes above `track_activation_threshold + 0.1 = 0.35` and uses the 0.1–0.25 boxes only to continue existing tracks through partial occlusion (ByteTrack's core idea). NMS is per tracker group, not per class: a car also scored as a truck is one object, while a rider and their motorcycle stay two.
21. **ByteTrack source.** `supervision`'s top-level `sv.ByteTrack` has been deprecated since 0.28 and is removed in 0.31. We import the class from `supervision.tracker.byte_tracker.core` with supervision pinned at 0.30.5 and filter only that deprecation warning. If we ever need to change the tracker, vendor that module (MIT) with attribution. supervision counts `lost_track_buffer` in frames at 30 fps, so seconds are converted with that factor.
22. **FFmpeg teardown deadlock.** Closing a frame-threaded PyAV decoder that still held frames (a read stopped early: window end, `t_end`, deadline) hung forever. It reproduced 3/3 in the test suite on macOS. Native stacks (`sample`) showed the main thread in `Stream` dealloc → libavcodec thread teardown → `pthread_cond_wait`, with all 9 `av:h264:df*` workers idle. Every PyAV read now ends by flushing the decoder (send end-of-stream, drain) before the container closes: 0/5 hangs since, and there is a regression test.
23. **Perception time safety.** `run_perception(budget_sec=…)` paces itself: while wall time is ahead of `budget × progress`, it detects only every k-th decoded frame (k ≤ `runtime.max_frame_skip`). `deadline` stops it and returns partial tracks. Both engage only on a machine that is too slow, so normal runs stay deterministic.
24. **Detector input size.** On a C3902 frame, `imgsz=1280` finds more small, far pedestrians (33 vs 19 persons at conf ≥ 0.25, mostly the far bus-stop crowd) for ~1.8× compute. Road users near the crossings are all found at 960. Keep 960 unless the T4 bench shows room (`bench.py --perception --imgsz 1280`). At 1280, COCO `backpack` fires on bags worn by pedestrians, so `road_obstacle` must ignore bag detections that overlap a person.
25. **Docker packaging.** `Dockerfile` in the root (the task FAQ accepts it in place of `requirements.txt`): `nvidia/cuda:12.6.3-base-ubuntu22.04` + Ubuntu's Python 3.10 + plain `pip install -r requirements.txt`. The "base" image is enough because the cu126 torch wheels bundle cuBLAS/cuDNN, and a "runtime" image would ship them twice. Weights are baked in at build time: `weights/download.sh` keeps files already present with the right checksum and fetches the rest (a BuildKit secret carries the token while the repo is private), so `docker run --network none` needs nothing. `--target app` builds a weightless image for packaging tests only. Run instructions stay in `kaggle/README.md` (Cell 5) instead of a separate `kaggle/run_on_kaggle.md`. `.gitattributes` forces LF on `*.sh` and on the files they read (`weights/SHA256SUMS`, `kaggle/samples.tsv`): on Windows checkouts with `core.autocrlf` they otherwise become CRLF, bash rejects the scripts, and a trailing `\r` ends up in weight file names and URLs (seen as `curl: (3) URL using bad/illegal format` in the Docker build).
26. **Calibration tool is a local web page, not cv2.imshow.** The runtime pins `opencv-python-headless` (§12.12), which has no GUI, and installing `opencv-python` next to it breaks `cv2`. `scripts/calibrate_scene.py` serves `scripts/calibrate_scene.html` on 127.0.0.1 with the same controls RUNBOOK P1.1 asks for (layer keys, Enter, U, S) plus zoom/pan for 4K. Shapes ↔ scene.json and consistency checks live in `roadwatch/scene/calibration.py`; `roadwatch/scene/overlay.py` draws the layers (`scripts/render_scene.py`, later `render.py`). `configs/reference.jpg` = per-pixel median of 49 frames spread over one sample (odd count, so each pixel is an observed value). Lane direction is an arrow drawn per lane (tail → head) and stored as a unit vector; `learn_lane_flow.py` (P1.2) proposes it from tracks. Signals are three lamp boxes sharing one id. A test validates the committed scene.json whenever it exists.
27. **Signal state.** Lamp score = mean V of the lamp box + mean V of its lamp-coloured pixels (`signal.hue_ranges`), normalised per lamp by its own off/on levels (offline: 5th/95th percentile over the video; online: running min/max), so a dim LED and an overexposed one both work. Brightest normalised lamp past `lit_threshold` wins; majority filter over `median_sec`. Consequence: online (Part B) reports `unknown` until each lamp has been seen on and off once (one cycle, typically 1–2 min); Part B's `red_light_active` is therefore 0 at the start of a video. Frames of any size work (boxes scale from `image_size`). Without lamp boxes, `flow_timeline` infers green/red from stop-line crossings and queues; it is low confidence, so `red_light` never runs on it (it needs the `signals` layer), while `stop_line` and `congestion` may.
28. **Lane directions from tracks.** `learn_lane_flow.py` proposes each lane's image direction as the mean unit vector of moving vehicle steps inside it (`lane_flow` thresholds in native px); it writes only lanes with ≥ `min_samples` steps and consistency ≥ `min_consistency`, and only after a terminal "y".
29. **Dev loop seam.** `pipeline.prepare` (kinematics) → `run_rules` (each rule in try/except; `force` runs disabled classes for tuning) → `postprocess` → `to_events` is the one path for Part A, cached tracks (`eval_dev.py`, `tune.py`) and the harness alike. `detect_events` skips perception entirely while no class is enabled and runnable, so the empty submission costs no GPU time. Post-processing steps 1–3, 5, 6 are in; step 4 (boundary refinement) and the lamp-based signal timeline need frame reads and come with RUNBOOK P2.1. `to_events` drops disabled classes as a last guard. Dev labels: `labels/raw/*.csv` (`labels/README.md`); a `none` row marks a watched video with no events; touching same-class labels are unioned.
30. **Demo backend.** `demo/` (FastAPI, one worker thread, in-memory jobs, 1 h expiry) runs the same pipeline on CPU with `demo/config.yaml` deep-merged over thresholds.yaml through `ROADWATCH_OVERRIDES` (set only by `demo/Dockerfile`; never by the submission). Only YOLO11m is released, so the CPU profile is YOLO11m at imgsz 640, one detection per 6 frames, on a 720p H.264 re-encode made with PyAV (no ffmpeg binary needed). Camera match = ORB + RANSAC inliers between the median upload frame and `configs/reference.jpg`; without a match the scene is empty, so scene rules disable themselves. Tracks are rescaled from the re-encode to the scene's `image_size`. Until `render.py` (P2.3) exists, or whenever it fails, the re-encoded video is returned un-annotated; risk stays [] until `RiskCore` (P2.2). User input can only produce 4xx JSON with a readable message; processing failures become job status `error`.
31. **Rule interpretations (Phase 2, all 13 rules written, all still disabled).** Signal-based rules (red_light, stop_line, and the red-queue exclusion) use only the lamp timeline; for the exclusion an *unknown* state counts as possibly red (a false stopped_vehicle costs more than a missed one), while red_light/stop_line never fire on unknown. A red_light crossing that never enters the intersection is left to stop_line. Lane polygons end at the stop line, so illegal_turn's "leaves the entry lane" means entering *another* lane; otherwise the start is the yaw-rate rise. road_obstacle runs its COCO branch only until the background model (b) lands with frame reads (P2.1). accident's contact test uses footprints within `contact_dist_m` (1.5 m): footprints are box bottom-centres, so a rear-end collision (centres a car length apart) is caught mainly through box overlap + shock only if the footprints come that close; revisit on real footage before enabling. near_miss ends when the pair stops closing (closest approach). Savitzky–Golay smoothing moves kinematic onsets (yaw rise, braking) up to ~0.3 s early; stride-1 refinement (P2.1) is where boundaries get exact.
32. **Part B weights.** The §8 start weights (b = -5, w_ttc = 5, w_closing = 1.5) give 0.18 for TTC 1 s at 6 m/s and 0.03 for TTC 3 s, missing §8's own targets (~0.6, ~0.15). Solving the three target equations and rounding gives b = -3.0, w_ttc = 5.5, w_closing = 1.0 (0.047 / 0.60 / 0.16); w_ped_ttc = 5.0. `red_light_active` = a vehicle faster than wrong_way's min speed in a lane whose (online, lamp-based) signal is red. Online kinematics repeat features' EMA update per track, forgetting tracks unseen for 2 s. The frame index is round(t * fps), so `--risk-stride` works. Tests: `tests/test_risk.py` (targets, causality, no file access in `step`).
33. **Part B time safety.** First real sample (C3896, RTX 4050 laptop, 12 cores): harness decode 0.99x, Part A perception 0.59x, Part B decode + step 1.52x (our step 0.66x: detect 19 ms, ByteTrack 10 ms, features 10 ms, 4K resize 5 ms per processed frame) -> 2.11x of 3x. Kaggle's slower decode (1.32x) would leave little margin, and Part B had no pacing, while going over 3x empties the whole video. `RiskCore` now skips processing slots whenever its step() time exceeds `risk.budget_factor` (0.6) x t + 2 s; it never engages on a fast enough machine. Confirm on the T4 with `bench.py --perception`.
34. **Per-video scene alignment.** The tripod is set up again for every recording. C3902's framing is 80–105 px off C3896's (4K) at the stop line, the median island and both crossings, measured on 100 of 128 matched SIFT points. C3905 is up to 56 px off. That is 1–2 m on the road, so a `scene.json` clicked once is wrong for other videos. `scene/registration.py` builds a median background (traffic drops out), normalises it with CLAHE, matches SIFT features to `configs/reference.jpg`, and fits a video → reference homography with seeded RANSAC. `Scene.transformed` then maps every layer into the video's pixels and composes the road-plane mapping exactly (it is not refitted). Part A registers once per video from `registration.frames` frames spread over it. Part B registers causally from frames it receives (`OnlineRegistration`); when it switches scene it restarts its online kinematics and lamp reader, otherwise the 1–2 m position jump reads as motion (regression test). An unconfident fit (too few inliers, or a corner moved more than 15% of the width) keeps the scene as clicked: the organizers say the test videos come from the same camera and angle.
35. **Vehicle ground extent.** Each vehicle is a `length × width` rectangle on the road (`kinematics.vehicle_dims_m` per class), oriented by its heading, or by its lane direction before it first moves. The box's bottom-centre (the footprint) is the rectangle's point nearest the camera, so the centre lies behind it by the rectangle's half-extent along the road-plane "down the image" direction. `front` and `rear` bumpers are centre ± L/2 along the heading. A car driving toward the camera has its front at the footprint; one driving away has its rear there. Persons, rows without heading or lane, and scenes without a homography fall back to the footprint. `vehicle_corners` gives the four ground corners (the wheels, for solid-line crossing).
36. **Batch-2 rules hardened (review of 2026-09-25).**
    - `red_light` needs a red phase of at least 3 s (shorter reds are lamp occlusions, e.g. a passing red bus). After crossing, the front must enter the intersection within 3 s without a sustained stop (below 1 m/s for 1 s); a sustained stop means stop_line. Slow samples before the line do not count, because pulling away from the head of the queue on red is the most common violation (regression test). The end is when the rear bumper clears the intersection.
    - `stop_line` follows `obj_id` through id switches, tolerates 3 s of occlusion, needs the front bumper at least 1.0 m past the line (covers homography error), and ends at green, or earlier if the vehicle moves on or backs behind the line.
    - `failure_to_yield` counts only pedestrians in the vehicle's path: within 5 m of its axis, from its rear bumper to 6 m ahead of its front. It runs from the front bumper reaching the crosswalk to the rear bumper clearing it; the old footprint-based timing matched ground truth at IoU 0.38.
    - `illegal_turn` needs a heading change of at least 45° and ends when the yaw rate drops back below the start threshold after its peak. The peak skips NaN yaw from slow samples (regression test).
    - `solid_line_crossing` uses the ground-rectangle corners (the wheels), not the image box corners.
    - Boundaries at sample transitions are timed at the midpoint of the two samples.
    - Signal state has an absolute test first: a lamp's lit-colour share in its own box minus that colour's share in its sibling boxes. A red bus in front of a head shows red in every box, so it cancels out. Priority is red, then green, then yellow. The per-lamp percentile test is only a fallback and needs some colour excess. This recognises yellow, which is lit only about 5% of the time, and clips that stay in one state.
    - Still open: `illegal_u_turn` does not yet stitch tracks that switch id mid-turn.

