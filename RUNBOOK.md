# RoadWatch — Claude Code Runbook (Thu night → Sunday)

Copy-paste prompts, in order. Each has an owner, a goal, and a "done when" check.
Nothing here asks the organizers anything; all open questions are decided in `docs/SPEC.md §12`.

## Roles
- **P1 — Integrator & Perception (Abdulaziz):** repo, packaging, video IO, detector/tracker, pipeline, post-processing, Docker/Kaggle runs, demo backend.
- **P2 — Scene, Labels, Rules & Part B:** scene calibration, dev labels, eval/tuning, most event rules, risk estimator.
- **P3 — Web, EDA & Renders:** website, EDA scripts, annotated renders, results pages, report text.

## How to use Claude Code here
- Put `CLAUDE.md`, `docs/SPEC.md`, `docs/WEBSITE_SPEC.md`, `RUNBOOK.md` in the repo root before the first session.
- Each person works in their own **git worktree + branch** (`git worktree add ../rw-p2 -b p2`), one Claude Code session per worktree. Merge to `main` often (at least every checkpoint).
- Start each task in **plan mode** (Shift+Tab). Read the plan, fix it, then let it code.
- `/clear` between unrelated tasks, so old context doesn't leak.
- Always end a task with: tests run, numbers reported, commit made.
- If Claude drifts (edits `run_submission.py`, adds magic numbers, downloads at runtime), stop it and paste the relevant CLAUDE.md rule.

## Human-only jobs (Claude can't do these)
- Watch every sample video fully. Write down what you see (P1 + P2 + P3 split the videos, tonight).
- Click the scene polygons in the calibration tool (P2).
- Label the dev set (all three, Friday morning, ~1 hour each).
- Run on a Kaggle T4 notebook; deploy Vercel + HF Space; record team info and photos.

---

## PHASE 0 — Thursday night: skeleton that runs end to end (P1)

Goal: before sleep, `run_submission.py` produces a valid `predictions.json` (empty events, zero risk) from our package, and it runs on a Kaggle T4.

### P0.1 Bootstrap
```
Read CLAUDE.md, docs/SPEC.md and docs/WEBSITE_SPEC.md fully. The organizers' starter kit
(solution.py, run_submission.py, evaluate.py, examples/) is in the repo root; read those three
files line by line and summarize: the exact CLI args of run_submission.py and evaluate.py, how
the harness computes t_sec and calls RiskEstimator, the order it calls detect_events vs step, and
how it enforces the time budget. If anything contradicts docs/SPEC.md, update SPEC §12 with the
harness's actual behaviour.

Then create the full repo layout from SPEC §2 with stubs: every module exists, has a docstring
stating its contract, and type-hinted function signatures that raise NotImplementedError or return
safe empty values. Create configs/thresholds.yaml with every class listed (enabled flags, min_score,
merge_gap, min_dur from SPEC §5–6), .gitignore (cache/, *.parquet, outputs), requirements.txt
(pinned), pyproject/ruff config, pytest setup.

Rewrite solution.py as a thin wrapper: CLASSES unchanged, detect_events calls
roadwatch.pipeline.detect_events inside try/except returning [] on error, RiskEstimator delegates
to roadwatch.risk.RiskCore and always returns a finite float in [0,1].

Done when: `python run_submission.py <samples dir> ...` writes predictions.json,
`python evaluate.py --pred predictions.json --validate-only` passes, `pytest -q` passes. Commit.
```

### P0.2 Video IO + benchmark
```
Implement roadwatch/video.py per SPEC §3: FrameReader(video_path, stride) yielding
(frame_idx, t_sec, frame_bgr) with t_sec computed exactly the way the harness does, VideoMeta
(fps, width, height, n_frames, duration), and read_window(t0, t1, stride=1). Use cv2 with grab()
for skipped frames; fall back gracefully on bad frames. Write scripts/bench.py: for each video,
time decode-only at stride 1 and 3, and print seconds per video-second. Add tests with a tiny
synthetic mp4 generated in the test.
Done when: bench prints numbers for all samples; tests pass; commit.
```

### P0.3 Detector + tracker + cache
```
Implement roadwatch/perception per SPEC §3: Detector (YOLO11m COCO, FP16 on CUDA, fp32 on CPU,
imgsz from config, class filter, batching, warm-up, loaded ONCE via a module-level cache from a
local path in weights/, no network, never .plot()), OnlineTracker wrapping supervision.ByteTrack
with separate trackers per class group, and run_perception(video_path, stride) -> TrackTable
(pandas, columns per SPEC §3). scripts/fetch_weights.py downloads weights into weights/ (dev
only). scripts/cache_tracks.py writes cache/tracks/<video>.parquet for every sample.
Set seeds and deterministic flags in roadwatch/config.py. Extend bench.py with detect+track
timing. Add tests/test_offline.py that blocks sockets and imports + runs detection on 1 frame.
Done when: parquet cache exists for all samples; bench shows detect+track time per
video-second at stride 3 (target ≤ 0.4× on T4); offline test passes; commit.
```

### P0.4 Packaging on clean machine
```
Write a Dockerfile (CUDA runtime base, Python 3.10+, requirements, weights copied in) whose
default command runs run_submission.py on /data/videos and then evaluate.py --validate-only.
Write the README skeleton: setup, the two commands, models+licences, datasets+licences,
attribution section, assumptions. Also write kaggle/run_on_kaggle.md: exact cells to clone the
repo, install requirements, run the harness on the samples, and print bench numbers.
Done when: docker build works locally (or on Kaggle) and `docker run --network none` produces a
valid predictions.json. Commit and merge to main.
```

**Checkpoint 0 (end of Thu):** empty-but-valid submission runs offline on a T4. Share bench numbers in the team chat.

---

## PHASE 1 — Friday: perception features, scene, labels, website shell (parallel)

### P2 — P1.1 Scene calibration tool
```
Read CLAUDE.md and docs/SPEC.md §4. Build scripts/calibrate_scene.py: an OpenCV tool that opens
a reference frame (median of 50 frames from a sample, saved to configs/reference.jpg), lets me pick
a layer from a keyboard menu (carriageway, sidewalks, parking_zones, intersection, lanes, exits,
stop_lines, crosswalks, solid_lines, signals red/yellow/green boxes, no_u_turn_zones, homography
points), click points, Enter to finish a shape, U to undo, prompt in terminal for ids/attributes
(lane direction, approach, signal id, allowed_exits, world coords for homography), S to save
configs/scene.json in the SPEC §4 schema, and draw all saved layers with labels. Also write
roadwatch/scene/scene.py: Scene.load(), point_in(layer), lane_of(point), segment crossing tests,
to_world()/to_image() via homography, and scripts/render_scene.py that saves an overlay PNG.
Tests for geometry helpers.
Done when: I can calibrate the full scene in < 30 minutes and the overlay PNG looks right. Commit.
```
(Human: P2 does the clicking right after.)

### P2 — P1.2 Lane flow + signal state
```
Implement scripts/learn_lane_flow.py: from cached tracks, compute median image-space velocity
direction per lane polygon and a grid flow field (for EDA); print proposed lane directions and
write them into scene.json only after I confirm in the terminal.
Implement roadwatch/scene/light.py SignalStateEstimator: per signal, mean V and hue-masked
brightness in red/yellow/green lamp boxes, pick the brightest lit lamp, median filter over 0.5 s,
produce a timeline [(t_start, t_end, state)] offline and an online update(frame, t) for Part B.
If scene has no signals, provide a flow-based fallback (stop-line approach lanes moving →
GREEN, stopped while cross flow moves → RED) and mark confidence low.
Add scripts/plot_signal_timeline.py for a quick visual check on each sample.
Done when: the timeline matches what I see in the video on 3 spot checks per sample. Commit.
```

### P1 — P1.3 Kinematics features
```
Read docs/SPEC.md §3 and §5. Implement roadwatch/features.py: add_kinematics(tt, scene,
mode="offline"|"online"): footprint→metres, Savitzky–Golay (offline) or EMA (online) smoothing,
vx, vy, speed, accel (signed along heading), heading_deg, yaw_rate, kin_valid, lane_id, on_road,
in_crosswalk, in_intersection, in_no_uturn, in_parking, sidewalk flags; front_point per SPEC §12.4;
position-persistence merging for stationary tracks (SPEC §3). Also pair_features(tt_frame):
pairwise distance, closing speed, TTC for converging pairs < 25 m. Vectorize with numpy/pandas.
Tests with synthetic straight-line and braking tracks where the correct speed/accel are known.
Done when: tests pass; on one sample, printed speed percentiles look plausible (city 20–60 km/h). Commit.
```

### ALL — Dev labels (human, Friday morning)
Split samples between the three of you. Label with SPEC §5 start/end conventions, in `labels/raw/<you>.csv`: `video,start,end,label,notes`. Use a player with frame stepping and a timestamp (mpv with OSD, or Label Studio video timeline). When unsure, write a note; P2 decides.

### P2 — P1.4 Labels → GT + eval loop
```
Write scripts/labels_to_gt.py: merge labels/raw/*.csv, validate (labels in CLASSES, start<end,
within duration), union same-class overlaps, write labels/dev_gt.json in the organizer GT format
with duration and fps read from each video. Write scripts/eval_dev.py: run the pipeline from
cached tracks (flag --no-cache to run full), write predictions_dev.json, call evaluate.py
--pred ... --gt labels/dev_gt.json, and print a per-class table (TP/FP/FN at each τ, F1) plus a
list of FPs and FNs with timestamps so we can look at them. Write scripts/tune.py: grid search
over a small YAML-defined grid of thresholds per class on cached tracks, reporting best values
rounded to sensible numbers.
Done when: eval_dev runs end to end on the empty pipeline and shows all-zero scores correctly. Commit.
```

### P3 — P1.5 EDA
```
Read docs/WEBSITE_SPEC.md (EDA section) and SPEC §3. Write scripts/eda.py that, from the sample
videos and cache/tracks, produces into web/public/data/eda/: video stats JSON (resolution, fps,
duration, mean brightness over time as lighting proxy), per-second object counts by class,
motion heatmap PNG overlaid on the reference frame, trajectory PNG coloured by heading, lane flow
arrows PNG, density per minute, speed histogram per lane, signal phase timeline. JSON for anything
charted interactively; PNG only for image overlays. Deterministic output.
Done when: all files generate from one command. Commit.
```

### P3 — P1.6 Website shell
```
Read docs/WEBSITE_SPEC.md. Create web/ as a Next.js static-export app with Tailwind and Plotly
(react-plotly.js, dynamic import, no SSR). Build every page from the spec with real layout and
placeholder data loaded from web/public/data/*.json (so later we only regenerate data). Components:
EventTimeline (clickable, seeks a <video>), RiskCurve synced to video currentTime, EventsTable,
ClassLegend with fixed per-class colours shared in one TS file, UploadDemo (calls the demo API
from WEBSITE_SPEC, polling with progress, friendly errors, two example buttons). Mobile-first.
Deploy to Vercel.
Done when: site is live on a public URL and every page renders on a phone. Commit.
```

### P1 — P1.7 Demo backend shell
```
Read docs/WEBSITE_SPEC.md (Demo API). Build demo/ as FastAPI + a single background worker thread
with an in-memory job queue, exactly the API contract in the spec. Validation (size, duration
≤120 s, codec), ffmpeg re-encode on input, progress stages, 1-hour expiry, never a 500 on bad
input. For now the pipeline call can be the current roadwatch pipeline (even if it returns few
events) plus roadwatch/render.py output. Dockerfile for a Hugging Face CPU Space (CPU config:
YOLO11s, stride 5, imgsz 640). Camera-match check vs configs/reference.jpg per spec.
Done when: deployed on HF Spaces; the Vercel site uploads a sample clip and gets a result back. Commit.
```

**Checkpoint 1 (end of Fri):** scene.json done, dev_gt.json done, eval loop runs, site + demo live (even with weak results).

---

## PHASE 2 — Saturday: event rules, Part B, renders (parallel)

For every rule prompt, use this template (fill the class list):
```
Read CLAUDE.md, docs/SPEC.md §5, §6, §7 and roadwatch/events/base.py. Implement the rules for:
<CLASSES>. Follow the trigger/start/end rules in SPEC §5 exactly; all numbers come from
configs/thresholds.yaml. Each rule returns Segments with a score in [0,1] and track_ids. Write
synthetic-track unit tests per class that check start/end within 0.2 s of the expected values,
plus a negative test (normal traffic → no event). Then run scripts/eval_dev.py and show the
per-class table and the FP/FN list. For every FP/FN, open the timestamp in the cached tracks,
explain the cause in one line, and propose one fix. Apply fixes that are general (not tuned to
one clip). Commit.
```

- **P1 — Batch 1:** `stopped_vehicle, wrong_way, jaywalking, congestion`
- **P2 — Batch 2:** `red_light, stop_line, failure_to_yield, solid_line_crossing, illegal_u_turn, illegal_turn`
- **P1 — Batch 3 (after batch 1):** `accident, near_miss, road_obstacle` (+ `fire_smoke` stays disabled)

### P1 — P2.1 Post-processing + boundary refinement
```
Read docs/SPEC.md §6. Implement roadwatch/postprocess.py steps 1–6 in order, and the boundary
refinement using video.read_window at stride 1 with detection on only those frames, under the
15% frame budget. Wire it into pipeline.py with per-rule try/except and per-stage timing logs.
Tests for merge, min-dur, union, clip, float casting, ongoing-at-end → duration.
Then run eval_dev with and without refinement and report mean F1 at τ=0.7 for both.
Done when: refinement improves or equals τ=0.7 F1; bench total ≤ 1.0× duration. Commit.
```

### P2 — P2.2 Part B risk estimator
```
Read docs/SPEC.md §8 and §1 (Score B). Implement roadwatch/risk.py RiskCore: its own online
perception (stride 3, shared global detector weights, own tracker), online kinematics, causal
signal state, the features and score formula from §8 with weights in thresholds.yaml,
hysteresis, exception safety. It must not open files or read cache/. Write scripts/eval_risk.py
that streams sample videos frame-by-frame exactly like the harness and reports AP, alarm F1, mTTA,
Score B vs labels/dev_gt.json, plus a plot of the risk curve with accident/near-miss spans shaded.
Add a test that asserts RiskCore never touches the filesystem (monkeypatch open/cv2.VideoCapture).
If dev has no accidents, also report: number of alarms per 10 minutes on normal traffic (target
≤ 1) and max score during near-misses.
Done when: numbers reported, false alarms on normal traffic ≤ 1 per 10 min, bench A+B ≤ 1.2×. Commit.
```

### P3 — P2.3 Renderer + results data
```
Read docs/WEBSITE_SPEC.md (visual rules). Implement roadwatch/render.py and
scripts/render_samples.py: for each sample, annotated H.264 MP4 (ffmpeg libx264 yuv420p
+faststart), poster JPG, events JSON, risk JSON, and short per-event clips (2 s before → 2 s
after) for the class gallery, into web/public/data/results/. Colours from one shared palette
file (also exported for the web). Optional face/plate blur flag, on by default for web output.
Done when: every sample renders and plays in Chrome and on a phone. Commit.
```

### P3 — P2.4 Results + dashboard pages with real data
```
Wire Results, Dashboard, and EDA pages to the generated JSON/MP4 files. Timeline click seeks the
video; risk curve cursor follows playback; class gallery; dev per-class F1 table from
eval_dev output (write it to web/public/data/metrics.json in eval_dev.py). Dashboard: events per
class, per lane, per minute, event-location heat on reference frame. Lighthouse check on mobile.
Done when: deployed and every sample fully browsable. Commit.
```

### ALL — P2.5 Tuning pass (Saturday evening)
```
Run scripts/tune.py for the enabled dev-validated classes, then apply SPEC §7 enable policy:
print a table class → dev F1 / number of predictions on samples / has dev labels / decision
(enable, raise threshold, disable). For classes with no dev labels, dump every predicted event
with a thumbnail strip so we can review by eye. Update thresholds.yaml with round values. Show
the before/after eval_dev table. Commit.
```
(Human: review every predicted event for classes without dev labels. Wrong → raise threshold or disable.)

**Checkpoint 2 (end of Sat):** all rules in, Part B in, eval numbers on site, demo returns real events.

---

## PHASE 3 — Saturday night / Sunday morning: hardening (P1 leads, all help)

### P3.1 Determinism + budget
```
Write/run scripts/check_determinism.py: run the full harness twice on all samples and diff
predictions.json (events exact, risk within 1e-6). Fix any source of nondeterminism (set/dict
iteration, unsorted groupby, CUDA flags). Run bench on the largest sample and report A, B, total
× duration. If total > 1.2×, raise stride or drop to YOLO11s and report the F1 change.
Done when: identical outputs, total ≤ 1.2× on T4. Commit.
```

### P3.2 Clean-machine test
```
Simulate the organizers: fresh clone of the tagged candidate commit in a new directory (or fresh
Kaggle notebook), install only from requirements.txt (and separately via Dockerfile), weights from
the repo/archive, network disabled at run time, run the two commands on the samples. Fix anything
that needed a manual step. Update README so a stranger can do it without asking us anything.
Done when: zero manual steps, validate-only passes. Commit.
```

### P3.3 Code quality sweep
```
Review the whole repo against the Code rubric (runs as submitted, reproducibility, structure,
engineering judgement). Remove dead code and unused files, make sure every module has a docstring,
ruff clean, pytest green, no notebooks as the only source, training/tuning scripts present, README
lists every dataset/model/licence and attribution. Generate predictions_samples.json with
scripts/make_predictions_samples.py and confirm it matches a fresh harness run.
Output a checklist of the rubric with pass/fail. Commit.
```

---

## PHASE 4 — Sunday: freeze and ship

Freeze model code **6 hours before the deadline.** After that: only README, website, and report changes.

### P4.1 Report + approach page
```
Write the one-page technical report (web + REPORT.md) in simple, plain English: what we built,
the pipeline, what is learned vs rule-based and why, datasets/models/licences, dev results table,
runtime numbers, what worked, what did not (with the concrete failure cases from Results), what we
would do next. Also finish the Approach page so a reader can rebuild the pipeline. Use only
numbers from generated files; do not invent any.
```

### P4.2 Final checklist (human, go through every line)
- [ ] Tagged commit exists; `git status` clean; tag pushed.
- [ ] Fresh clone + requirements + `--network none` → valid predictions.json.
- [ ] Two runs identical. Total runtime ≤ 1.2× duration on T4.
- [ ] Weights ≤ 5 GB, in the repo or in a downloadable archive linked in README.
- [ ] No class emitted that we didn't review; fire_smoke disabled unless validated.
- [ ] `run_submission.py`, `evaluate.py`, `examples/` byte-identical to the starter kit (`git diff` against the original).
- [ ] Website live: all pages, every sample annotated, demo works with a fresh upload from a phone, example buttons work.
- [ ] Team page complete (roles, contributions, GitHub, LinkedIn, portfolios, past projects).
- [ ] Links page: repo, weights, predictions_samples.json.
- [ ] Uptime ping on the demo; HF Space not sleeping.

---

## Emergency prompts

**Over time budget**
```
Bench shows total > 1.5× duration. Profile each stage with scripts/bench.py --profile and propose
the smallest change that brings total ≤ 1.0× with the least F1 loss (stride, imgsz, model size,
refinement budget, batch size). Apply it, re-run bench and eval_dev, report both.
```

**Harness crash / empty output on some video**
```
run_submission.py logged an exception for <video>. Reproduce it in isolation, find the root
cause, fix it, add a regression test, and make sure the failure path in pipeline.py would have
still returned partial events instead of [] for the whole video.
```

**Too many false positives for a class**
```
Class <X> has <N> false positives on dev (list from eval_dev). Group them by cause (occlusion,
ID switch, calibration, queue, rider, etc.). Propose general fixes per group, apply the top two,
and show the per-class table before/after. If precision is still < 0.5, raise min_score so FP = 0
on samples and report the recall cost.
```

**Demo broken right before judging**
```
The demo fails on <symptom>. Make the backend never fail the user: catch the error, return
status=error with a clear message, and fall back to returning events without the annotated video
if rendering is what fails. Redeploy and test with a sample upload and a random non-camera clip.
```
