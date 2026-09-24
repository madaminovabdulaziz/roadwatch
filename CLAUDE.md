# CLAUDE.md — RoadWatch (WIUT Hackathon, elimination task)

You are working on a hackathon submission with a hard deadline on Sunday.
This file is the law. `docs/SPEC.md` is the full technical spec. `docs/WEBSITE_SPEC.md` covers the site and demo.
Read both before you change code in an area you have not touched in this session.

## What we build
A system for ONE fixed CCTV road camera (25 fps, several-minute .mp4 clips):
- **Part A (70% of model score):** `detect_events(video_path) -> [[start_sec, end_sec, label], ...]` for 14 fixed classes.
- **Part B (30%):** `RiskEstimator.reset(meta)` / `.step(frame, t_sec) -> float` = P(accident starts within 5 s), causal.
- A public website with a live demo, EDA, results, and a report.

Elimination score = 0.6 × Model + 0.25 × Website + 0.15 × Code.
**If the package does not run on a clean machine, Model = 0.** Running beats clever. Always.

## Non-negotiable rules
1. `run_submission.py` and `evaluate.py` are the organizers' files. **Never edit them.** Never edit anything in `examples/`.
2. `solution.py` at the repo root must expose exactly: `CLASSES`, `detect_events`, `RiskEstimator`. Keep it thin; logic lives in the `roadwatch/` package.
3. **No network at inference.** All weights live in `weights/`. Nothing may download at runtime (no hub pulls, no font downloads, no pip installs). Do not use Ultralytics `.plot()` or any helper that fetches assets.
4. **No closed or paid model APIs** anywhere in the inference path. Open weights only.
5. **Part B is strictly causal.** `RiskEstimator` never opens the video file, never reads Part A output or caches, and uses only frames passed to `step()` so far.
6. **Determinism.** Fixed seeds; `torch.backends.cudnn.deterministic=True`, `benchmark=False`; TF32 off. Two runs must give the same `predictions.json`.
7. **Time budget.** Part A + Part B together must finish within 3× the video duration on a T4 (16 GB), and that budget also pays for the harness decoding every 4K frame (SPEC §12.10, §12.16). Our target is ≤ 1.0× for our own work on top of that decode. Measure with `scripts/bench.py` after every perception change.
8. **Never crash the harness.** Every event rule runs inside try/except in `pipeline.py`; a failing rule logs and returns []. `step()` must always return a finite float in [0, 1].
9. Label ids are exactly the 14 in `CLASSES`. Never add ids. Never emit a class whose `enabled` flag is false in `configs/thresholds.yaml`.
10. Same-class segments must never overlap in our output (merge them). Clip every segment to `[0, duration]`, require `start < end`, cast to plain Python `float`.

## Architecture (one line each)
- `roadwatch/video.py`: decode, sample every N-th frame, timestamps `t = frame_idx / fps`.
- `roadwatch/perception/`: detector (FP16) + ByteTrack (supervision) → `TrackTable` (pandas).
- `roadwatch/scene/`: loads `configs/scene.json` (hand-calibrated polygons/lines), homography to metres, traffic-light state.
- `roadwatch/features.py`: per-track kinematics in metres: speed, accel, heading, lane, zone flags; pair TTC.
- `roadwatch/events/<class>.py`: one module per class, same signature, returns raw `Segment`s.
- `roadwatch/postprocess.py`: gap-fill, min-duration, boundary refinement, same-class union, clip.
- `roadwatch/pipeline.py`: orchestrates Part A.
- `roadwatch/risk.py`: Part B online perception + risk model.
- `roadwatch/render.py`: annotated MP4 (H.264 via ffmpeg) + event JSON for the website/demo.

## How to work in this repo
- Before coding a module, restate the contract from `docs/SPEC.md` in 3–5 lines, then implement.
- Small commits, one concern each. Message format: `area: what changed`.
- Tunables live in `configs/thresholds.yaml`, never as magic numbers in code. Use round, defensible values.
- Develop against **cached tracks** (`cache/tracks/<video>.parquet`) so rule iteration takes seconds, not minutes.
- After touching events or postprocess: run `python scripts/eval_dev.py` and paste the per-class table into your summary.
- After touching perception or risk: run `python scripts/bench.py` and report seconds per video-second.
- Add a unit test for every geometry helper and every event rule (synthetic tracks are fine: see SPEC §9).
- Keep code readable: type hints, docstrings on public functions, no dead code, no notebooks as the only source.
- If a requirement is ambiguous, pick the option in `docs/SPEC.md §12 (Decisions)`, write it down there if new, and move on. Do not stall.

## Commands
```bash
uv venv --python 3.11 .venv && uv pip install --index-strategy unsafe-best-match -r requirements-dev.txt  # dev env
pip install -r requirements.txt            # what the organizers run (Linux, CUDA 12.6 torch)
python scripts/fetch_weights.py            # dev only; weights are committed/archived for submission
python scripts/cache_tracks.py samples/    # detect+track once per sample video
python run_submission.py --videos samples --out predictions.json --team roadwatch   # organizers' harness
python evaluate.py --pred predictions.json --validate-only
python scripts/eval_dev.py                 # score vs labels/dev_gt.json, per-class table
python scripts/bench.py samples/           # runtime vs budget
pytest -q
```

## Definition of done (for any task)
Code runs, tests pass, `evaluate.py --validate-only` passes on a fresh `predictions.json`, bench is within target, and you reported the numbers.
