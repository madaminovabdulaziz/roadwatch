"""Part B risk (SPEC §8, RUNBOOK P2.2): score targets, features, causality, no file access in step()."""

from __future__ import annotations

import builtins
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from roadwatch.config import WEIGHTS_DIR, load_thresholds
from roadwatch.risk import OnlineKinematics, RiskCore, risk_features, risk_score
from roadwatch.scene.scene import Scene
from roadwatch.types import FrameDetections

PX = 10.0
FPS = 30.0
SCENE = Scene(
    {
        "homography": {
            "image_pts": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "world_pts": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        "carriageway": [[0, 200], [1000, 200], [1000, 600], [0, 600]],
        "lanes": [
            {
                "id": "e1",
                "polygon": [[0, 400], [1000, 400], [1000, 500], [0, 500]],
                "direction": [1, 0],
                "signal": "L1",
            }
        ],
    }
)


def cfg() -> dict:
    c = dict(load_thresholds()["risk"])
    groups = load_thresholds()["perception"]["tracker_groups"]
    c["groups"] = {"movers": groups["vehicles"] + groups["two_wheelers"], "persons": groups["persons"]}
    c["wrong_way_speed_mps"], c["wrong_way_max_cos"] = 2.0, -0.5
    return c


CALM = {
    "ttc_min": math.inf,
    "closing_speed": 0.0,
    "decel_max": 0.0,
    "wrong_way": 0.0,
    "red_light": 0.0,
    "ped_ttc": math.inf,
}


def test_score_meets_the_spec_targets() -> None:
    c = cfg()
    assert risk_score(CALM, c) < 0.05
    assert 0.55 <= risk_score({**CALM, "ttc_min": 1.0, "closing_speed": 6.0}, c) <= 0.65
    assert 0.10 <= risk_score({**CALM, "ttc_min": 3.0, "closing_speed": 6.0}, c) <= 0.20


def rows(*objs: tuple) -> pd.DataFrame:
    """(id, cls, x_m, y_m, vx, vy, ax, ay) -> the frame table risk_features expects."""
    df = pd.DataFrame(objs, columns=["track_id", "cls", "X", "Y", "vx", "vy", "ax", "ay"])
    return df.assign(fx=df["X"] * PX, fy=df["Y"] * PX)


def test_features() -> None:
    c = cfg()
    head_on = risk_features(
        rows((1, "car", 10, 45, 5, 0, 0, 0), (2, "car", 30, 45.5, -5, 0, 0, 0)), SCENE, set(), c
    )
    assert head_on["ttc_min"] == pytest.approx(2.0, abs=0.01) and head_on["closing_speed"] == pytest.approx(
        10.0, abs=0.1
    )
    ped = risk_features(
        rows((1, "car", 10, 45, 8, 0, 0, 0), (2, "person", 26, 45, 0, 0, 0, 0)), SCENE, set(), c
    )
    assert ped["ped_ttc"] == pytest.approx(2.0) and math.isinf(ped["ttc_min"])
    wrong = risk_features(rows((1, "car", 50, 45, -8, 0, 0, 0)), SCENE, set(), c)
    assert wrong["wrong_way"] == 1.0 and wrong["red_light"] == 0.0
    hard = risk_features(rows((1, "car", 50, 45, 8, 0, -7, 0)), SCENE, {"e1"}, c)
    assert hard["decel_max"] == pytest.approx(7.0 - c["decel_floor_mps2"])  # braking above everyday
    assert hard["red_light"] == 0.0  # in its lane (no intersection here): approaching a red is legal
    mild = risk_features(rows((1, "car", 50, 45, 8, 0, -3, 0)), SCENE, set(), c)
    assert mild["decel_max"] == 0.0
    assert risk_features(pd.DataFrame(), SCENE, set(), c) == CALM


def test_online_kinematics_converge_and_average_out_jitter() -> None:
    kin = OnlineKinematics(cfg())
    ids = np.array([7])
    outs = [kin.update(i / 10, ids, np.array([[2.0 * i / 10, 0.0]]))[0][0, 0] for i in range(40)]
    assert math.isnan(outs[0])  # no velocity until the fit has enough samples
    assert abs(outs[-1] - 2.0) < 1e-9  # a line fit is exact on uniform motion

    # a parked car whose footprint jitters by 0.3 m: the fitted speed stays well under walking pace
    rng = np.random.default_rng(0)
    parked = OnlineKinematics(cfg())
    speeds = [
        np.hypot(*parked.update(i / 10, ids, rng.normal(0, 0.3, size=(1, 2)))[0][0]) for i in range(300)
    ]
    assert np.nanpercentile(speeds, 95) < 1.5


class FakeDetector:
    """Scripted detections: `boxes(t)` -> list of (x_m, y_m, cls_id) footprints (4 m x 3 m boxes)."""

    def __init__(self, boxes) -> None:
        self.boxes = boxes

    def frame_size_for(self, width: int, height: int) -> tuple[int, int]:
        return 100, 100

    def predict(self, batch, native_size=None):
        out = []
        for idx, t, _ in batch:
            objs = self.boxes(t)
            xyxy = np.array(
                [[x * PX - 20, y * PX - 30, x * PX + 20, y * PX] for x, y, _ in objs], np.float32
            ).reshape(-1, 4)
            out.append(
                FrameDetections(
                    idx,
                    t,
                    xyxy,
                    np.full(len(objs), 0.9, np.float32),
                    np.array([c for *_, c in objs], np.int64),
                )
            )
        return out


# The head-on car drives west, so in a lane with an eastbound direction wrong_way would fire too.
NO_LANES = Scene({k: v for k, v in SCENE.layers.items() if k != "lanes"})


def run(boxes, seconds: float, scene: Scene = NO_LANES) -> list[tuple[float, float]]:
    core = RiskCore(detector=FakeDetector(boxes), scene=scene)
    core.reset(
        {"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": int(seconds * FPS)}
    )
    frame = np.zeros((1000, 1000, 3), np.uint8)
    return [(i / FPS, core.step(frame, i / FPS)) for i in range(int(seconds * FPS))]


def test_head_on_course_raises_the_alarm_before_the_crash_and_calm_stays_low() -> None:
    crash = run(lambda t: [(20 + 10 * t, 45, 2), (80 - 10 * t, 45.5, 2)], 2.9)  # they meet at t = 3 s
    scores = dict(crash)
    assert all(0.0 <= s <= 1.0 and math.isfinite(s) for _, s in crash)
    assert max(s for t, s in crash if t < 0.5) < 0.1
    assert max(s for t, s in crash if t > 1.8) >= 0.5  # alarm at least ~1 s before contact
    assert scores[crash[-1][0]] >= 0.5
    calm = run(lambda t: [(20 + 10 * t, 45, 2), (35 + 10 * t, 45, 2)], 5)  # same speed, 15 m apart
    assert max(s for _, s in calm) < 0.05


def test_no_homography_gives_the_bias_only() -> None:
    flat = run(lambda t: [(20 + 10 * t, 45, 2), (80 - 10 * t, 45.5, 2)], 2.9, scene=Scene())
    assert max(s for _, s in flat) < 0.05


def test_no_homography_never_runs_the_detector() -> None:
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"detector.{name} used without a homography")

    core = RiskCore(detector=Untouchable(), scene=Scene())
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": 30})
    frame = np.zeros((10, 10, 3), np.uint8)
    scores = {core.step(frame, i / FPS) for i in range(30)}
    assert len(scores) == 1 and next(iter(scores)) < 0.05


def test_step_never_touches_files(monkeypatch: pytest.MonkeyPatch) -> None:
    core = RiskCore(detector=FakeDetector(lambda t: [(20 + 10 * t, 45, 2)]), scene=SCENE)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": 90})

    def forbidden(*args, **kwargs):
        raise AssertionError("RiskCore.step touched the filesystem")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    frame = np.zeros((1000, 1000, 3), np.uint8)
    assert all(0.0 <= core.step(frame, i / FPS) <= 1.0 for i in range(90))


def test_skipped_frames_return_the_previous_score_and_risk_stride_works() -> None:
    core = RiskCore(detector=FakeDetector(lambda t: []), scene=SCENE)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": 90})
    frame = np.zeros((1000, 1000, 3), np.uint8)
    first = core.step(frame, 0.0)
    assert core.step(frame, 1 / FPS) == first
    assert core.step(frame, 30 / FPS) == pytest.approx(first)  # a jump (--risk-stride 30) still processes


@pytest.mark.skipif(not (WEIGHTS_DIR / "yolo11m.torchscript").exists(), reason="weights not downloaded")
def test_real_detector_path_returns_floats() -> None:
    core = RiskCore(scene=SCENE)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 640, "height": 360, "n_frames": 10})
    frame = np.zeros((360, 640, 3), np.uint8)
    assert all(0.0 <= core.step(frame, i / FPS) <= 1.0 for i in range(10))


def test_pacing_skips_work_when_behind_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    import roadwatch.risk as risk_mod

    clock = iter(x * 0.5 for x in range(10_000))  # every processed step appears to take 0.5 s
    monkeypatch.setattr(risk_mod.time, "perf_counter", lambda: next(clock))
    calls = []
    fake = FakeDetector(lambda t: [])
    original = fake.predict
    fake.predict = lambda *a, **k: calls.append(1) or original(*a, **k)
    core = RiskCore(detector=fake, scene=SCENE)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": 300})
    frame = np.zeros((1000, 1000, 3), np.uint8)
    for i in range(300):  # 10 s, 100 processing slots at stride 3
        assert 0.0 <= core.step(frame, i / FPS) <= 1.0
    # budget 0.6 x 10 s + 2 s = 8 s of step time at 0.5 s each -> about 16 slots, the rest skipped
    assert 10 <= len(calls) <= 20 and core.skipped >= 80


def test_aligning_the_scene_mid_video_does_not_look_like_motion() -> None:
    # Part B aligns the scene to the video from the frames it receives (SPEC §12.34). The moment it does,
    # every footprint's position in metres jumps (here 1.5 m); stationary cars must not read as braking.
    shift = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # reference -> video: 15 px

    class AlignAtOneSecond:
        done = False

        def offer(self, frame, t_sec):
            if t_sec < 1.0:
                return None
            self.done = True
            return NO_LANES.transformed(shift, (1000, 1000))

    core = RiskCore(detector=FakeDetector(lambda t: [(30, 45, 2)]), scene=NO_LANES)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": 90})
    core.registration = AlignAtOneSecond()
    frame = np.zeros((1000, 1000, 3), np.uint8)
    scores = [core.step(frame, i / FPS) for i in range(90)]
    assert max(scores) < 0.05, max(scores)


# ------------------------------------------------------------------ normal traffic must stay calm (§12.40)
def jittered(positions, sigma_m: float):
    """Boxes at `positions(t)` with deterministic Gaussian footprint jitter (detector noise)."""

    def boxes(t: float):
        rng = np.random.default_rng(int(round(t * FPS)))
        return [(x + rng.normal(0, sigma_m), y + rng.normal(0, sigma_m), c) for x, y, c in positions(t)]

    return boxes


def test_a_stationary_queue_with_detector_jitter_raises_no_alarm() -> None:
    # 15 cars waiting at a red light in 3 lanes, 6.5 m apart, footprints jittering by 0.3 m (about
    # +-3 px far from the camera at 4K): nothing moves, so nothing may look like a collision course.
    queue = [(20 + 6.5 * k, y, 2) for k in range(5) for y in (45.0, 48.5, 52.0)]
    scores = run(jittered(lambda t: queue, 0.3), 30)
    assert max(s for _, s in scores) < 0.2


def test_a_normal_stop_behind_a_stopped_car_raises_no_alarm() -> None:
    # 12 m/s, braking at 3 m/s^2 from 2 s on, stopping 2 m behind a car standing at x = 60 m.
    def follower(t: float) -> float:
        brake = min(max(t - 2.0, 0.0), 4.0)
        return 10 + 12 * min(t, 2.0) + 12 * brake - 1.5 * brake**2

    scores = run(jittered(lambda t: [(follower(t), 45, 2), (60, 45, 2)], 0.05), 8)
    assert max(s for _, s in scores) < 0.5


def test_leaving_through_the_frame_edge_is_not_braking() -> None:
    # A car drives down the image at 10 m/s and out through the bottom edge (y = 100 m = 1000 px): its
    # clipped box freezes the footprint, which must not read as a hard stop next to a waiting car.
    def boxes(t: float):
        y = 50 + 10 * t
        return [(20, min(y, 100.0), 2), (30, 97.0, 2)]  # a waiting car 10 m to the side

    scores = run(boxes, 6, scene=NO_LANES)
    assert max(s for _, s in scores) < 0.1


def test_red_light_feature_fires_past_the_line_not_in_the_queue() -> None:
    # Lane e1 (signal L1, red) ends at the stop line x = 50 m; the intersection starts there.
    scene = Scene(
        {
            **SCENE.layers,
            "lanes": [{**SCENE.layers["lanes"][0], "polygon": [[0, 400], [500, 400], [500, 500], [0, 500]]}],
            "intersection": [[500, 200], [1000, 200], [1000, 600], [500, 600]],
        }
    )
    approaching = pd.DataFrame(
        {
            "track_id": [1],
            "cls": ["car"],
            "X": [40.0],
            "Y": [45.0],
            "vx": [8.0],
            "vy": [0.0],
            "ax": [0.0],
            "ay": [0.0],
            "fx": [400.0],
            "fy": [450.0],
            "last_lane": ["e1"],
            "age": [3.0],
        }
    )
    running = approaching.assign(X=55.0, fx=550.0)  # past the line, came from e1, still red
    assert risk_features(approaching, scene, {"e1"}, cfg())["red_light"] == 0.0
    assert risk_features(running, scene, {"e1"}, cfg())["red_light"] == 1.0
    assert risk_features(running, scene, set(), cfg())["red_light"] == 0.0


def test_part_b_paces_against_the_videos_real_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Part A marked this video's start 100 s ago: its 3x budget (27 s at 2.7x for 10 s) is gone, so
    # once Part B has seen enough to project its finish it must stop processing (the harness still
    # needs its own decoding time; over 3x the whole video, Part A included, scores empty).
    import time as _time

    from roadwatch import budget

    monkeypatch.setitem(budget._STARTS, "v.mp4", _time.perf_counter() - 100.0)
    core = RiskCore(detector=FakeDetector(lambda t: [(20 + 10 * t, 45, 2)]), scene=NO_LANES)
    core.reset({"video_id": "v.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": int(10 * FPS)})
    frame = np.zeros((1000, 1000, 3), np.uint8)
    scores = [core.step(frame, i / FPS) for i in range(int(10 * FPS))]
    pace_after = core.cfg["pace_after_sec"]
    assert core.skipped >= int((10 - pace_after) * FPS / core.cfg["stride"]) - 1
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_on_time_part_b_never_skips() -> None:
    # Without a Part A start mark the deadline counts from reset; a fast machine never engages pacing,
    # so normal runs stay deterministic.
    core = RiskCore(detector=FakeDetector(lambda t: [(20 + 10 * t, 45, 2)]), scene=NO_LANES)
    core.reset({"video_id": "fresh.mp4", "fps": FPS, "width": 1000, "height": 1000, "n_frames": int(8 * FPS)})
    frame = np.zeros((1000, 1000, 3), np.uint8)
    for i in range(int(8 * FPS)):
        core.step(frame, i / FPS)
    assert core.skipped == 0
