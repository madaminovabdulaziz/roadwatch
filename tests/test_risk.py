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
    c["vehicle_dims"] = load_thresholds()["kinematics"]["vehicle_dims_m"]
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
    # Head-on at 8 m/s each, footprints 20 m apart. Moving across the view, a car's footprint is the
    # middle of its side, so the bumpers are 20 - 4.6 = 15.4 m apart: TTC 15.4 / 16, and stopping short
    # needs 16^2 / (2 * 15.4) = 8.3 m/s^2 of braking, far beyond everyday driving.
    head_on = risk_features(
        rows((1, "car", 10, 45, 8, 0, 0, 0), (2, "car", 30, 45.5, -8, 0, 0, 0)), SCENE, set(), c
    )
    assert head_on["ttc_min"] == pytest.approx(15.4 / 16, abs=0.01)
    assert head_on["closing_speed"] == pytest.approx(16.0, abs=0.1)
    # The same encounter at 5 m/s each needs only 3.2 m/s^2 to stop: an ordinary stop, no conflict.
    slow = risk_features(
        rows((1, "car", 10, 45, 5, 0, 0, 0), (2, "car", 30, 45.5, -5, 0, 0, 0)), SCENE, set(), c
    )
    assert math.isinf(slow["ttc_min"])
    # A standing pedestrian (a point) 12 m ahead of a car at 10 m/s: gap 12 - 2.3 = 9.7 m, 5.2 m/s^2.
    ped = risk_features(
        rows((1, "car", 10, 45, 10, 0, 0, 0), (2, "person", 22, 45, 0, 0, 0, 0)), SCENE, set(), c
    )
    assert ped["ped_ttc"] == pytest.approx(0.97, abs=0.01) and math.isinf(ped["ttc_min"])
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
    # 12 m/s, braking at 3 m/s^2 from 2 s on, stopping 2 m (bumper to bumper) behind a 4.6 m car standing
    # at x = 64.6 m. (Before SPEC §12.47 this test stopped the footprints 2 m apart, which with real car
    # lengths is 2.6 m of overlap: a crash.)
    def follower(t: float) -> float:
        brake = min(max(t - 2.0, 0.0), 4.0)
        return 10 + 12 * min(t, 2.0) + 12 * brake - 1.5 * brake**2

    scores = run(jittered(lambda t: [(follower(t), 45, 2), (64.6, 45, 2)], 0.05), 8)
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


def test_a_scene_clicked_at_another_resolution_is_scaled_to_the_frames() -> None:
    # the demo feeds 720p frames while configs/scene.json was clicked on the 4K reference frame
    head_on = lambda t: [(20 + 10 * t, 45, 2), (80 - 10 * t, 45.5, 2)]  # noqa: E731
    at_2x = NO_LANES.transformed(np.diag([2.0, 2.0, 1.0]), (2000, 2000))
    assert run(head_on, 2.9, scene=at_2x) == run(head_on, 2.9)


# ------------------------------------------------ real traffic replayed from the track caches (SPEC §12.47)
# Before §12.47 Part B raised 31 alarms in 7.4 min of accident-free C3902 + C3905 traffic
# (scripts/risk_replay.py). Each test below is one of those patterns rebuilt synthetically; the crash
# tests above and below must keep alarming early.


def test_a_rear_end_into_a_standing_queue_alarms_early() -> None:
    # 14 m/s, no braking, into a car standing at x = 64.6 m (contact when the gap closes, t ~ 2.4 s).
    scores = run(lambda t: [(30 + 14 * t, 45, 2), (64.6, 45, 2)], 2.3)
    assert max(s for t, s in scores if t < 1.0) < 0.1
    assert max(s for t, s in scores if t >= 1.4) >= 0.5  # at least ~1 s before contact


def test_arriving_at_the_queue_with_ordinary_braking_stays_calm() -> None:
    # C3902 0:01: 8.3 m/s, 10.8 m bumper gap to a stopped car, braking at 3.2 m/s^2, the acceleration
    # estimate lagging behind: needs 3.2 m/s^2 to stop, which drivers here do at every red.
    def follower(t: float) -> float:
        v = max(8.3 - 3.2 * t, 0.0)
        return 40 + (8.3**2 - v**2) / (2 * 3.2)

    scores = run(jittered(lambda t: [(follower(t), 45, 2), (40 + 4.6 + 10.8 + 4.6, 45, 2)], 0.05), 5)
    assert max(s for _, s in scores) < 0.2


def test_overtaking_a_bus_in_the_next_lane_stays_calm() -> None:
    # C3902 0:51: a car at 16 m/s passing a bus at 8 m/s, 3.4 m to the side (one lane): no conflict,
    # though the car closes on the bus fast and its footprint drifts 1 m toward the bus's.
    def boxes(t: float):
        drift = 0.5 * math.sin(3 * t)  # the tall bus's footprint wobbles across its lane
        return [(20 + 16 * t, 45.0, 2), (40 + 8 * t, 48.4 - 1.0 + drift, 5)]

    scores = run(boxes, 4)
    assert max(s for _, s in scores) < 0.2


def test_a_car_parked_in_a_parking_zone_is_not_an_obstacle() -> None:
    # C3902 0:01: a bus drives past (and pulls in behind) a car parked in the kerb lane.
    parking = Scene({**NO_LANES.layers, "parking_zones": [[[500, 400], [900, 400], [900, 480], [500, 480]]]})
    scores = run(lambda t: [(20 + 7 * t, 44, 5), (70, 44, 2)], 5, scene=parking)
    assert max(s for _, s in scores) < 0.1
    # the same standing car in a traffic lane is an obstacle: driving into it at 10 m/s alarms
    assert max(s for _, s in run(lambda t: [(20 + 10 * t, 44, 2), (70, 44, 2)], 4.4)) >= 0.5


def test_a_pedestrian_beside_a_car_is_not_in_its_path_but_one_stepping_in_is() -> None:
    # C3902 2:55: walking beside a moving car (1.5 m from its centre line) is not a conflict ...
    beside = run(lambda t: [(20 + 9 * t, 45, 2), (40 + 1.4 * t, 46.5, 0)], 3)
    assert max(s for _, s in beside) < 0.2
    # ... stepping into its path 15 m ahead of it is
    stepping = run(lambda t: [(20 + 9 * t, 45, 2), (41, 48 - 1.4 * t, 0)], 2.0)
    assert max(s for _, s in stepping) >= 0.5


def test_far_from_the_camera_jitter_is_not_measured() -> None:
    # C3902 4:13: the far end of an approach, where one 4K pixel is 10-24 cm of road. There two queued
    # cars and a car approaching at 12 m/s look like a collision course; they are not measured at all.
    far = Scene({**NO_LANES.layers, "image_size": [3840, 2160]})  # PX = 10 px/m -> 0.1 m per 4K pixel
    scores = run(jittered(lambda t: [(20 + 12 * t, 45, 2), (60, 45, 2)], 0.8), 2.5, scene=far)
    assert max(s for _, s in scores) < 0.1


def test_braking_is_read_only_where_it_can_be_measured() -> None:
    c = cfg()
    base = rows((1, "car", 50, 45, 8, 0, -7, 0)).assign(age=3.0)
    assert risk_features(base, SCENE, set(), c)["decel_max"] == pytest.approx(7.0 - c["decel_floor_mps2"])
    young = base.assign(age=0.8)  # a new box's first fits are noise
    assert risk_features(young, SCENE, set(), c)["decel_max"] == 0.0
    impossible = base.assign(ax=-30.0)  # no tyre brakes at 30 m/s^2: a re-emerging box, not braking
    assert risk_features(impossible, SCENE, set(), c)["decel_max"] == 0.0


def test_a_gap_that_does_not_shrink_is_not_a_conflict() -> None:
    # C3905 1:06: a car beside a turning truck. The measured speeds say it closes at 6.5 m/s, but the
    # measured gap stays 1 m: two merged boxes or a biased footprint, not an approach.
    c = cfg()
    memory: dict = {}
    frame = rows((1, "car", 20, 45, 9.9, 0, 0, 0), (2, "truck", 31.6, 45, 3.4, 0, 0, 0))
    for k in range(6):
        feats = risk_features(frame, SCENE, set(), c, memory, t=k / 10)
    assert math.isinf(feats["ttc_min"])
    # the same pair really closing (gap shrinking at the closing speed) is confirmed after 3 frames
    memory = {}
    for k in range(6):
        moving = frame.assign(X=[20 + 6.5 * k / 10, 31.6])
        feats = risk_features(moving, SCENE, set(), c, memory, t=k / 10)
    assert math.isfinite(feats["ttc_min"])
